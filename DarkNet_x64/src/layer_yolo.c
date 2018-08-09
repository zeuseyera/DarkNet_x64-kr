#include "layer_yolo.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

// 욜로단 만들기
layer make_yolo_layer( int batch
					, int ww		// 너비
					, int hh		// 높이
					, int nn		// 마스크 개수
					, int total		// 앵커 쌍 개수???
					, int *mask		// 마스크값
					, int classes )	// 분류개수(COCO: 80개, VOC: 20개)
{
	int ii;
	layer Lyr	= { 0 };
	Lyr.type	= YOLO;

	Lyr.n		= nn;		// 마스크 개수(3개: 3,4,5	또는 1,2,3)
	// 앵커 쌍 개수: 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
	Lyr.total	= total;
	Lyr.batch	= batch;
	Lyr.w		= ww;
	Lyr.h		= hh;
	Lyr.c		= nn*(classes + 4 + 1);
	Lyr.out_w	= Lyr.w;
	Lyr.out_h	= Lyr.h;
	Lyr.out_c	= Lyr.c;
	Lyr.classes	= classes;	// 분류개수
	Lyr.cost	= calloc( 1, sizeof( float ) );
	Lyr.biases	= calloc( total*2, sizeof( float ) );

	if ( mask ) Lyr.mask = mask;
	else
	{
		Lyr.mask = calloc( nn, sizeof( int ) );

		for ( ii=0; ii < nn; ++ii )
		{
			Lyr.mask[ii] = ii;
		}
	}

	Lyr.bias_updates = calloc( nn*2, sizeof( float ) );
	Lyr.outputs	= ww*hh*nn*(classes + 4 + 1);
	Lyr.inputs	= Lyr.outputs;
	Lyr.truths	= 90*(4 + 1);	// 목표값 개수(욜로: 90*(4 + 1))
	Lyr.delta	= calloc( batch*Lyr.outputs, sizeof( float ) );
	Lyr.output	= calloc( batch*Lyr.outputs, sizeof( float ) );

	for ( ii=0; ii < total*2; ++ii )
	{
		Lyr.biases[ii] = 0.5f;
	}

	Lyr.forward		= forward_yolo_layer;
	Lyr.backward	= backward_yolo_layer;
	Lyr.BoJa_NaOnGab = visualize_yolo_layer_output;

	#ifdef GPU
	Lyr.forward_gpu		= forward_yolo_layer_gpu;
	Lyr.backward_gpu	= backward_yolo_layer_gpu;
	Lyr.output_gpu		= cuda_make_array( Lyr.output, batch*Lyr.outputs );
	Lyr.delta_gpu		= cuda_make_array( Lyr.delta, batch*Lyr.outputs );
	#endif

	//fprintf( stderr, "yolo\n" );
	fprintf( stderr, "욜로단\n" );
	srand( 0 );

	return Lyr;
}

void resize_yolo_layer( layer *Lyr, int ww, int hh )
{
	Lyr->w = ww;
	Lyr->h = hh;

	Lyr->outputs	= hh*ww*Lyr->n*(Lyr->classes + 4 + 1);
	Lyr->inputs		= Lyr->outputs;

	Lyr->output		= realloc( Lyr->output, Lyr->batch*Lyr->outputs*sizeof( float ) );
	Lyr->delta		= realloc( Lyr->delta, Lyr->batch*Lyr->outputs*sizeof( float ) );

	#ifdef GPU
	cuda_free( Lyr->delta_gpu );
	cuda_free( Lyr->output_gpu );

	Lyr->delta_gpu	= cuda_make_array( Lyr->delta, Lyr->batch*Lyr->outputs );
	Lyr->output_gpu	= cuda_make_array( Lyr->output, Lyr->batch*Lyr->outputs );
	#endif
}
// 상자정보(시작 가로, 세로 그리고 너비 높이에 대한 비율정보)를 채워서 반환한다
box get_yolo_box( float *x		// 예측값(출력값)
				, float *biases	// 편향값(앵커 쌍 개수 x 2)
				, int nn		// 마스크 값(3,4,5 또는 1,2,3)
				, int index		// 상자 순번
				, int ii		// 가로(열) 위치
				, int jj		// 세로(행) 위치
				, int lw		// 단 사비너비
				, int lh		// 단 사비높이
				, int ww		// 망 사비너비
				, int hh		// 망 사비높이
				, int stride )	// 단의 한판 크기
{
	box bb;

	bb.x = (ii + x[index + 0*stride]) / lw;
	bb.y = (jj + x[index + 1*stride]) / lh;
	bb.w = exp(  x[index + 2*stride] ) * biases[2*nn]   / ww;
	bb.h = exp(  x[index + 3*stride] ) * biases[2*nn+1] / hh;

	return bb;
}
// 상자 오차
float delta_yolo_box( box truth
					, float *x
					, float *biases
					, int nn		// 마스크 순번
					, int index		// 상자순번
					, int ii		// 열위치
					, int jj		// 행위치
					, int lw		// 단 사비너비
					, int lh		// 단 사비높이
					, int nw		// 망 사비너비
					, int nh		// 망 사비높이
					, float *delta	// 오차값 저장소
					, float scale	// 축척
					, int stride )	// 단의 한판 크기(보)
{
	box pred = get_yolo_box( x, biases, nn, index, ii, jj, lw, lh, nw, nh, stride );
	float iou = box_iou( pred, truth );

	float tx = (truth.x*lw - ii);
	float ty = (truth.y*lh - jj);
	float tw = log( truth.w*nw / biases[2*nn] );
	float th = log( truth.h*nh / biases[2*nn + 1] );

	delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
	delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
	delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
	delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
	return iou;
}
// 분류종 오차
void delta_yolo_class( float *output
					, float *delta		// 오차값(계산한 오차값을 저장할 메모리)
					, int index			// 분류순번
					, int class			// 계층순번
					, int classes		// 분류개수
					, int stride		// 판 보
					, float *avg_cat )
{
	int nn;

	if ( delta[index] )
	{
		delta[index + stride*class] = 1 - output[index + stride*class];

		if ( avg_cat ) *avg_cat += output[index + stride*class];
		return;
	}

	for ( nn=0; nn < classes; ++nn )
	{
		delta[index + stride*nn] = ((nn == class) ? 1 : 0) - output[index + stride*nn];

		if ( nn == class && avg_cat ) *avg_cat += output[index + stride*nn];
	}
}
// 화소 위치에 해당하는 상자순번을 알아낸다
static int entry_index( layer Lyr	// 단
					, int batch		// 사리순번
					, int location	// 화소위치
					, int entry )	// 입력판 순번
{
	int nn	= location / (Lyr.w*Lyr.h);	// 몫
	int loc	= location % (Lyr.w*Lyr.h);	// 나머지

	return batch	* Lyr.outputs
		 + nn		* Lyr.w * Lyr.h * ( 4 + Lyr.classes + 1 )
		 + entry	* Lyr.w * Lyr.h
		 + loc;
}

void forward_yolo_layer( const layer Lyr, network net )
{
	int ii, jj, bb, tt, nn;
	memcpy( Lyr.output, net.input, Lyr.outputs*Lyr.batch*sizeof( float ) );

	#ifndef GPU
	for ( bb=0; bb < Lyr.batch; ++bb )
	{
		for ( nn=0; nn < Lyr.n; ++nn )
		{
			int index	= entry_index( Lyr, bb, nn*Lyr.w*Lyr.h, 0 );
			activate_array( Lyr.output + index, 2*Lyr.w*Lyr.h, LOGISTIC );
			index		= entry_index( Lyr, bb, nn*Lyr.w*Lyr.h, 4 );
			activate_array( Lyr.output + index, (1+Lyr.classes)*Lyr.w*Lyr.h, LOGISTIC );
		}
	}
	#endif

	memset( Lyr.delta, 0, Lyr.outputs * Lyr.batch * sizeof( float ) );
	if ( !net.train ) return;

	float avg_iou	= 0;
	float recall	= 0;
	float recall75	= 0;
	float avg_cat	= 0;
	float avg_obj	= 0;
	float avg_anyobj = 0;
	int count		= 0;
	int class_count	= 0;
	*(Lyr.cost)		= 0;
	// 사리 반복
	for ( bb=0; bb < Lyr.batch; ++bb )
	{	// 높이 반복
		for ( jj=0; jj < Lyr.h; ++jj )
		{	// 너비 반복
			for ( ii=0; ii < Lyr.w; ++ii )
			{	// 판(마스크판) 반복
				for ( nn=0; nn < Lyr.n; ++nn )
				{
					int WiChi	= nn*Lyr.w*Lyr.h + jj*Lyr.w + ii;	// 화소위치

					// 상자 순번을 알아낸다
					int box_index = entry_index( Lyr
											, bb	// 사리순번
											//, nn*Lyr.w*Lyr.h + jj*Lyr.w + ii	// 화소위치
											, WiChi	// 화소위치
											, 0 );	// 입력판 순번
					// 예측한 상자를 가져온다
					box pred = get_yolo_box( Lyr.output
										, Lyr.biases
										, Lyr.mask[nn]	// 마스크 값
										, box_index		// 상자 순번
										, ii			// 가로위치
										, jj			// 세로위치
										, Lyr.w			// 단 사비너비
										, Lyr.h			// 단 사비높이
										, net.w			// 망 사비너비
										, net.h			// 망 사비높이
										, Lyr.w*Lyr.h );// 단의 한판 크기

					float	best_iou	= 0;	// 최적겹침비
					int		best_t		= 0;	// 최적상자
					// 상자 개수를 반복해서 ...
					for ( tt=0; tt < Lyr.max_boxes; ++tt )
					{	// 목표값 상자를 가져온다
						box truth = float_to_box( net.truth + tt*(4 + 1) + bb*Lyr.truths, 1 );

						if ( !truth.x ) break;
						// 교차 겹침 결합(IOU: Intersection Over Union)
						// 두 상자의 교차해서 겹친를 결합크기로 나눈 비율을 계산
						float iou = box_iou( pred, truth );

						if ( iou > best_iou )
						{
							best_iou	= iou;
							best_t		= tt;
						}
					}
					// 개체 순번을 알아낸다
					int obj_index = entry_index( Lyr
											, bb	// 사리순번
											//, nn*Lyr.w*Lyr.h + jj*Lyr.w + ii	// 화소위치
											, WiChi	// 화소위치
											, 4 );	// 입력판 순번

					avg_anyobj += Lyr.output[obj_index];
					Lyr.delta[obj_index] = 0 - Lyr.output[obj_index];

					if ( best_iou > Lyr.ignore_thresh )
					{
						Lyr.delta[obj_index] = 0;
					}
					// 
					if ( best_iou > Lyr.truth_thresh )
					{
						Lyr.delta[obj_index] = 1 - Lyr.output[obj_index];
						// 계층순번을 찾는다
						int class = net.truth[best_t*(4 + 1) + bb*Lyr.truths + 4];

						if ( Lyr.map ) class = Lyr.map[class];	// 계층지도에서 계층순번을 찾는다
						// 분류 순번을 알아낸다
						int class_index = entry_index( Lyr
													, bb		// 사리순번
													//, nn*Lyr.w*Lyr.h + jj*Lyr.w + ii	// 화소위치
													, WiChi		// 화소위치
													, 4 + 1 );	// 입력판 순번
						// 분류종 오차를 계산
						delta_yolo_class( Lyr.output
										, Lyr.delta
										, class_index	// 분류순번
										, class			// 계층순번
										, Lyr.classes	// 분류개수
										, Lyr.w*Lyr.h	// 단의 한판 크기(보)
										, 0 );
						// 목표값 상자를 가져온다
						box truth = float_to_box( net.truth + best_t*(4 + 1) + bb*Lyr.truths, 1 );
						// 상자오차를 계산???
						delta_yolo_box( truth
									, Lyr.output
									, Lyr.biases
									, Lyr.mask[nn]			// 마스크 순번
									, box_index				// 상자순번
									, ii					// 열위치
									, jj					// 행위치
									, Lyr.w					// 단 사비너비
									, Lyr.h					// 단 사비높이
									, net.w					// 망 사비너비
									, net.h					// 망 사비높이
									, Lyr.delta				// 오차값 저장소
									, (2-truth.w*truth.h)	// 축척
									, Lyr.w*Lyr.h );		// 단의 한판 크기(보)
					}
				}
			}
		}

		// 모든 상자를 반복한다
		for ( tt=0; tt < Lyr.max_boxes; ++tt )
		{
			// 목표값상자 크기를 가져온다
			box truth = float_to_box( net.truth + tt*(4 + 1) + bb*Lyr.truths, 1 );

			if ( !truth.x ) break;

			float best_iou = 0;
			int best_n	= 0;
			ii			= (truth.x * Lyr.w);
			jj			= (truth.y * Lyr.h);

			box truth_shift	= truth;

			truth_shift.x = truth_shift.y = 0;

			for ( nn=0; nn < Lyr.total; ++nn )
			{
				box pred	= { 0 };
				pred.w		= Lyr.biases[2*nn]/net.w;
				pred.h		= Lyr.biases[2*nn+1]/net.h;
				float iou	= box_iou( pred, truth_shift );

				if ( iou > best_iou )
				{
					best_iou	= iou;
					best_n		= nn;
				}
			}

			int mask_n = int_index( Lyr.mask, best_n, Lyr.n );

			if ( mask_n >= 0 )
			{
				int box_index = entry_index( Lyr
										, bb
										, mask_n*Lyr.w*Lyr.h + jj*Lyr.w + ii
										, 0 );
				float iou = delta_yolo_box( truth
										, Lyr.output
										, Lyr.biases
										, best_n
										, box_index
										, ii
										, jj
										, Lyr.w
										, Lyr.h
										, net.w
										, net.h
										, Lyr.delta
										, (2-truth.w*truth.h)
										, Lyr.w*Lyr.h );

				int obj_index = entry_index( Lyr
										, bb
										, mask_n*Lyr.w*Lyr.h + jj*Lyr.w + ii
										, 4 );

				avg_obj += Lyr.output[obj_index];
				Lyr.delta[obj_index] = 1 - Lyr.output[obj_index];

				int class = net.truth[tt*(4 + 1) + bb*Lyr.truths + 4];
				if ( Lyr.map ) class = Lyr.map[class];
				int class_index = entry_index( Lyr
										, bb
										, mask_n*Lyr.w*Lyr.h + jj*Lyr.w + ii
										, 4 + 1 );

				delta_yolo_class( Lyr.output
								, Lyr.delta
								, class_index
								, class
								, Lyr.classes
								, Lyr.w*Lyr.h
								, &avg_cat );

				++count;
				++class_count;

				if ( iou > 0.5f )	recall		+= 1;
				if ( iou > 0.75f )	recall75	+= 1;

				avg_iou += iou;
			}
		}
	}

	*(Lyr.cost) = pow( mag_array( Lyr.delta, Lyr.outputs * Lyr.batch ), 2 );

	printf( "Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n"
		, net.index
		, avg_iou / count
		, avg_cat / class_count
		, avg_obj / count
		, avg_anyobj / (Lyr.w*Lyr.h*Lyr.n*Lyr.batch)
		, recall / count
		, recall75 / count
		, count );
}

void backward_yolo_layer( const layer Lyr, network net )
{
	axpy_cpu( Lyr.batch*Lyr.inputs, 1, Lyr.delta, 1, net.delta, 1 );
}

void correct_yolo_boxes( detection *dets, int nn, int ww, int hh, int netw, int neth, int relative )
{
	int ii;
	int new_w=0;
	int new_h=0;
	if ( ((float)netw/ww) < ((float)neth/hh) )
	{
		new_w = netw;
		new_h = (hh * netw)/ww;
	}
	else
	{
		new_h = neth;
		new_w = (ww * neth)/hh;
	}

	for ( ii=0; ii < nn; ++ii )
	{
		box b = dets[ii].bbox;
		b.x =  (b.x - (netw - new_w)/2.0f/netw) / ((float)new_w/netw);
		b.y =  (b.y - (neth - new_h)/2.0f/neth) / ((float)new_h/neth);
		b.w *= (float)netw/new_w;
		b.h *= (float)neth/new_h;
		if ( !relative )
		{
			b.x *= ww;
			b.w *= ww;
			b.y *= hh;
			b.h *= hh;
		}
		dets[ii].bbox = b;
	}
}
// 출력값이 문턱값보다 큰 출력개수를 반환한다
int yolo_num_detections( layer Lyr, float thresh )
{
	int ii, nn;
	int count = 0;

	// 모든 입력화소을 반복한다
	for ( ii=0; ii < Lyr.w*Lyr.h; ++ii )
	{
		// 포집(출력)개수를 반복
		for ( nn=0; nn < Lyr.n; ++nn )
		{
			//	출력값위치를 계산한다
			int obj_index  = entry_index( Lyr
										, 0					// 사리수
										, nn*Lyr.w*Lyr.h + ii	//
										, 4 );				//
			// 출력값이 문턱값보다 큰 출력개수 계수한다
			if ( Lyr.output[obj_index] > thresh )
			{
				++count;
			}
		}
	}

	return count;
}
// 단의 출력값(사각배열, 이미지)과 좌,우로 값을 서로바꾼값을 더한다음 값을 평균함
void avg_flipped_yolo( layer Lyr )
{
	int ww, hh, nn, zz;
	float *flip = Lyr.output + Lyr.outputs;	// 왜 출력개수를 건너뛰는가???, 미할당 메모리접근 오류가 없는가???
	// 사비높이(행) 반복
	for ( hh=0; hh < Lyr.h; ++hh )
	{
		// 사비너비(열)의 반만 반복
		for ( ww=0; ww < Lyr.w/2; ++ww )
		{
			// 포집개수 반복
			for ( nn=0; nn < Lyr.n; ++nn )
			{
				// 분류개수 반복( 왜 +4, +1 을 하는가??? )
				for ( zz=0; zz < Lyr.classes + 4 + 1; ++zz )
				{
					// 좌측 지적위치
					//		 분류위치			+ 포집위치	 + 행위치	  + 열위치(좌측끝이 시작)
					int i1 = zz*Lyr.w*Lyr.h*Lyr.n + nn*Lyr.w*Lyr.h + hh*Lyr.w + ww;
					// 우측 지적위치
					//		 분류위치			+ 포집위치	 + 행위치	  + 열위치(우측끝이 시작)
					int i2 = zz*Lyr.w*Lyr.h*Lyr.n + nn*Lyr.w*Lyr.h + hh*Lyr.w + (Lyr.w - ww - 1);

					float swap	= flip[i1];	// 좌측 지적위치 값 담아두기
					flip[i1]	= flip[i2];	// 좌측 지적위치 값을 우측 지적위치 값으로 변경
					flip[i2]	= swap;		// 우측 지적위치 값을 담아둔(좌측 지적위치) 값으로 변경

					if ( zz == 0 )
					{
						flip[i1] = -flip[i1];	// 부호 바꿈
						flip[i2] = -flip[i2];	// 부호 바꿈
					}
				}
			}
		}
	}
	// 출력개수 반복
	for ( nn=0; nn < Lyr.outputs; ++nn )
	{
		Lyr.output[nn] = ( Lyr.output[nn] + flip[nn] ) / 2.0f;
	}
}

int get_yolo_detections( layer Lyr	// 단
					, int ww		// 단 사비너비
					, int hh		// 단 사비높이
					, int netw		// 망 사비너비
					, int neth		// 망 사비높이
					, float thresh
					, int *map
					, int relative
					, detection *dets )
{
	int ii, jj, nn;
	float *predictions = Lyr.output;	// 예측값(신경망 출력값)

	if ( Lyr.batch == 2 ) avg_flipped_yolo( Lyr );

	int count = 0;
	// 사비화소 전체를 반복
	for ( ii=0; ii < Lyr.w*Lyr.h; ++ii )
	{
		int row = ii / Lyr.w;	// 행
		int col = ii % Lyr.w;	// 열
		// 포집개수 반복
		for ( nn=0; nn < Lyr.n; ++nn )
		{
			int obj_index  = entry_index( Lyr, 0, nn*Lyr.w*Lyr.h + ii, 4 );	// 개체지적
			float objectness = predictions[obj_index];					// 개체상태

			if ( objectness <= thresh ) continue;	//

			int box_index  = entry_index( Lyr, 0, nn*Lyr.w*Lyr.h + ii, 0 );	// 상자지적
			// 욜로상자 가져오기
			dets[count].bbox = get_yolo_box( predictions	// 출력값
										, Lyr.biases		// 편향값
										, Lyr.mask[nn]		// 포집순번(편향값순번)
										, box_index			// 상자지적
										, col				// 열 위치
										, row				// 행 위치
										, Lyr.w				// 단 사비너비
										, Lyr.h				// 단 사비높이
										, netw				// 망 사비너비
										, neth				// 망 사비높이
										, Lyr.w*Lyr.h );	// 보
			dets[count].objectness = objectness;
			dets[count].classes = Lyr.classes;

			for ( jj=0; jj < Lyr.classes; ++jj )
			{
				int class_index = entry_index( Lyr, 0, nn*Lyr.w*Lyr.h + ii, 4 + 1 + jj );	// 분류지적
				float prob = objectness*predictions[class_index];
				dets[count].prob[jj] = (prob > thresh) ? prob : 0;
			}
			++count;
		}
	}
	// 욜로상자 보정
	correct_yolo_boxes( dets, count, ww, hh, netw, neth, relative );
	return count;
}

#ifdef GPU
void forward_yolo_layer_gpu( const layer Lyr, network net )
{
	copy_gpu( Lyr.batch*Lyr.inputs, net.input_gpu, 1, Lyr.output_gpu, 1 );

	int bb, nn;
	for ( bb=0; bb < Lyr.batch; ++bb )
	{
		for ( nn=0; nn < Lyr.n; ++nn )
		{
			int index = entry_index( Lyr, bb, nn*Lyr.w*Lyr.h, 0 );
			activate_array_gpu( Lyr.output_gpu + index, 2*Lyr.w*Lyr.h, LOGISTIC );
			index = entry_index( Lyr, bb, nn*Lyr.w*Lyr.h, 4 );
			activate_array_gpu( Lyr.output_gpu + index, (1+Lyr.classes)*Lyr.w*Lyr.h, LOGISTIC );
		}
	}

	if ( !net.train || Lyr.onlyforward )
	{
		cuda_pull_array( Lyr.output_gpu, Lyr.output, Lyr.batch*Lyr.outputs );
		return;
	}

	cuda_pull_array( Lyr.output_gpu, net.input, Lyr.batch*Lyr.inputs );
	forward_yolo_layer( Lyr, net );
	cuda_push_array( Lyr.delta_gpu, Lyr.delta, Lyr.batch*Lyr.outputs );
}

void backward_yolo_layer_gpu( const layer Lyr, network net )
{
	axpy_gpu( Lyr.batch*Lyr.inputs, 1, Lyr.delta_gpu, 1, net.delta_gpu, 1 );
}
#endif

// 하나의 포집가중값 주소를 이미지배열에 주소를 복사함
image pull_yolo_out( layer Lyr, int nn )
{
	int ww = Lyr.out_w;		// 출력 너비
	int hh = Lyr.out_h;		// 출력 높이
	int cc = 1;				// 
	int bo = ww*hh*nn;		// 출력장수 보
	return float_to_image( ww, hh, cc, Lyr.output + bo );
}
// 메모리를 할당하고 할당한 메모리에 가중값을 복사한다
image *pull_yolo_image_out( layer Lyr )
{
	image *out = calloc( Lyr.out_c, sizeof( image ) );	// 포집판 개수만큼 이미지메모리 할당

	int ii;
	// 나온이미지 판개수 반복
	for ( ii=0; ii < Lyr.out_c; ++ii )
	{
		// 담아둘 메모리를 할당하고, 자료값을 복사하고, 담아둔 주소를 반환한다
		out[ii] = copy_image( pull_yolo_out( Lyr, ii ) );
		normalize_image( out[ii] );	//포집판별로 고르기를 하면 특징표현이 되는가???
	}

	//normalize_image_MuRi( out, Lyr.n );	//이미지무리 전체를 고르기한다

	return out;
}
// 욜로단 나온값 시각화
image *visualize_yolo_layer_output( layer Lyr, char *window, image *prev_out )
{
	// 포집판 개수만큼 이미지메모리를 할당하고 담아둔 주소를 복사한다
	image *single_out = pull_yolo_image_out( Lyr );
	// 이미지를 정사각형으로 배열을 조정하고 화면에 보여준다
	show_images( single_out, Lyr.out_c, window );

	char buff[256];

	sprintf_s( buff, 256, "%s: Output", window );

	return single_out;
}

