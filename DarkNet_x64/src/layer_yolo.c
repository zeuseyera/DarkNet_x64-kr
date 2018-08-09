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

// ��δ� �����
layer make_yolo_layer( int batch
					, int ww		// �ʺ�
					, int hh		// ����
					, int nn		// ����ũ ����
					, int total		// ��Ŀ �� ����???
					, int *mask		// ����ũ��
					, int classes )	// �з�����(COCO: 80��, VOC: 20��)
{
	int ii;
	layer Lyr	= { 0 };
	Lyr.type	= YOLO;

	Lyr.n		= nn;		// ����ũ ����(3��: 3,4,5	�Ǵ� 1,2,3)
	// ��Ŀ �� ����: 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
	Lyr.total	= total;
	Lyr.batch	= batch;
	Lyr.w		= ww;
	Lyr.h		= hh;
	Lyr.c		= nn*(classes + 4 + 1);
	Lyr.out_w	= Lyr.w;
	Lyr.out_h	= Lyr.h;
	Lyr.out_c	= Lyr.c;
	Lyr.classes	= classes;	// �з�����
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
	Lyr.truths	= 90*(4 + 1);	// ��ǥ�� ����(���: 90*(4 + 1))
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
	fprintf( stderr, "��δ�\n" );
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
// ��������(���� ����, ���� �׸��� �ʺ� ���̿� ���� ��������)�� ä���� ��ȯ�Ѵ�
box get_yolo_box( float *x		// ������(��°�)
				, float *biases	// ���Ⱚ(��Ŀ �� ���� x 2)
				, int nn		// ����ũ ��(3,4,5 �Ǵ� 1,2,3)
				, int index		// ���� ����
				, int ii		// ����(��) ��ġ
				, int jj		// ����(��) ��ġ
				, int lw		// �� ���ʺ�
				, int lh		// �� ������
				, int ww		// �� ���ʺ�
				, int hh		// �� ������
				, int stride )	// ���� ���� ũ��
{
	box bb;

	bb.x = (ii + x[index + 0*stride]) / lw;
	bb.y = (jj + x[index + 1*stride]) / lh;
	bb.w = exp(  x[index + 2*stride] ) * biases[2*nn]   / ww;
	bb.h = exp(  x[index + 3*stride] ) * biases[2*nn+1] / hh;

	return bb;
}
// ���� ����
float delta_yolo_box( box truth
					, float *x
					, float *biases
					, int nn		// ����ũ ����
					, int index		// ���ڼ���
					, int ii		// ����ġ
					, int jj		// ����ġ
					, int lw		// �� ���ʺ�
					, int lh		// �� ������
					, int nw		// �� ���ʺ�
					, int nh		// �� ������
					, float *delta	// ������ �����
					, float scale	// ��ô
					, int stride )	// ���� ���� ũ��(��)
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
// �з��� ����
void delta_yolo_class( float *output
					, float *delta		// ������(����� �������� ������ �޸�)
					, int index			// �з�����
					, int class			// ��������
					, int classes		// �з�����
					, int stride		// �� ��
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
// ȭ�� ��ġ�� �ش��ϴ� ���ڼ����� �˾Ƴ���
static int entry_index( layer Lyr	// ��
					, int batch		// �縮����
					, int location	// ȭ����ġ
					, int entry )	// �Է��� ����
{
	int nn	= location / (Lyr.w*Lyr.h);	// ��
	int loc	= location % (Lyr.w*Lyr.h);	// ������

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
	// �縮 �ݺ�
	for ( bb=0; bb < Lyr.batch; ++bb )
	{	// ���� �ݺ�
		for ( jj=0; jj < Lyr.h; ++jj )
		{	// �ʺ� �ݺ�
			for ( ii=0; ii < Lyr.w; ++ii )
			{	// ��(����ũ��) �ݺ�
				for ( nn=0; nn < Lyr.n; ++nn )
				{
					int WiChi	= nn*Lyr.w*Lyr.h + jj*Lyr.w + ii;	// ȭ����ġ

					// ���� ������ �˾Ƴ���
					int box_index = entry_index( Lyr
											, bb	// �縮����
											//, nn*Lyr.w*Lyr.h + jj*Lyr.w + ii	// ȭ����ġ
											, WiChi	// ȭ����ġ
											, 0 );	// �Է��� ����
					// ������ ���ڸ� �����´�
					box pred = get_yolo_box( Lyr.output
										, Lyr.biases
										, Lyr.mask[nn]	// ����ũ ��
										, box_index		// ���� ����
										, ii			// ������ġ
										, jj			// ������ġ
										, Lyr.w			// �� ���ʺ�
										, Lyr.h			// �� ������
										, net.w			// �� ���ʺ�
										, net.h			// �� ������
										, Lyr.w*Lyr.h );// ���� ���� ũ��

					float	best_iou	= 0;	// ������ħ��
					int		best_t		= 0;	// ��������
					// ���� ������ �ݺ��ؼ� ...
					for ( tt=0; tt < Lyr.max_boxes; ++tt )
					{	// ��ǥ�� ���ڸ� �����´�
						box truth = float_to_box( net.truth + tt*(4 + 1) + bb*Lyr.truths, 1 );

						if ( !truth.x ) break;
						// ���� ��ħ ����(IOU: Intersection Over Union)
						// �� ������ �����ؼ� ��ģ�� ����ũ��� ���� ������ ���
						float iou = box_iou( pred, truth );

						if ( iou > best_iou )
						{
							best_iou	= iou;
							best_t		= tt;
						}
					}
					// ��ü ������ �˾Ƴ���
					int obj_index = entry_index( Lyr
											, bb	// �縮����
											//, nn*Lyr.w*Lyr.h + jj*Lyr.w + ii	// ȭ����ġ
											, WiChi	// ȭ����ġ
											, 4 );	// �Է��� ����

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
						// ���������� ã�´�
						int class = net.truth[best_t*(4 + 1) + bb*Lyr.truths + 4];

						if ( Lyr.map ) class = Lyr.map[class];	// ������������ ���������� ã�´�
						// �з� ������ �˾Ƴ���
						int class_index = entry_index( Lyr
													, bb		// �縮����
													//, nn*Lyr.w*Lyr.h + jj*Lyr.w + ii	// ȭ����ġ
													, WiChi		// ȭ����ġ
													, 4 + 1 );	// �Է��� ����
						// �з��� ������ ���
						delta_yolo_class( Lyr.output
										, Lyr.delta
										, class_index	// �з�����
										, class			// ��������
										, Lyr.classes	// �з�����
										, Lyr.w*Lyr.h	// ���� ���� ũ��(��)
										, 0 );
						// ��ǥ�� ���ڸ� �����´�
						box truth = float_to_box( net.truth + best_t*(4 + 1) + bb*Lyr.truths, 1 );
						// ���ڿ����� ���???
						delta_yolo_box( truth
									, Lyr.output
									, Lyr.biases
									, Lyr.mask[nn]			// ����ũ ����
									, box_index				// ���ڼ���
									, ii					// ����ġ
									, jj					// ����ġ
									, Lyr.w					// �� ���ʺ�
									, Lyr.h					// �� ������
									, net.w					// �� ���ʺ�
									, net.h					// �� ������
									, Lyr.delta				// ������ �����
									, (2-truth.w*truth.h)	// ��ô
									, Lyr.w*Lyr.h );		// ���� ���� ũ��(��)
					}
				}
			}
		}

		// ��� ���ڸ� �ݺ��Ѵ�
		for ( tt=0; tt < Lyr.max_boxes; ++tt )
		{
			// ��ǥ������ ũ�⸦ �����´�
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
// ��°��� ���ΰ����� ū ��°����� ��ȯ�Ѵ�
int yolo_num_detections( layer Lyr, float thresh )
{
	int ii, nn;
	int count = 0;

	// ��� �Է�ȭ���� �ݺ��Ѵ�
	for ( ii=0; ii < Lyr.w*Lyr.h; ++ii )
	{
		// ����(���)������ �ݺ�
		for ( nn=0; nn < Lyr.n; ++nn )
		{
			//	��°���ġ�� ����Ѵ�
			int obj_index  = entry_index( Lyr
										, 0					// �縮��
										, nn*Lyr.w*Lyr.h + ii	//
										, 4 );				//
			// ��°��� ���ΰ����� ū ��°��� ����Ѵ�
			if ( Lyr.output[obj_index] > thresh )
			{
				++count;
			}
		}
	}

	return count;
}
// ���� ��°�(�簢�迭, �̹���)�� ��,��� ���� ���ιٲ۰��� ���Ѵ��� ���� �����
void avg_flipped_yolo( layer Lyr )
{
	int ww, hh, nn, zz;
	float *flip = Lyr.output + Lyr.outputs;	// �� ��°����� �ǳʶٴ°�???, ���Ҵ� �޸����� ������ ���°�???
	// ������(��) �ݺ�
	for ( hh=0; hh < Lyr.h; ++hh )
	{
		// ���ʺ�(��)�� �ݸ� �ݺ�
		for ( ww=0; ww < Lyr.w/2; ++ww )
		{
			// �������� �ݺ�
			for ( nn=0; nn < Lyr.n; ++nn )
			{
				// �з����� �ݺ�( �� +4, +1 �� �ϴ°�??? )
				for ( zz=0; zz < Lyr.classes + 4 + 1; ++zz )
				{
					// ���� ������ġ
					//		 �з���ġ			+ ������ġ	 + ����ġ	  + ����ġ(�������� ����)
					int i1 = zz*Lyr.w*Lyr.h*Lyr.n + nn*Lyr.w*Lyr.h + hh*Lyr.w + ww;
					// ���� ������ġ
					//		 �з���ġ			+ ������ġ	 + ����ġ	  + ����ġ(�������� ����)
					int i2 = zz*Lyr.w*Lyr.h*Lyr.n + nn*Lyr.w*Lyr.h + hh*Lyr.w + (Lyr.w - ww - 1);

					float swap	= flip[i1];	// ���� ������ġ �� ��Ƶα�
					flip[i1]	= flip[i2];	// ���� ������ġ ���� ���� ������ġ ������ ����
					flip[i2]	= swap;		// ���� ������ġ ���� ��Ƶ�(���� ������ġ) ������ ����

					if ( zz == 0 )
					{
						flip[i1] = -flip[i1];	// ��ȣ �ٲ�
						flip[i2] = -flip[i2];	// ��ȣ �ٲ�
					}
				}
			}
		}
	}
	// ��°��� �ݺ�
	for ( nn=0; nn < Lyr.outputs; ++nn )
	{
		Lyr.output[nn] = ( Lyr.output[nn] + flip[nn] ) / 2.0f;
	}
}

int get_yolo_detections( layer Lyr	// ��
					, int ww		// �� ���ʺ�
					, int hh		// �� ������
					, int netw		// �� ���ʺ�
					, int neth		// �� ������
					, float thresh
					, int *map
					, int relative
					, detection *dets )
{
	int ii, jj, nn;
	float *predictions = Lyr.output;	// ������(�Ű�� ��°�)

	if ( Lyr.batch == 2 ) avg_flipped_yolo( Lyr );

	int count = 0;
	// ���ȭ�� ��ü�� �ݺ�
	for ( ii=0; ii < Lyr.w*Lyr.h; ++ii )
	{
		int row = ii / Lyr.w;	// ��
		int col = ii % Lyr.w;	// ��
		// �������� �ݺ�
		for ( nn=0; nn < Lyr.n; ++nn )
		{
			int obj_index  = entry_index( Lyr, 0, nn*Lyr.w*Lyr.h + ii, 4 );	// ��ü����
			float objectness = predictions[obj_index];					// ��ü����

			if ( objectness <= thresh ) continue;	//

			int box_index  = entry_index( Lyr, 0, nn*Lyr.w*Lyr.h + ii, 0 );	// ��������
			// ��λ��� ��������
			dets[count].bbox = get_yolo_box( predictions	// ��°�
										, Lyr.biases		// ���Ⱚ
										, Lyr.mask[nn]		// ��������(���Ⱚ����)
										, box_index			// ��������
										, col				// �� ��ġ
										, row				// �� ��ġ
										, Lyr.w				// �� ���ʺ�
										, Lyr.h				// �� ������
										, netw				// �� ���ʺ�
										, neth				// �� ������
										, Lyr.w*Lyr.h );	// ��
			dets[count].objectness = objectness;
			dets[count].classes = Lyr.classes;

			for ( jj=0; jj < Lyr.classes; ++jj )
			{
				int class_index = entry_index( Lyr, 0, nn*Lyr.w*Lyr.h + ii, 4 + 1 + jj );	// �з�����
				float prob = objectness*predictions[class_index];
				dets[count].prob[jj] = (prob > thresh) ? prob : 0;
			}
			++count;
		}
	}
	// ��λ��� ����
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

// �ϳ��� �������߰� �ּҸ� �̹����迭�� �ּҸ� ������
image pull_yolo_out( layer Lyr, int nn )
{
	int ww = Lyr.out_w;		// ��� �ʺ�
	int hh = Lyr.out_h;		// ��� ����
	int cc = 1;				// 
	int bo = ww*hh*nn;		// ������ ��
	return float_to_image( ww, hh, cc, Lyr.output + bo );
}
// �޸𸮸� �Ҵ��ϰ� �Ҵ��� �޸𸮿� ���߰��� �����Ѵ�
image *pull_yolo_image_out( layer Lyr )
{
	image *out = calloc( Lyr.out_c, sizeof( image ) );	// ������ ������ŭ �̹����޸� �Ҵ�

	int ii;
	// �����̹��� �ǰ��� �ݺ�
	for ( ii=0; ii < Lyr.out_c; ++ii )
	{
		// ��Ƶ� �޸𸮸� �Ҵ��ϰ�, �ڷᰪ�� �����ϰ�, ��Ƶ� �ּҸ� ��ȯ�Ѵ�
		out[ii] = copy_image( pull_yolo_out( Lyr, ii ) );
		normalize_image( out[ii] );	//�����Ǻ��� ���⸦ �ϸ� Ư¡ǥ���� �Ǵ°�???
	}

	//normalize_image_MuRi( out, Lyr.n );	//�̹������� ��ü�� �����Ѵ�

	return out;
}
// ��δ� ���°� �ð�ȭ
image *visualize_yolo_layer_output( layer Lyr, char *window, image *prev_out )
{
	// ������ ������ŭ �̹����޸𸮸� �Ҵ��ϰ� ��Ƶ� �ּҸ� �����Ѵ�
	image *single_out = pull_yolo_image_out( Lyr );
	// �̹����� ���簢������ �迭�� �����ϰ� ȭ�鿡 �����ش�
	show_images( single_out, Lyr.out_c, window );

	char buff[256];

	sprintf_s( buff, 256, "%s: Output", window );

	return single_out;
}

