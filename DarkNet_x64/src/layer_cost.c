#include "layer_cost.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

COST_TYPE get_cost_type( char *s )
{
	if ( strcmp( s, "seg"		)==0 )	return SEG;
	if ( strcmp( s, "sse"		)==0 )	return SSE;
	if ( strcmp( s, "masked"	)==0 )	return MASKED;
	if ( strcmp( s, "smooth"	)==0 )	return SMOOTH;
	if ( strcmp( s, "L1"		)==0 )	return L1;
	if ( strcmp( s, "wgan"		)==0 )	return WGAN;

	//fprintf( stderr, "Couldn't find cost type %s, going with SSE\n", s );	//  [7/11/2018 jobs]
	fprintf( stderr, "%s 유형 cost를 찾을수 없음, SSE로 진행...\n", s );	//  [7/11/2018 jobs]

	return SSE;
}

char *get_cost_string( COST_TYPE a )
{
	switch ( a )
	{
		case SEG:
			return "seg";
		case SSE:
			return "sse";
		case MASKED:
			return "masked";
		case SMOOTH:
			return "smooth";
		case L1:
			return "L1";
		case WGAN:
			return "wgan";
	}
	return "sse";
}

cost_layer make_cost_layer( int batch, int inputs, COST_TYPE cost_type, float scale )
{
	fprintf( stderr, "cost                                           %4d\n", inputs );
	cost_layer Lyr = { 0 };
	Lyr.type = COST;

	Lyr.scale		= scale;
	Lyr.batch		= batch;
	Lyr.inputs		= inputs;
	Lyr.outputs		= inputs;
	Lyr.cost_type	= cost_type;
	Lyr.delta		= calloc( inputs*batch, sizeof( float ) );
	Lyr.output		= calloc( inputs*batch, sizeof( float ) );
	Lyr.cost		= calloc( 1, sizeof( float ) );

	Lyr.forward		= forward_cost_layer;
	Lyr.backward	= backward_cost_layer;
	Lyr.BoJa_NaOnGab = visualize_cost_layer_output;

	#ifdef GPU
	Lyr.forward_gpu		= forward_cost_layer_gpu;
	Lyr.backward_gpu	= backward_cost_layer_gpu;

	Lyr.delta_gpu	= cuda_make_array( Lyr.output, inputs*batch );
	Lyr.output_gpu	= cuda_make_array( Lyr.delta, inputs*batch );
	#endif

	return Lyr;
}

void resize_cost_layer( cost_layer *Lyr, int inputs )
{
	Lyr->inputs	= inputs;
	Lyr->outputs = inputs;
	Lyr->delta	= realloc( Lyr->delta, inputs*Lyr->batch*sizeof( float ) );
	Lyr->output	= realloc( Lyr->output, inputs*Lyr->batch*sizeof( float ) );

	#ifdef GPU
	cuda_free( Lyr->delta_gpu );
	cuda_free( Lyr->output_gpu );
	Lyr->delta_gpu	= cuda_make_array( Lyr->delta, inputs*Lyr->batch );
	Lyr->output_gpu	= cuda_make_array( Lyr->output, inputs*Lyr->batch );
	#endif
}

void forward_cost_layer( cost_layer Lyr, network net )
{
	if ( !net.truth )
		return;	// 목표값이 없으면

	if ( Lyr.cost_type == MASKED )
	{
		int ii;
		for ( ii = 0; ii < Lyr.batch*Lyr.inputs; ++ii )
		{
			if ( net.truth[ii] == SECRET_NUM ) net.input[ii] = SECRET_NUM;
		}
	}

	if ( Lyr.cost_type == SMOOTH )
	{
		smooth_l1_cpu( Lyr.batch*Lyr.inputs, net.input, net.truth, Lyr.delta, Lyr.output );
	}
	else if ( Lyr.cost_type == L1 )
	{	// 오차는 -1, +1 로 계산, 손실은 양수값으로 변환
		l1_cpu( Lyr.batch*Lyr.inputs, net.input, net.truth, Lyr.delta, Lyr.output );
	}
	else
	{	// 출력단의 출력값과 목표값으로 각 출력오차(편차)와 SSE손실을 계산한다
		l2_cpu( Lyr.batch*Lyr.inputs, net.input, net.truth, Lyr.delta, Lyr.output );
	}

	Lyr.cost[0] = sum_array( Lyr.output, Lyr.batch*Lyr.inputs );
}

void backward_cost_layer( const cost_layer Lyr, network net )
{
	axpy_cpu( Lyr.batch*Lyr.inputs, Lyr.scale, Lyr.delta, 1, net.delta, 1 );
}

#ifdef GPU

void pull_cost_layer( cost_layer Lyr )
{
	cuda_pull_array( Lyr.delta_gpu, Lyr.delta, Lyr.batch*Lyr.inputs );
}

void push_cost_layer( cost_layer Lyr )
{
	cuda_push_array( Lyr.delta_gpu, Lyr.delta, Lyr.batch*Lyr.inputs );
}

int float_abs_compare ( const void * a, const void * b )
{
	float fa = *(const float*)a;
	if ( fa < 0 ) fa = -fa;

	float fb = *(const float*)b;
	if ( fb < 0 ) fb = -fb;

	return (fa > fb) - (fa < fb);
}

void forward_cost_layer_gpu( cost_layer Lyr, network net )
{
	if ( !net.truth )
		return;	// 목표값이 없으면

	if ( Lyr.smooth )
	{
		scal_gpu( Lyr.batch*Lyr.inputs, (1-Lyr.smooth), net.truth_gpu, 1 );
		add_gpu( Lyr.batch*Lyr.inputs, Lyr.smooth * 1.0f/Lyr.inputs, net.truth_gpu, 1 );
	}

	if ( Lyr.cost_type == SMOOTH )
	{
		smooth_l1_gpu( Lyr.batch*Lyr.inputs, net.input_gpu, net.truth_gpu, Lyr.delta_gpu, Lyr.output_gpu );
	}
	else if ( Lyr.cost_type == L1 )
	{	// 오차는 -1, +1 로 계산, 손실은 양수값으로 변환
		l1_gpu( Lyr.batch*Lyr.inputs, net.input_gpu, net.truth_gpu, Lyr.delta_gpu, Lyr.output_gpu );
	}
	else if ( Lyr.cost_type == WGAN )
	{
		wgan_gpu( Lyr.batch*Lyr.inputs, net.input_gpu, net.truth_gpu, Lyr.delta_gpu, Lyr.output_gpu );
	}
	else
	{	// 출력단의 출력값과 목표값으로 각 출력오차(편차)와 SSE손실을 계산한다
		l2_gpu( Lyr.batch*Lyr.inputs, net.input_gpu, net.truth_gpu, Lyr.delta_gpu, Lyr.output_gpu );
	}

	if ( Lyr.cost_type == SEG && Lyr.noobject_scale != 1 )
	{
		scale_mask_gpu( Lyr.batch*Lyr.inputs, Lyr.delta_gpu, 0, net.truth_gpu, Lyr.noobject_scale );
		scale_mask_gpu( Lyr.batch*Lyr.inputs, Lyr.output_gpu, 0, net.truth_gpu, Lyr.noobject_scale );
	}

	if ( Lyr.cost_type == MASKED )
	{
		mask_gpu( Lyr.batch*Lyr.inputs, net.delta_gpu, SECRET_NUM, net.truth_gpu, 0 );
	}

	if ( Lyr.ratio )
	{
		cuda_pull_array( Lyr.delta_gpu, Lyr.delta, Lyr.batch*Lyr.inputs );
		qsort( Lyr.delta, Lyr.batch*Lyr.inputs, sizeof( float ), float_abs_compare );

		int nn			= (1-Lyr.ratio) * Lyr.batch*Lyr.inputs;
		float thresh	= Lyr.delta[nn];
		thresh			= 0;

		printf( "%f\n", thresh );
		supp_gpu( Lyr.batch*Lyr.inputs, thresh, Lyr.delta_gpu, 1 );
	}

	if ( Lyr.thresh )
	{
		supp_gpu( Lyr.batch*Lyr.inputs, Lyr.thresh*1./Lyr.inputs, Lyr.delta_gpu, 1 );
	}

	cuda_pull_array( Lyr.output_gpu, Lyr.output, Lyr.batch*Lyr.inputs );
	Lyr.cost[0] = sum_array( Lyr.output, Lyr.batch*Lyr.inputs );
}

void backward_cost_layer_gpu( const cost_layer Lyr, network net )
{
	axpy_gpu( Lyr.batch*Lyr.inputs, Lyr.scale, Lyr.delta_gpu, 1, net.delta_gpu, 1 );
}
#endif

// 하나의 나온값 주소를 이미지배열에 주소를 복사함
image pull_cost_out( cost_layer Lyr, int nn )
{
	int ww = 1;		// 출력 너비
	int hh = 1;		// 출력 높이
	int cc = 1;				// 
	int bo = ww*hh*nn;		// 출력장수 보
	return float_to_image( ww, hh, cc, Lyr.output + bo );
}
// 메모리를 할당하고 할당한 메모리에 나온값을 복사한다
image *pull_cost_image_out( cost_layer Lyr )
{
	image *out = calloc( Lyr.outputs, sizeof( image ) );	// 포집판 개수만큼 이미지메모리 할당

	int ii;
	// 나온이미지 판개수 반복
	for ( ii=0; ii < Lyr.outputs; ++ii )
	{
		// 담아둘 메모리를 할당하고, 자료값을 복사하고, 담아둔 주소를 반환한다
		out[ii] = copy_image( pull_cost_out( Lyr, ii ) );
	}

	normalize_image_MuRi( out, Lyr.outputs );	//이미지무리 전체를 고르기한다

	return out;
}
// 활성층 나온값 시각화
image *visualize_cost_layer_output( cost_layer Lyr, char *window, image *prev_out )
{
	// 포집판 개수만큼 이미지메모리를 할당하고 담아둔 주소를 복사한다
	image *single_out = pull_cost_image_out( Lyr );
	// 이미지를 정사각형으로 배열을 조정하고 화면에 보여준다
	show_images( single_out, Lyr.outputs, window );

	char buff[256];

	sprintf_s( buff, 256, "%s: Output", window );

	return single_out;
}
