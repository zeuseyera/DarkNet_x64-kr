#include "layer_connected.h"
#include "layer_convolutional.h"
#include "layer_batchnorm.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

layer make_connected_layer( int batch
						, int inputs
						, int outputs
						, ACTIVATION activation
						, int batch_normalize
						, int adam )
{
	int ii;
	layer Lyr = { 0 };
	Lyr.learning_rate_scale = 1;
	Lyr.type	= CONNECTED;

	Lyr.inputs	= inputs;
	Lyr.outputs	= outputs;
	Lyr.batch	= batch;
	Lyr.batch_normalize = batch_normalize;
	Lyr.h		= 1;
	Lyr.w		= 1;
	Lyr.c		= inputs;
	Lyr.out_h	= 1;
	Lyr.out_w	= 1;
	Lyr.out_c	= outputs;
	Lyr.nweights	= Lyr.c*Lyr.out_c;
	Lyr.nbiases		= Lyr.out_c;

	Lyr.output			= calloc( batch*outputs, sizeof( float ) );
	Lyr.delta			= calloc( batch*outputs, sizeof( float ) );

	Lyr.weight_updates	= calloc( inputs*outputs, sizeof( float ) );
	Lyr.bias_updates	= calloc( outputs, sizeof( float ) );

	Lyr.weights			= calloc( outputs*inputs, sizeof( float ) );
	Lyr.biases			= calloc( outputs, sizeof( float ) );

	Lyr.forward			= forward_connected_layer;
	Lyr.backward		= backward_connected_layer;
	Lyr.update			= update_connected_layer;
	Lyr.BoJa_NaOnGab	= visualize_connected_layer_output;
	Lyr.BoJa_MuGeGab	= visualize_connected_layer_weight;

	//float scale = 1.0/sqrt(inputs);
	float scale = sqrt( 2.0/inputs );
	for ( ii=0; ii < outputs*inputs; ++ii )
	{
		Lyr.weights[ii] = scale*rand_uniform( -1, 1 );
	}

	for ( ii=0; ii < outputs; ++ii )
	{
		Lyr.biases[ii] = 0;
	}

	if ( adam )
	{
		Lyr.m		= calloc( Lyr.inputs*Lyr.outputs, sizeof( float ) );
		Lyr.v		= calloc( Lyr.inputs*Lyr.outputs, sizeof( float ) );
		Lyr.bias_m	= calloc( Lyr.outputs, sizeof( float ) );
		Lyr.scale_m	= calloc( Lyr.outputs, sizeof( float ) );
		Lyr.bias_v	= calloc( Lyr.outputs, sizeof( float ) );
		Lyr.scale_v	= calloc( Lyr.outputs, sizeof( float ) );
	}

	if ( batch_normalize )
	{
		Lyr.scales			= calloc( outputs, sizeof( float ) );
		Lyr.scale_updates	= calloc( outputs, sizeof( float ) );

		for ( ii=0; ii < outputs; ++ii )
		{
			Lyr.scales[ii] = 1;
		}

		Lyr.mean			= calloc( outputs, sizeof( float ) );
		Lyr.mean_delta		= calloc( outputs, sizeof( float ) );
		Lyr.variance		= calloc( outputs, sizeof( float ) );
		Lyr.variance_delta	= calloc( outputs, sizeof( float ) );

		Lyr.rolling_mean	= calloc( outputs, sizeof( float ) );
		Lyr.rolling_variance = calloc( outputs, sizeof( float ) );

		Lyr.x				= calloc( batch*outputs, sizeof( float ) );
		Lyr.x_norm			= calloc( batch*outputs, sizeof( float ) );
	}

	#ifdef GPU
	Lyr.forward_gpu			= forward_connected_layer_gpu;
	Lyr.backward_gpu		= backward_connected_layer_gpu;
	Lyr.update_gpu			= update_connected_layer_gpu;

	Lyr.weights_gpu			= cuda_make_array( Lyr.weights, outputs*inputs );
	Lyr.biases_gpu			= cuda_make_array( Lyr.biases, outputs );

	Lyr.weight_updates_gpu	= cuda_make_array( Lyr.weight_updates, outputs*inputs );
	Lyr.bias_updates_gpu	= cuda_make_array( Lyr.bias_updates, outputs );

	Lyr.output_gpu			= cuda_make_array( Lyr.output, outputs*batch );
	Lyr.delta_gpu			= cuda_make_array( Lyr.delta, outputs*batch );

	if ( adam )
	{
		Lyr.m_gpu			= cuda_make_array( 0, inputs*outputs );
		Lyr.v_gpu			= cuda_make_array( 0, inputs*outputs );
		Lyr.bias_m_gpu		= cuda_make_array( 0, outputs );
		Lyr.bias_v_gpu		= cuda_make_array( 0, outputs );
		Lyr.scale_m_gpu		= cuda_make_array( 0, outputs );
		Lyr.scale_v_gpu		= cuda_make_array( 0, outputs );
	}

	if ( batch_normalize )
	{
		Lyr.mean_gpu			= cuda_make_array( Lyr.mean, outputs );
		Lyr.variance_gpu		= cuda_make_array( Lyr.variance, outputs );

		Lyr.rolling_mean_gpu	= cuda_make_array( Lyr.mean, outputs );
		Lyr.rolling_variance_gpu = cuda_make_array( Lyr.variance, outputs );

		Lyr.mean_delta_gpu		= cuda_make_array( Lyr.mean, outputs );
		Lyr.variance_delta_gpu	= cuda_make_array( Lyr.variance, outputs );

		Lyr.scales_gpu			= cuda_make_array( Lyr.scales, outputs );
		Lyr.scale_updates_gpu	= cuda_make_array( Lyr.scale_updates, outputs );

		Lyr.x_gpu				= cuda_make_array( Lyr.output, Lyr.batch*outputs );
		Lyr.x_norm_gpu			= cuda_make_array( Lyr.output, Lyr.batch*outputs );

		#ifdef CUDNN
		cudnnCreateTensorDescriptor( &Lyr.normTensorDesc );
		cudnnCreateTensorDescriptor( &Lyr.dstTensorDesc );
		cudnnSetTensor4dDescriptor( Lyr.dstTensorDesc
								, CUDNN_TENSOR_NCHW
								, CUDNN_DATA_FLOAT
								, Lyr.batch
								, Lyr.out_c
								, Lyr.out_h
								, Lyr.out_w );
		cudnnSetTensor4dDescriptor( Lyr.normTensorDesc
								, CUDNN_TENSOR_NCHW
								, CUDNN_DATA_FLOAT
								, 1
								, Lyr.out_c
								, 1
								, 1 );
		#endif
	}
	#endif

	Lyr.activation = activation;
	fprintf( stderr, "connected                            %4d  ->  %4d\n", inputs, outputs );

	return Lyr;
}

void update_connected_layer( layer Lyr, update_args a )
{
	float learning_rate = a.learning_rate*Lyr.learning_rate_scale;
	float momentum	= a.momentum;
	float decay		= a.decay;
	int batch		= a.batch;

	axpy_cpu( Lyr.outputs, learning_rate/batch, Lyr.bias_updates, 1, Lyr.biases, 1 );
	scal_cpu( Lyr.outputs, momentum, Lyr.bias_updates, 1 );

	if ( Lyr.batch_normalize )
	{
		axpy_cpu( Lyr.outputs, learning_rate/batch, Lyr.scale_updates, 1, Lyr.scales, 1 );
		scal_cpu( Lyr.outputs, momentum, Lyr.scale_updates, 1 );
	}

	axpy_cpu( Lyr.inputs*Lyr.outputs, -decay*batch, Lyr.weights, 1, Lyr.weight_updates, 1 );
	axpy_cpu( Lyr.inputs*Lyr.outputs, learning_rate/batch, Lyr.weight_updates, 1, Lyr.weights, 1 );
	scal_cpu( Lyr.inputs*Lyr.outputs, momentum, Lyr.weight_updates, 1 );
}

void forward_connected_layer( layer Lyr, network net )
{
	fill_cpu( Lyr.outputs*Lyr.batch, 0, Lyr.output, 1 );

	int m		= Lyr.batch;
	int k		= Lyr.inputs;
	int n		= Lyr.outputs;
	float *a	= net.input;
	float *b	= Lyr.weights;
	float *c	= Lyr.output;

	gemm( 0, 1, m, n, k, 1, a, k, b, k, 1, c, n );

	if ( Lyr.batch_normalize )
	{
		forward_batchnorm_layer( Lyr, net );
	}
	else
	{
		add_bias( Lyr.output, Lyr.biases, Lyr.batch, Lyr.outputs, 1 );
	}

	activate_array( Lyr.output, Lyr.outputs*Lyr.batch, Lyr.activation );
}

void backward_connected_layer( layer Lyr, network net )
{
	gradient_array( Lyr.output, Lyr.outputs*Lyr.batch, Lyr.activation, Lyr.delta );

	if ( Lyr.batch_normalize )
	{
		backward_batchnorm_layer( Lyr, net );
	}
	else
	{
		backward_bias( Lyr.bias_updates, Lyr.delta, Lyr.batch, Lyr.outputs, 1 );
	}

	int m		= Lyr.outputs;
	int k		= Lyr.batch;
	int n		= Lyr.inputs;
	float *a	= Lyr.delta;
	float *b	= net.input;
	float *c	= Lyr.weight_updates;

	gemm( 1, 0, m, n, k, 1, a, m, b, n, 1, c, n );

	m = Lyr.batch;
	k = Lyr.outputs;
	n = Lyr.inputs;

	a = Lyr.delta;
	b = Lyr.weights;
	c = net.delta;

	if ( c ) gemm( 0, 0, m, n, k, 1, a, k, b, n, 1, c, n );
}


void denormalize_connected_layer( layer Lyr )
{
	int i, j;
	for ( i = 0; i < Lyr.outputs; ++i )
	{
		float scale = Lyr.scales[i]/sqrt( Lyr.rolling_variance[i] + 0.000001 );

		for ( j = 0; j < Lyr.inputs; ++j )
		{
			Lyr.weights[i*Lyr.inputs + j] *= scale;
		}

		Lyr.biases[i] -= Lyr.rolling_mean[i] * scale;
		Lyr.scales[i] = 1;
		Lyr.rolling_mean[i] = 0;
		Lyr.rolling_variance[i] = 1;
	}
}


void statistics_connected_layer( layer Lyr )
{
	if ( Lyr.batch_normalize )
	{
		printf( "Scales " );
		print_statistics( Lyr.scales, Lyr.outputs );
		/*
		printf( "Rolling Mean " );
		print_statistics( Lyr.rolling_mean, Lyr.outputs );
		printf( "Rolling Variance " );
		print_statistics( Lyr.rolling_variance, Lyr.outputs );
		*/
	}

	printf( "Biases " );
	print_statistics( Lyr.biases, Lyr.outputs );
	printf( "Weights " );
	print_statistics( Lyr.weights, Lyr.outputs );
}

#ifdef GPU

void pull_connected_layer( layer Lyr )
{
	cuda_pull_array( Lyr.weights_gpu, Lyr.weights, Lyr.inputs*Lyr.outputs );
	cuda_pull_array( Lyr.biases_gpu, Lyr.biases, Lyr.outputs );
	cuda_pull_array( Lyr.weight_updates_gpu, Lyr.weight_updates, Lyr.inputs*Lyr.outputs );
	cuda_pull_array( Lyr.bias_updates_gpu, Lyr.bias_updates, Lyr.outputs );
	if ( Lyr.batch_normalize )
	{
		cuda_pull_array( Lyr.scales_gpu, Lyr.scales, Lyr.outputs );
		cuda_pull_array( Lyr.rolling_mean_gpu, Lyr.rolling_mean, Lyr.outputs );
		cuda_pull_array( Lyr.rolling_variance_gpu, Lyr.rolling_variance, Lyr.outputs );
	}
}

void push_connected_layer( layer Lyr )
{
	cuda_push_array( Lyr.weights_gpu, Lyr.weights, Lyr.inputs*Lyr.outputs );
	cuda_push_array( Lyr.biases_gpu, Lyr.biases, Lyr.outputs );
	cuda_push_array( Lyr.weight_updates_gpu, Lyr.weight_updates, Lyr.inputs*Lyr.outputs );
	cuda_push_array( Lyr.bias_updates_gpu, Lyr.bias_updates, Lyr.outputs );
	if ( Lyr.batch_normalize )
	{
		cuda_push_array( Lyr.scales_gpu, Lyr.scales, Lyr.outputs );
		cuda_push_array( Lyr.rolling_mean_gpu, Lyr.rolling_mean, Lyr.outputs );
		cuda_push_array( Lyr.rolling_variance_gpu, Lyr.rolling_variance, Lyr.outputs );
	}
}

void update_connected_layer_gpu( layer Lyr, update_args a )
{
	float learning_rate	= a.learning_rate*Lyr.learning_rate_scale;
	float momentum		= a.momentum;
	float decay			= a.decay;
	int batch			= a.batch;

	if ( a.adam )
	{
		adam_update_gpu( Lyr.weights_gpu
					, Lyr.weight_updates_gpu
					, Lyr.m_gpu
					, Lyr.v_gpu
					, a.B1
					, a.B2
					, a.eps
					, decay
					, learning_rate
					, Lyr.inputs*Lyr.outputs
					, batch
					, a.t );
		adam_update_gpu( Lyr.biases_gpu
					, Lyr.bias_updates_gpu
					, Lyr.bias_m_gpu
					, Lyr.bias_v_gpu
					, a.B1
					, a.B2
					, a.eps
					, decay
					, learning_rate
					, Lyr.outputs
					, batch
					, a.t );

		if ( Lyr.scales_gpu )
		{
			adam_update_gpu( Lyr.scales_gpu
					, Lyr.scale_updates_gpu
					, Lyr.scale_m_gpu
					, Lyr.scale_v_gpu
					, a.B1
					, a.B2
					, a.eps
					, decay
					, learning_rate
					, Lyr.outputs
					, batch
					, a.t );
		}
	}
	else
	{
		axpy_gpu( Lyr.outputs, learning_rate/batch, Lyr.bias_updates_gpu, 1, Lyr.biases_gpu, 1 );
		scal_gpu( Lyr.outputs, momentum, Lyr.bias_updates_gpu, 1 );

		if ( Lyr.batch_normalize )
		{
			axpy_gpu( Lyr.outputs, learning_rate/batch, Lyr.scale_updates_gpu, 1, Lyr.scales_gpu, 1 );
			scal_gpu( Lyr.outputs, momentum, Lyr.scale_updates_gpu, 1 );
		}

		axpy_gpu( Lyr.inputs*Lyr.outputs, -decay*batch, Lyr.weights_gpu, 1, Lyr.weight_updates_gpu, 1 );
		axpy_gpu( Lyr.inputs*Lyr.outputs, learning_rate/batch, Lyr.weight_updates_gpu, 1, Lyr.weights_gpu, 1 );
		scal_gpu( Lyr.inputs*Lyr.outputs, momentum, Lyr.weight_updates_gpu, 1 );
	}
}

void forward_connected_layer_gpu( layer Lyr, network net )
{
	fill_gpu( Lyr.outputs*Lyr.batch, 0, Lyr.output_gpu, 1 );

	int m = Lyr.batch;
	int k = Lyr.inputs;
	int n = Lyr.outputs;
	float * a = net.input_gpu;
	float * b = Lyr.weights_gpu;
	float * c = Lyr.output_gpu;

	gemm_gpu( 0, 1, m, n, k, 1, a, k, b, k, 1, c, n );

	if ( Lyr.batch_normalize )
	{
		forward_batchnorm_layer_gpu( Lyr, net );
	}
	else
	{
		add_bias_gpu( Lyr.output_gpu, Lyr.biases_gpu, Lyr.batch, Lyr.outputs, 1 );
	}

	activate_array_gpu( Lyr.output_gpu, Lyr.outputs*Lyr.batch, Lyr.activation );
}

void backward_connected_layer_gpu( layer Lyr, network net )
{
	constrain_gpu( Lyr.outputs*Lyr.batch, 1, Lyr.delta_gpu, 1 );
	gradient_array_gpu( Lyr.output_gpu, Lyr.outputs*Lyr.batch, Lyr.activation, Lyr.delta_gpu );

	if ( Lyr.batch_normalize )
	{
		backward_batchnorm_layer_gpu( Lyr, net );
	}
	else
	{
		backward_bias_gpu( Lyr.bias_updates_gpu, Lyr.delta_gpu, Lyr.batch, Lyr.outputs, 1 );
	}

	int m = Lyr.outputs;
	int k = Lyr.batch;
	int n = Lyr.inputs;
	float * a = Lyr.delta_gpu;
	float * b = net.input_gpu;
	float * c = Lyr.weight_updates_gpu;

	gemm_gpu( 1, 0, m, n, k, 1, a, m, b, n, 1, c, n );

	m = Lyr.batch;
	k = Lyr.outputs;
	n = Lyr.inputs;

	a = Lyr.delta_gpu;
	b = Lyr.weights_gpu;
	c = net.delta_gpu;

	if ( c ) gemm_gpu( 0, 0, m, n, k, 1, a, k, b, n, 1, c, n );
}

// 하나의 포집가중값 주소를 이미지배열에 주소를 복사함
image pull_connected_weight( layer Lyr, int nn )
{
	int hh = Lyr.h;			// 포집 높이
	int ww = Lyr.w;			// 포집 너비
	int cc = Lyr.c / (Lyr.groups > 0 ? Lyr.groups : Lyr.c);	// 사비개수
	return float_to_image( ww, hh, cc, Lyr.weights + nn*hh*ww*cc );
}
// 메모리를 할당하고 할당한 메모리에 가중값을 복사한다
image *pull_connected_image_weights( layer Lyr )
{
	image *weights	= calloc( Lyr.nweights, sizeof( image ) );	// 포집판 개수만큼 이미지메모리 할당

	int ii;
	// 포집판 개수 반복
	for ( ii=0; ii < Lyr.nweights; ++ii )
	{
		// 담아둘 메모리를 할당하고, 자료값을 복사하고, 담아둔 주소를 반환한다
		weights[ii] = copy_image( pull_connected_weight( Lyr, ii ) );
		//normalize_image( weights[ii] );	//포집판별로 고르기를 하면 특징표현이 되는가???
	}

	normalize_image_MuRi( weights, Lyr.nweights );	//이미지무리 전체를 고르기한다

	return weights;
}
// 연결층 가중값 시각화
image *visualize_connected_layer_weight( layer Lyr, char *window, image *prev_weights )
{
	// 포집판 개수만큼 이미지메모리를 할당하고 담아둔 주소를 복사한다
	image *single_weights = pull_connected_image_weights( Lyr );
	show_images( single_weights, Lyr.nweights, window );

	char buff[256];
	//sprintf(buff
	sprintf_s( buff, 256
		, "%s: Output"
		, window );

	//show_image(dc, buff);
	//save_image(dc, buff);
	//free_image( dc );		//  [7/6/2018 jobs]
	return single_weights;
}
// 하나의 나온값 주소를 이미지배열에 주소를 복사함
image pull_connected_out( layer Lyr, int nn )
{
	int ww = Lyr.out_w;		// 출력 너비
	int hh = Lyr.out_h;		// 출력 높이
	int cc = 1;				// 
	int bo = ww*hh*nn;		// 출력장수 보
	return float_to_image( ww, hh, cc, Lyr.output + bo );
}
// 메모리를 할당하고 할당한 메모리에 나온값을 복사한다
image *pull_connected_image_out( layer Lyr )
{
	image *out = calloc( Lyr.outputs, sizeof( image ) );	// 포집판 개수만큼 이미지메모리 할당

	int ii;
	// 나온이미지 판개수 반복
	for ( ii=0; ii < Lyr.out_c; ++ii )
	{
		// 담아둘 메모리를 할당하고, 자료값을 복사하고, 담아둔 주소를 반환한다
		out[ii] = copy_image( pull_connected_out( Lyr, ii ) );
	}

	normalize_image_MuRi( out, Lyr.outputs );	//이미지무리 전체를 고르기한다

	return out;
}
// 연결층 나온값 시각화
image *visualize_connected_layer_output( layer Lyr, char *window, image *prev_out )
{
	// 포집판 개수만큼 이미지메모리를 할당하고 담아둔 주소를 복사한다
	image *single_out = pull_connected_image_out( Lyr );
	// 이미지를 정사각형으로 배열을 조정하고 화면에 보여준다
	show_images( single_out, Lyr.outputs, window );

	char buff[256];

	sprintf_s( buff, 256, "%s: Output", window );

	return single_out;
}

#endif
