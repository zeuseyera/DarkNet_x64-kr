
#include "layer_convolutional.h"
#include "utils.h"
#include "layer_batchnorm.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

#ifdef AI2
#include "xnor_layer.h"
#endif

void swap_binary( convolutional_layer *Lyr )
{
	float *swap		= Lyr->weights;
	Lyr->weights	= Lyr->binary_weights;
	Lyr->binary_weights = swap;

	#ifdef GPU
	swap			= Lyr->weights_gpu;
	Lyr->weights_gpu = Lyr->binary_weights_gpu;
	Lyr->binary_weights_gpu = swap;
	#endif
}

void binarize_weights( float *weights, int n, int size, float *binary )
{
	int i, f;
	for ( f = 0; f < n; ++f )
	{
		float mean = 0;
		for ( i = 0; i < size; ++i )
		{
			mean += fabs( weights[f*size + i] );
		}
		mean = mean / size;
		for ( i = 0; i < size; ++i )
		{
			binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
		}
	}
}

void binarize_cpu( float *input, int n, float *binary )
{
	int i;
	for ( i = 0; i < n; ++i )
	{
		binary[i] = (input[i] > 0) ? 1 : -1;
	}
}

void binarize_input( float *input, int n, int size, float *binary )
{
	int i, s;
	for ( s = 0; s < size; ++s )
	{
		float mean = 0;
		for ( i = 0; i < n; ++i )
		{
			mean += fabs( input[i*size + s] );
		}
		mean = mean / n;
		for ( i = 0; i < n; ++i )
		{
			binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
		}
	}
}

int convolutional_out_height( convolutional_layer Lyr )
{
	return (Lyr.h + 2*Lyr.pad - Lyr.size) / Lyr.stride + 1;
}

int convolutional_out_width( convolutional_layer Lyr )
{
	return (Lyr.w + 2*Lyr.pad - Lyr.size) / Lyr.stride + 1;
}

image get_convolutional_image( convolutional_layer Lyr )
{
	return float_to_image( Lyr.out_w, Lyr.out_h, Lyr.out_c, Lyr.output );
}

image get_convolutional_delta( convolutional_layer Lyr )
{
	return float_to_image( Lyr.out_w, Lyr.out_h, Lyr.out_c, Lyr.delta );
}

static size_t get_workspace_size( layer Lyr )
{
	#ifdef CUDNN
	if ( gpu_index >= 0 )
	{
		size_t most = 0;
		size_t s = 0;
		cudnnGetConvolutionForwardWorkspaceSize( cudnn_handle()
							, Lyr.srcTensorDesc
							, Lyr.weightDesc
							, Lyr.convDesc
							, Lyr.dstTensorDesc
							, Lyr.fw_algo
							, &s );
		if ( s > most ) most = s;
		cudnnGetConvolutionBackwardFilterWorkspaceSize( cudnn_handle()
							, Lyr.srcTensorDesc
							, Lyr.ddstTensorDesc
							, Lyr.convDesc
							, Lyr.dweightDesc
							, Lyr.bf_algo
							, &s );
		if ( s > most ) most = s;
		cudnnGetConvolutionBackwardDataWorkspaceSize( cudnn_handle()
							, Lyr.weightDesc
							, Lyr.ddstTensorDesc
							, Lyr.convDesc
							, Lyr.dsrcTensorDesc
							, Lyr.bd_algo
							, &s );
		if ( s > most ) most = s;
		return most;
	}
	#endif
	return (size_t)Lyr.out_h*Lyr.out_w*Lyr.size*Lyr.size*Lyr.c/Lyr.groups*sizeof( float );
}

#ifdef GPU
#ifdef CUDNN
void cudnn_convolutional_setup( layer *Lyr )
{
	cudnnSetTensor4dDescriptor( Lyr->dsrcTensorDesc
							, CUDNN_TENSOR_NCHW
							, CUDNN_DATA_FLOAT
							, Lyr->batch
							, Lyr->c
							, Lyr->h
							, Lyr->w );
	cudnnSetTensor4dDescriptor( Lyr->ddstTensorDesc
							, CUDNN_TENSOR_NCHW
							, CUDNN_DATA_FLOAT
							, Lyr->batch
							, Lyr->out_c
							, Lyr->out_h
							, Lyr->out_w );

	cudnnSetTensor4dDescriptor( Lyr->srcTensorDesc
							, CUDNN_TENSOR_NCHW
							, CUDNN_DATA_FLOAT
							, Lyr->batch
							, Lyr->c
							, Lyr->h
							, Lyr->w );
	cudnnSetTensor4dDescriptor( Lyr->dstTensorDesc
							, CUDNN_TENSOR_NCHW
							, CUDNN_DATA_FLOAT
							, Lyr->batch
							, Lyr->out_c
							, Lyr->out_h
							, Lyr->out_w );
	cudnnSetTensor4dDescriptor( Lyr->normTensorDesc
							, CUDNN_TENSOR_NCHW
							, CUDNN_DATA_FLOAT
							, 1
							, Lyr->out_c
							, 1
							, 1 );

	cudnnSetFilter4dDescriptor( Lyr->dweightDesc
							, CUDNN_DATA_FLOAT
							, CUDNN_TENSOR_NCHW
							, Lyr->n
							, Lyr->c/Lyr->groups
							, Lyr->size
							, Lyr->size );
	cudnnSetFilter4dDescriptor( Lyr->weightDesc
							, CUDNN_DATA_FLOAT
							, CUDNN_TENSOR_NCHW
							, Lyr->n
							, Lyr->c/Lyr->groups
							, Lyr->size
							, Lyr->size );

	#if CUDNN_MAJOR >= 6	// cudnn 6.0
	cudnnSetConvolution2dDescriptor( Lyr->convDesc
							, Lyr->pad
							, Lyr->pad
							, Lyr->stride
							, Lyr->stride
							, 1
							, 1
							, CUDNN_CROSS_CORRELATION
							, CUDNN_DATA_FLOAT );
	#else					// cudnn 5.1
	cudnnSetConvolution2dDescriptor( Lyr->convDesc
							, Lyr->pad
							, Lyr->pad
							, Lyr->stride
							, Lyr->stride
							, 1
							, 1
							, CUDNN_CROSS_CORRELATION );
	#endif

	#if CUDNN_MAJOR >= 7
	cudnnSetConvolutionGroupCount( Lyr->convDesc, Lyr->groups );
	#else
	if ( Lyr->groups > 1 )
	{
		//error( "CUDNN < 7 doesn't support groups, please upgrade!" );
		error( "CUDNN 7 미만은 포집 모듬을 지원하지 않음, 판올림을 하라!" );
	}
	#endif

	cudnnGetConvolutionForwardAlgorithm( cudnn_handle()
							, Lyr->srcTensorDesc
							, Lyr->weightDesc
							, Lyr->convDesc
							, Lyr->dstTensorDesc
							, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
							, 4000000000
							, &Lyr->fw_algo );
	cudnnGetConvolutionBackwardDataAlgorithm( cudnn_handle()
							, Lyr->weightDesc
							, Lyr->ddstTensorDesc
							, Lyr->convDesc
							, Lyr->dsrcTensorDesc
							, CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT
							, 4000000000
							, &Lyr->bd_algo );
	cudnnGetConvolutionBackwardFilterAlgorithm( cudnn_handle()
							, Lyr->srcTensorDesc
							, Lyr->ddstTensorDesc
							, Lyr->convDesc
							, Lyr->dweightDesc
							, CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT
							, 4000000000
							, &Lyr->bf_algo );
}
#endif
#endif

convolutional_layer make_convolutional_layer( int batch
											, int h
											, int w
											, int c
											, int n
											, int groups
											, int size
											, int stride
											, int padding
											, ACTIVATION activation
											, int batch_normalize
											, int binary
											, int xnor
											, int adam )
{
	int i;
	convolutional_layer Lyr = { 0 };
	Lyr.type = CONVOLUTIONAL;

	Lyr.groups = groups;
	Lyr.h = h;
	Lyr.w = w;
	Lyr.c = c;
	Lyr.n = n;
	Lyr.binary	= binary;
	Lyr.xnor	= xnor;
	Lyr.batch	= batch;
	Lyr.stride	= stride;
	Lyr.size	= size;
	Lyr.pad		= padding;
	Lyr.batch_normalize = batch_normalize;

	Lyr.weights		= calloc( c/groups*n*size*size, sizeof( float ) );
	Lyr.weight_updates = calloc( c/groups*n*size*size, sizeof( float ) );

	Lyr.biases		= calloc( n, sizeof( float ) );
	Lyr.bias_updates = calloc( n, sizeof( float ) );

	Lyr.nweights	= ( c/groups )*n*size*size;
	Lyr.nbiases		= n;

	//float scale = 1.0/sqrt(size*size*c);
	float scale = sqrt( 2.0/( ( size*size*c )/Lyr.groups ) );
	//printf("convscale %f\n", scale);
	//scale = .02;

	//for ( i = 0; i < c*n*size*size; ++i )
	//	Lyr.weights[i] = scale*rand_uniform(-1, 1);
	for ( i = 0; i < Lyr.nweights; ++i )
		Lyr.weights[i] = scale*rand_normal();

	int out_w	= convolutional_out_width( Lyr );
	int out_h	= convolutional_out_height( Lyr );
	Lyr.out_h	= out_h;
	Lyr.out_w	= out_w;
	Lyr.out_c	= n;
	Lyr.outputs	= Lyr.out_h * Lyr.out_w * Lyr.out_c;
	Lyr.inputs	= Lyr.w * Lyr.h * Lyr.c;

	Lyr.output = calloc( Lyr.batch*Lyr.outputs, sizeof( float ) );
	Lyr.delta  = calloc( Lyr.batch*Lyr.outputs, sizeof( float ) );

	Lyr.forward		= forward_convolutional_layer;
	Lyr.backward	= backward_convolutional_layer;
	Lyr.update		= update_convolutional_layer;
	Lyr.BoJa_NaOnGab = visualize_convolutional_layer_output;
	Lyr.BoJa_MuGeGab = visualize_convolutional_layer_weight;

	if ( binary )
	{
		Lyr.binary_weights	= calloc( Lyr.nweights, sizeof( float ) );
		Lyr.cweights		= calloc( Lyr.nweights, sizeof( char ) );
		Lyr.scales			= calloc( n, sizeof( float ) );
	}

	if ( xnor )
	{
		Lyr.binary_weights	= calloc( Lyr.nweights, sizeof( float ) );
		Lyr.binary_input	= calloc( Lyr.inputs*Lyr.batch, sizeof( float ) );
	}

	if ( batch_normalize )
	{
		Lyr.scales			= calloc( n, sizeof( float ) );
		Lyr.scale_updates	= calloc( n, sizeof( float ) );
		for ( i = 0; i < n; ++i )
		{
			Lyr.scales[i] = 1;
		}

		Lyr.mean			= calloc( n, sizeof( float ) );
		Lyr.variance		= calloc( n, sizeof( float ) );

		Lyr.mean_delta		= calloc( n, sizeof( float ) );
		Lyr.variance_delta	= calloc( n, sizeof( float ) );

		Lyr.rolling_mean	= calloc( n, sizeof( float ) );
		Lyr.rolling_variance = calloc( n, sizeof( float ) );
		Lyr.x				= calloc( Lyr.batch*Lyr.outputs, sizeof( float ) );
		Lyr.x_norm			= calloc( Lyr.batch*Lyr.outputs, sizeof( float ) );
	}

	if ( adam )
	{
		Lyr.m		= calloc( Lyr.nweights, sizeof( float ) );
		Lyr.v		= calloc( Lyr.nweights, sizeof( float ) );
		Lyr.bias_m	= calloc( n, sizeof( float ) );
		Lyr.scale_m	= calloc( n, sizeof( float ) );
		Lyr.bias_v	= calloc( n, sizeof( float ) );
		Lyr.scale_v	= calloc( n, sizeof( float ) );
	}

	#ifdef GPU
	Lyr.forward_gpu	= forward_convolutional_layer_gpu;
	Lyr.backward_gpu = backward_convolutional_layer_gpu;
	Lyr.update_gpu	= update_convolutional_layer_gpu;

	if ( gpu_index >= 0 )
	{
		if ( adam )
		{
			Lyr.m_gpu		= cuda_make_array( Lyr.m, Lyr.nweights );
			Lyr.v_gpu		= cuda_make_array( Lyr.v, Lyr.nweights );
			Lyr.bias_m_gpu	= cuda_make_array( Lyr.bias_m, n );
			Lyr.bias_v_gpu	= cuda_make_array( Lyr.bias_v, n );
			Lyr.scale_m_gpu	= cuda_make_array( Lyr.scale_m, n );
			Lyr.scale_v_gpu	= cuda_make_array( Lyr.scale_v, n );
		}

		Lyr.weights_gpu		= cuda_make_array( Lyr.weights, Lyr.nweights );
		Lyr.weight_updates_gpu = cuda_make_array( Lyr.weight_updates, Lyr.nweights );

		Lyr.biases_gpu		= cuda_make_array( Lyr.biases, n );
		Lyr.bias_updates_gpu = cuda_make_array( Lyr.bias_updates, n );

		Lyr.delta_gpu		= cuda_make_array( Lyr.delta, Lyr.batch*out_h*out_w*n );
		Lyr.output_gpu		= cuda_make_array( Lyr.output, Lyr.batch*out_h*out_w*n );

		if ( binary )
		{
			Lyr.binary_weights_gpu = cuda_make_array( Lyr.weights, Lyr.nweights );
		}
		if ( xnor )
		{
			Lyr.binary_weights_gpu = cuda_make_array( Lyr.weights, Lyr.nweights );
			Lyr.binary_input_gpu	= cuda_make_array( 0, Lyr.inputs*Lyr.batch );
		}

		if ( batch_normalize )
		{
			Lyr.mean_gpu			= cuda_make_array( Lyr.mean, n );
			Lyr.variance_gpu		= cuda_make_array( Lyr.variance, n );

			Lyr.rolling_mean_gpu	= cuda_make_array( Lyr.mean, n );
			Lyr.rolling_variance_gpu = cuda_make_array( Lyr.variance, n );

			Lyr.mean_delta_gpu		= cuda_make_array( Lyr.mean, n );
			Lyr.variance_delta_gpu = cuda_make_array( Lyr.variance, n );

			Lyr.scales_gpu			= cuda_make_array( Lyr.scales, n );
			Lyr.scale_updates_gpu	= cuda_make_array( Lyr.scale_updates, n );

			Lyr.x_gpu			= cuda_make_array( Lyr.output, Lyr.batch*out_h*out_w*n );
			Lyr.x_norm_gpu		= cuda_make_array( Lyr.output, Lyr.batch*out_h*out_w*n );
		}

		#ifdef CUDNN
		cudnnCreateTensorDescriptor( &Lyr.normTensorDesc );
		cudnnCreateTensorDescriptor( &Lyr.srcTensorDesc );
		cudnnCreateTensorDescriptor( &Lyr.dstTensorDesc );
		cudnnCreateFilterDescriptor( &Lyr.weightDesc );
		cudnnCreateTensorDescriptor( &Lyr.dsrcTensorDesc );
		cudnnCreateTensorDescriptor( &Lyr.ddstTensorDesc );
		cudnnCreateFilterDescriptor( &Lyr.dweightDesc );
		cudnnCreateConvolutionDescriptor( &Lyr.convDesc );
		cudnn_convolutional_setup( &l );
		#endif

	}
	#endif

	Lyr.workspace_size = get_workspace_size( Lyr );
	Lyr.activation = activation;

	fprintf( stderr
			, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n"
			, n
			, size
			, size
			, stride
			, w
			, h
			, c
			, Lyr.out_w
			, Lyr.out_h
			, Lyr.out_c
			, ( ( 2.0 * Lyr.n * Lyr.size*Lyr.size*Lyr.c ) /
			  ( Lyr.groups * Lyr.out_h*Lyr.out_w ) ) / 1000000000.0 );

	return Lyr;
}

void denormalize_convolutional_layer( convolutional_layer Lyr )
{
	int i, j;
	for ( i = 0; i < Lyr.n; ++i )
	{
		float scale = Lyr.scales[i]/sqrt( Lyr.rolling_variance[i] + 0.00001 );

		for ( j = 0; j < Lyr.c/Lyr.groups*Lyr.size*Lyr.size; ++j )
		{
			Lyr.weights[i*Lyr.c/Lyr.groups*Lyr.size*Lyr.size + j] *= scale;
		}

		Lyr.biases[i] -= Lyr.rolling_mean[i] * scale;
		Lyr.scales[i] = 1;
		Lyr.rolling_mean[i] = 0;
		Lyr.rolling_variance[i] = 1;
	}
}

/*
void test_convolutional_layer()
{
	convolutional_layer l = make_convolutional_layer(1, 5, 5, 3, 2, 5, 2, 1, LEAKY, 1, 0, 0, 0);
	Lyr.batch_normalize = 1;
	float data[] = {1,1,1,1,1,
					1,1,1,1,1,
					1,1,1,1,1,
					1,1,1,1,1,
					1,1,1,1,1,
					2,2,2,2,2,
					2,2,2,2,2,
					2,2,2,2,2,
					2,2,2,2,2,
					2,2,2,2,2,
					3,3,3,3,3,
					3,3,3,3,3,
					3,3,3,3,3,
					3,3,3,3,3,
					3,3,3,3,3};
	//net.input = data;
	//forward_convolutional_layer(l);
}
*/

void resize_convolutional_layer( convolutional_layer *Lyr, int w, int h )
{
	Lyr->w = w;
	Lyr->h = h;
	int out_w = convolutional_out_width( *Lyr );
	int out_h = convolutional_out_height( *Lyr );

	Lyr->out_w = out_w;
	Lyr->out_h = out_h;

	Lyr->outputs	= Lyr->out_h * Lyr->out_w * Lyr->out_c;
	Lyr->inputs	= Lyr->w * Lyr->h * Lyr->c;

	Lyr->output	= realloc( Lyr->output, Lyr->batch*Lyr->outputs*sizeof( float ) );
	Lyr->delta	= realloc( Lyr->delta, Lyr->batch*Lyr->outputs*sizeof( float ) );

	if ( Lyr->batch_normalize )
	{
		Lyr->x		= realloc( Lyr->x, Lyr->batch*Lyr->outputs*sizeof( float ) );
		Lyr->x_norm	= realloc( Lyr->x_norm, Lyr->batch*Lyr->outputs*sizeof( float ) );
	}

	#ifdef GPU
	cuda_free( Lyr->delta_gpu );
	cuda_free( Lyr->output_gpu );

	Lyr->delta_gpu	= cuda_make_array( Lyr->delta, Lyr->batch*Lyr->outputs );
	Lyr->output_gpu	= cuda_make_array( Lyr->output, Lyr->batch*Lyr->outputs );

	if ( Lyr->batch_normalize )
	{
		cuda_free( Lyr->x_gpu );
		cuda_free( Lyr->x_norm_gpu );

		Lyr->x_gpu		= cuda_make_array( Lyr->output, Lyr->batch*Lyr->outputs );
		Lyr->x_norm_gpu	= cuda_make_array( Lyr->output, Lyr->batch*Lyr->outputs );
	}
	#ifdef CUDNN
	cudnn_convolutional_setup( l );
	#endif
	#endif

	Lyr->workspace_size = get_workspace_size( *Lyr );
}

void add_bias( float *output, float *biases, int batch, int n, int size )
{
	int i, j, b;
	for ( b = 0; b < batch; ++b )
	{
		for ( i = 0; i < n; ++i )
		{
			for ( j = 0; j < size; ++j )
			{
				output[(b*n + i)*size + j] += biases[i];
			}
		}
	}
}

void scale_bias( float *output, float *scales, int batch, int n, int size )
{
	int i, j, b;
	for ( b = 0; b < batch; ++b )
	{
		for ( i = 0; i < n; ++i )
		{
			for ( j = 0; j < size; ++j )
			{
				output[(b*n + i)*size + j] *= scales[i];
			}
		}
	}
}

void backward_bias( float *bias_updates, float *delta, int batch, int n, int size )
{
	int i, b;
	for ( b = 0; b < batch; ++b )
	{
		for ( i = 0; i < n; ++i )
		{
			bias_updates[i] += sum_array( delta+size*(i+b*n), size );
		}
	}
}

void forward_convolutional_layer( convolutional_layer Lyr, network net )
{
	int i, j;

	fill_cpu( Lyr.outputs*Lyr.batch, 0, Lyr.output, 1 );

	if ( Lyr.xnor )
	{
		binarize_weights( Lyr.weights, Lyr.n, Lyr.c/Lyr.groups*Lyr.size*Lyr.size, Lyr.binary_weights );
		swap_binary( &Lyr );
		binarize_cpu( net.input, Lyr.c*Lyr.h*Lyr.w*Lyr.batch, Lyr.binary_input );
		net.input = Lyr.binary_input;
	}

	int m = Lyr.n/Lyr.groups;
	int k = Lyr.size*Lyr.size*Lyr.c/Lyr.groups;
	int n = Lyr.out_w*Lyr.out_h;
	for ( i = 0; i < Lyr.batch; ++i )
	{
		for ( j = 0; j < Lyr.groups; ++j )
		{
			float *a = Lyr.weights + j*Lyr.nweights/Lyr.groups;
			float *b = net.workspace;
			float *c = Lyr.output + (i*Lyr.groups + j)*n*m;
			float *im =  net.input + (i*Lyr.groups + j)*Lyr.c/Lyr.groups*Lyr.h*Lyr.w;

			if ( Lyr.size == 1 )
			{
				b = im;
			}
			else
			{
				im2col_cpu( im, Lyr.c/Lyr.groups, Lyr.h, Lyr.w, Lyr.size, Lyr.stride, Lyr.pad, b );
			}
			gemm( 0, 0, m, n, k, 1, a, k, b, n, 1, c, n );
		}
	}

	if ( Lyr.batch_normalize )
	{
		forward_batchnorm_layer( Lyr, net );
	}
	else
	{
		add_bias( Lyr.output, Lyr.biases, Lyr.batch, Lyr.n, Lyr.out_h*Lyr.out_w );
	}

	activate_array( Lyr.output, Lyr.outputs*Lyr.batch, Lyr.activation );
	if ( Lyr.binary || Lyr.xnor ) swap_binary( &Lyr );
}

void backward_convolutional_layer( convolutional_layer Lyr, network net )
{
	int i, j;
	int m = Lyr.n/Lyr.groups;
	int n = Lyr.size*Lyr.size*Lyr.c/Lyr.groups;
	int k = Lyr.out_w*Lyr.out_h;

	gradient_array( Lyr.output, Lyr.outputs*Lyr.batch, Lyr.activation, Lyr.delta );

	if ( Lyr.batch_normalize )
	{
		backward_batchnorm_layer( Lyr, net );
	}
	else
	{
		backward_bias( Lyr.bias_updates, Lyr.delta, Lyr.batch, Lyr.n, k );
	}

	for ( i = 0; i < Lyr.batch; ++i )
	{
		for ( j = 0; j < Lyr.groups; ++j )
		{
			float *a = Lyr.delta + (i*Lyr.groups + j)*m*k;
			float *b = net.workspace;
			float *c = Lyr.weight_updates + j*Lyr.nweights/Lyr.groups;

			float *im  = net.input + (i*Lyr.groups + j)*Lyr.c/Lyr.groups*Lyr.h*Lyr.w;
			float *imd = net.delta + (i*Lyr.groups + j)*Lyr.c/Lyr.groups*Lyr.h*Lyr.w;

			if ( Lyr.size == 1 )
			{
				b = im;
			}
			else
			{
				im2col_cpu( im
						, Lyr.c/Lyr.groups
						, Lyr.h
						, Lyr.w
						, Lyr.size
						, Lyr.stride
						, Lyr.pad
						, b );
			}

			gemm( 0, 1, m, n, k, 1, a, k, b, k, 1, c, n );

			if ( net.delta )
			{
				a = Lyr.weights + j*Lyr.nweights/Lyr.groups;
				b = Lyr.delta + (i*Lyr.groups + j)*m*k;
				c = net.workspace;

				if ( Lyr.size == 1 )
				{
					c = imd;
				}

				gemm( 1, 0, n, k, m, 1, a, n, b, k, 0, c, k );

				if ( Lyr.size != 1 )
				{
					col2im_cpu( net.workspace
							, Lyr.c/Lyr.groups
							, Lyr.h
							, Lyr.w
							, Lyr.size
							, Lyr.stride
							, Lyr.pad
							, imd );
				}
			}
		}
	}
}

void update_convolutional_layer( convolutional_layer Lyr, update_args a )
{
	float learning_rate = a.learning_rate*Lyr.learning_rate_scale;
	float momentum	= a.momentum;
	float decay		= a.decay;
	int batch		= a.batch;

	axpy_cpu( Lyr.n, learning_rate/batch, Lyr.bias_updates, 1, Lyr.biases, 1 );
	scal_cpu( Lyr.n, momentum, Lyr.bias_updates, 1 );

	if ( Lyr.scales )
	{
		axpy_cpu( Lyr.n, learning_rate/batch, Lyr.scale_updates, 1, Lyr.scales, 1 );
		scal_cpu( Lyr.n, momentum, Lyr.scale_updates, 1 );
	}

	axpy_cpu( Lyr.nweights, -decay*batch, Lyr.weights, 1, Lyr.weight_updates, 1 );
	axpy_cpu( Lyr.nweights, learning_rate/batch, Lyr.weight_updates, 1, Lyr.weights, 1 );
	scal_cpu( Lyr.nweights, momentum, Lyr.weight_updates, 1 );
}

// 하나의 포집가중값 주소를 이미지배열에 주소를 복사함
image pull_convolutional_weight( convolutional_layer Lyr, int nn )
{
	int hh = Lyr.size;			// 포집 높이
	int ww = Lyr.size;			// 포집 너비
	int cc = Lyr.c/Lyr.groups;	// 사비개수
	return float_to_image( ww, hh, cc, Lyr.weights + nn*hh*ww*cc );
}

void rgbgr_weights( convolutional_layer Lyr )
{
	int ii;
	for ( ii=0; ii < Lyr.n; ++ii )
	{
		image im = pull_convolutional_weight( Lyr, ii );
		if ( im.c == 3 )
		{
			rgbgr_image( im );
		}
	}
}

void rescale_weights( convolutional_layer Lyr, float scale, float trans )
{
	int ii;
	for ( ii=0; ii < Lyr.n; ++ii )
	{
		image im = pull_convolutional_weight( Lyr, ii );
		if ( im.c == 3 )
		{
			scale_image( im, scale );
			float sum = sum_array( im.data, im.w*im.h*im.c );
			Lyr.biases[ii] += sum*trans;
		}
	}
}

// 메모리를 할당하고 할당한 메모리에 가중값을 복사한다
image *pull_convolutional_weights( convolutional_layer Lyr )
{
	image *weights = calloc( Lyr.n, sizeof( image ) );	// 포집판 개수만큼 이미지메모리 할당

	int ii;
	// 포집판 개수 반복
	for ( ii=0; ii < Lyr.n; ++ii )
	{
		// 담아둘 메모리를 할당하고, 자료값을 복사하고, 담아둔 주소를 반환한다
		weights[ii] = copy_image( pull_convolutional_weight( Lyr, ii ) );
		normalize_image( weights[ii] );	//포집판별로 고르기를 하면 특징표현이 되는가???
/*
		char buff[256];
		sprintf(buff, "filter%d", i);
		save_image(weights[i], buff);
*/
	}

	//normalize_image_MuRi( weights, Lyr.n );	//이미지무리 전체를 고르기한다

	//error("hey");
	return weights;
}
// 나선층 가중값 시각화
image *visualize_convolutional_layer_weight( convolutional_layer Lyr, char *window, image *prev_weights )
{
	// 포집판 개수만큼 이미지메모리를 할당하고 담아둔 주소를 복사한다
	image *single_weights = pull_convolutional_weights( Lyr );
	show_images( single_weights, Lyr.n, window );

	//image delta	= get_convolutional_image( l );		//  [7/6/2018 jobs]
	//image dc	= collapse_image_layers( delta, 1 );	//  [7/6/2018 jobs]

	char buff[256];
	//sprintf(buff, "%s: Output", window);
	sprintf_s( buff, 256, "%s: Output", window );
	//show_image(dc, buff);
	//save_image(dc, buff);
	//free_image( dc );		//  [7/6/2018 jobs]
	return single_weights;
}
// 하나의 포집가중값 주소를 이미지배열에 주소를 복사함
image pull_convolutional_out( convolutional_layer Lyr, int nn )
{
	int ww = Lyr.out_w;		// 출력 너비
	int hh = Lyr.out_h;		// 출력 높이
	int cc = 1;				// 
	int bo = ww*hh*nn;		// 출력장수 보
	return float_to_image( ww, hh, cc, Lyr.output + bo );
}
// 메모리를 할당하고 할당한 메모리에 가중값을 복사한다
image *pull_convolutional_image_out( convolutional_layer Lyr )
{
	image *out = calloc( Lyr.n, sizeof( image ) );	// 포집판 개수만큼 이미지메모리 할당

	int ii;
	// 나온이미지 판개수 반복
	for ( ii=0; ii < Lyr.out_c; ++ii )
	{
		// 담아둘 메모리를 할당하고, 자료값을 복사하고, 담아둔 주소를 반환한다
		out[ii] = copy_image( pull_convolutional_out( Lyr, ii ) );
		normalize_image( out[ii] );	//포집판별로 고르기를 하면 특징표현이 되는가???
	}

	//normalize_image_MuRi( out, Lyr.n );	//이미지무리 전체를 고르기한다

	return out;
}
// 나선층 나온값 시각화
image *visualize_convolutional_layer_output( convolutional_layer Lyr, char *window, image *prev_out )
{
	// 포집판 개수만큼 이미지메모리를 할당하고 담아둔 주소를 복사한다
	image *single_out = pull_convolutional_image_out( Lyr );
	// 이미지를 정사각형으로 배열을 조정하고 화면에 보여준다
	show_images( single_out, Lyr.n, window );

	char buff[256];

	sprintf_s( buff, 256, "%s: Output", window );

	return single_out;
}
