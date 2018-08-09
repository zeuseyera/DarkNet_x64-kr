#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

#ifdef CUDNN
#pragma comment(lib, "cudnn.lib")  
#endif

extern "C" {
#include "layer_convolutional.h"
#include "layer_batchnorm.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
}

__global__ void binarize_kernel(float *x, int n, float *binary)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if (i >= n)
		return;

    binary[i] = (x[i] >= 0) ? 1.0f : -1.0f;
}
// 입력값을 0을 기준으로 1.0 또는 -1.0 의 값으로 이진화 한다
void binarize_gpu( float *x			// 입력값
				, int n				// 병렬처리 개수
				, float *binary )	// 계산한 값
{
    binarize_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, binary);
    check_error(cudaPeekAtLastError());
}

__global__ void binarize_input_kernel(float *input, int n, int size, float *binary)
{
    int s = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if (s >= size)
		return;

    int i = 0;
    float mean = 0;

    for(i = 0; i < n; ++i)
	{
        mean += abs(input[i*size + s]);
    }
    mean = mean / n;

    for(i = 0; i < n; ++i)
	{
        binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
    }
}

void binarize_input_gpu(float *input, int n, int size, float *binary)
{
    binarize_input_kernel<<<cuda_gridsize(size), BLOCK>>>(input, n, size, binary);
    check_error(cudaPeekAtLastError());
}


__global__ void binarize_weights_kernel( float *weights		// 입력값
										, int	SaRiSu		// 병렬처리 개수
										, int	GaeSu		// 계산할 개수
										, float	*binary )	// 계산한 값
{
    int f = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if ( f >= SaRiSu )
		return;

    int i = 0;
    float mean = 0;
	// GaeSu 개수의 가중값을 양수값(절대값)으로 변환하고, 변환한 값을 모두 더한다
    for( i=0; i<GaeSu; ++i )
	{
        mean += abs( weights[f*GaeSu + i] );
    }
    mean = mean / GaeSu;	// GaeSu 개수로 나누어 평균을 구한다

    for ( i=0; i<GaeSu; ++i )
	{	// 가중값이 0 보다 크면 양의 평균값, 가중값이 0 보다 작으면 음의 평균값
        binary[f*GaeSu + i] = (weights[f*GaeSu + i] > 0) ? mean : -mean;
        //binary[f*GaeSu + i] = weights[f*GaeSu + i];
    }
}
// 모든 가중값을 더하여 평균값을 구하고, 0을 기준으로 양 또는 음의 평균값으로 이진화 한다
void binarize_weights_gpu( float *weights	// 입력값
						, int	SaRiSu		// 병렬처리 개수
						, int	GaeSu		// 계산할 개수
						, float	*binary )	// 계산한 값
{
    binarize_weights_kernel<<<cuda_gridsize(SaRiSu), BLOCK>>>( weights, SaRiSu, GaeSu, binary );
    check_error( cudaPeekAtLastError() );
}
// 나선층의 순방향 계산
void forward_convolutional_layer_gpu( convolutional_layer l, network_state state )
{
	// 출력값을 0으로 설정한다
    fill_ongpu( l.outputs*l.batch, 0.0f, l.output_gpu, 1 );

    if( l.binary )
	{
		// 가중값을 양, 음의 평균값으로 이진화 한다
        binarize_weights_gpu( l.weights_gpu
							, l.n
							, l.c*l.size*l.size
							, l.binary_weights_gpu );
        swap_binary( &l );	// weights_gpu 와 binary_weights_gpu 의 값을 서로 바꾼다
    }

    if( l.xnor )
	{
		// 가중값을 양, 음의 평균값으로 이진화 한다
		binarize_weights_gpu( l.weights_gpu
							, l.n
							, l.c*l.size*l.size
							, l.binary_weights_gpu );
        swap_binary(&l);	// weights_gpu 와 binary_weights_gpu 의 값을 서로 바꾼다
		// 입력값을 1.0, -1.0 으로 이진화 한다
        binarize_gpu( state.input, l.c*l.h*l.w*l.batch, l.binary_input_gpu );
        state.input = l.binary_input_gpu;
    }

	// 나선층 순방향 출력값을 계산한다
#ifdef CUDNN
    float one = 1.0f;
    cudnnConvolutionForward( cudnn_handle()	// 쿠다 제어기
						, &one				//*alpha
						, l.srcTensorDesc	//xDesc    입력값 구조
						, state.input		//*x       입력값
						, l.weightDesc		//wDesc    가중값 구조
						, l.weights_gpu		//*w       가중값
						, l.convDesc		//convDesc 나선구조
						, l.fw_algo			//algo
						, state.workspace	//작업장 주소
						, l.workspace_size	//작업장 크기
						, &one				//*beta
						, l.dstTensorDesc	//yDesc    출력값 구조
						, l.output_gpu );	//*y       출력값

#else
    int i;
    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = l.out_w*l.out_h;

    for ( i=0; i<l.batch; ++i )
	{
		im2col_ongpu( state.input + i*l.c*l.h*l.w
					, l.c
					, l.h
					, l.w
					, l.size
					, l.stride
					, l.pad
					, state.workspace );

		float *a = l.weights_gpu;
		float *b = state.workspace;
		float *c = l.output_gpu;

		gemm_ongpu( 0
					, 0
					, m
					, n
					, k
					, 1.0f
					, a
					, k
					, b
					, n
					, 1.0f
					, c + i*m*n
					, n );
    }
#endif

    if ( l.batch_normalize )
	{
        forward_batchnorm_layer_gpu( l, state );
    }

    add_bias_gpu( l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h );

    activate_array_ongpu( l.output_gpu, l.outputs*l.batch, l.activation );

    //if(l.dot > 0) dot_error_gpu(l);
    if( l.binary || l.xnor )
		swap_binary(&l);
	//cudaDeviceSynchronize();	// for correct profiling of performance
}
// 나선층 역방향 계산
void backward_convolutional_layer_gpu( convolutional_layer l, network_state state )
{
	// 신경망 출력값으로 기울기를 계산한다
    gradient_array_ongpu( l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu );
	// 계산된 기울기로 편향값을 갱신한다
    backward_bias_gpu( l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h );

    if ( l.batch_normalize )
	{
        backward_batchnorm_layer_gpu( l, state );
        //axpy_ongpu(l.outputs*l.batch, -state.net.decay, l.x_gpu, 1, l.delta_gpu, 1);
    }
	else
	{
        //axpy_ongpu(l.outputs*l.batch, -state.net.decay, l.output_gpu, 1, l.delta_gpu, 1);
    }

    float *original_input = state.input;

    if ( l.xnor )
		state.input = l.binary_input_gpu;

	// 나선층 역방향 출력값을 계산한다
#ifdef CUDNN
    float one = 1.0f;
	// 가중값에 대한 기울기를 계산한다
    cudnnConvolutionBackwardFilter( cudnn_handle()
								, &one				//*alpha
								, l.srcTensorDesc	//xDesc    입력값 구조
								, state.input		//*x       입력값
								, l.ddstTensorDesc	//dyDesc   출력값 구조
								, l.delta_gpu		//*dy      출력값 오차
								, l.convDesc		//convDesc 나선구조
								, l.bf_algo			//algo
								, state.workspace	//작업장 주소
								, l.workspace_size	//작업장 크기
								, &one				//*beta
								, l.dweightDesc		//dwDesc   가중값 구조
								, l.weight_updates_gpu );//*dw 가중값 오차

    if( state.delta )
	{
        if ( l.binary || l.xnor )
			swap_binary(&l);
		// 출력오차와 가중값으로 입력값 오차를 계산한다
        cudnnConvolutionBackwardData( cudnn_handle()
								, &one				//*alpha
								, l.weightDesc		//wDesc    가중값 구조
								, l.weights_gpu		//*w       가중값
								, l.ddstTensorDesc	//dyDesc   출력값 구조
								, l.delta_gpu		//*dy      출력값 오차
								, l.convDesc		//convDesc 나선구조
								, l.bd_algo			//algo
								, state.workspace	//작업장 주소
								, l.workspace_size	//작업장 크기
								, &one				//*beta
								, l.dsrcTensorDesc	//dxDesc   입력값 구조
								, state.delta );	//*dx      입력값 오차

        if ( l.binary || l.xnor )
			swap_binary( &l );

        if ( l.xnor )
			gradient_array_ongpu( original_input, l.batch*l.c*l.h*l.w, HARDTAN, state.delta );
    }

#else
    int m = l.n;
    int n = l.size*l.size*l.c;
    int k = l.out_w*l.out_h;

    int i;

    for( i=0; i<l.batch; ++i )
	{
        float *a = l.delta_gpu;
        float *b = state.workspace;
        float *c = l.weight_updates_gpu;

        im2col_ongpu( state.input + i*l.c*l.h*l.w
					, l.c
					, l.h
					, l.w
					, l.size
					, l.stride
					, l.pad
					, state.workspace );

        gemm_ongpu( 0
					, 1
					, m
					, n
					, k
					, 1
					, a + i*m*k
					, k
					, b
					, k
					, 1
					, c
					, n );

        if ( state.delta )
		{
            if(l.binary || l.xnor)
				swap_binary(&l);

            float * a = l.weights_gpu;
            float * b = l.delta_gpu;
            float * c = state.workspace;

            gemm_ongpu( 1
					, 0
					, n
					, k
					, m
					, 1
					, a
					, n
					, b + i*k*m
					, k
					, 0
					, c
					, k );

            col2im_ongpu( state.workspace
					, l.c
					, l.h
					, l.w
					, l.size
					, l.stride
					, l.pad
					, state.delta + i*l.c*l.h*l.w );

            if ( l.binary || l.xnor )
			{
                swap_binary(&l);
            }

            if ( l.xnor )
				gradient_array_ongpu( original_input + i*l.c*l.h*l.w
									, l.c*l.h*l.w
									, HARDTAN
									, state.delta + i*l.c*l.h*l.w );
        }
    }
#endif
}

void pull_convolutional_layer( convolutional_layer layer )
{
    cuda_pull_array(layer.weights_gpu, layer.weights, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_pull_array(layer.weight_updates_gpu, layer.weight_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);

    if (layer.batch_normalize)
	{
        cuda_pull_array(layer.scales_gpu, layer.scales, layer.n);
        cuda_pull_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
        cuda_pull_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
    }

    if (layer.adam)
	{
        cuda_pull_array(layer.m_gpu, layer.m, layer.c*layer.n*layer.size*layer.size);
        cuda_pull_array(layer.v_gpu, layer.v, layer.c*layer.n*layer.size*layer.size);
    }
}

void push_convolutional_layer( convolutional_layer layer )
{
    cuda_push_array( layer.weights_gpu, layer.weights, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array( layer.biases_gpu, layer.biases, layer.n);
    cuda_push_array( layer.weight_updates_gpu, layer.weight_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array( layer.bias_updates_gpu, layer.bias_updates, layer.n);

    if ( layer.batch_normalize )
	{
        cuda_push_array( layer.scales_gpu, layer.scales, layer.n );
        cuda_push_array( layer.rolling_mean_gpu, layer.rolling_mean, layer.n );
        cuda_push_array( layer.rolling_variance_gpu, layer.rolling_variance, layer.n );
    }

    if ( layer.adam )
	{
        cuda_push_array( layer.m_gpu, layer.m, layer.c*layer.n*layer.size*layer.size );
        cuda_push_array( layer.v_gpu, layer.v, layer.c*layer.n*layer.size*layer.size );
    }
}

void update_convolutional_layer_gpu(convolutional_layer layer, int batch, float learning_rate, float momentum, float decay)
{
    int size = layer.size*layer.size*layer.c*layer.n;
    axpy_ongpu(layer.n, learning_rate/batch, layer.bias_updates_gpu, 1, layer.biases_gpu, 1);
    scal_ongpu(layer.n, momentum, layer.bias_updates_gpu, 1);

    if(layer.scales_gpu){
        axpy_ongpu(layer.n, learning_rate/batch, layer.scale_updates_gpu, 1, layer.scales_gpu, 1);
        scal_ongpu(layer.n, momentum, layer.scale_updates_gpu, 1);
    }

    if(layer.adam){
        scal_ongpu(size, layer.B1, layer.m_gpu, 1);
        scal_ongpu(size, layer.B2, layer.v_gpu, 1);

        axpy_ongpu(size, -decay*batch, layer.weights_gpu, 1, layer.weight_updates_gpu, 1);

        axpy_ongpu(size, -(1-layer.B1), layer.weight_updates_gpu, 1, layer.m_gpu, 1);
        mul_ongpu(size, layer.weight_updates_gpu, 1, layer.weight_updates_gpu, 1);
        axpy_ongpu(size, (1-layer.B2), layer.weight_updates_gpu, 1, layer.v_gpu, 1);

        adam_gpu(size, layer.weights_gpu, layer.m_gpu, layer.v_gpu, layer.B1, layer.B2, learning_rate/batch, layer.eps, layer.t+1);
        fill_ongpu(size, 0, layer.weight_updates_gpu, 1);
    }else{
        axpy_ongpu(size, -decay*batch, layer.weights_gpu, 1, layer.weight_updates_gpu, 1);
        axpy_ongpu(size, learning_rate/batch, layer.weight_updates_gpu, 1, layer.weights_gpu, 1);
        scal_ongpu(size, momentum, layer.weight_updates_gpu, 1);
    }
}


