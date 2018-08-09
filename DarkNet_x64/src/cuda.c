int gpu_index = 0;

#ifdef GPU

#include "cuda.h"
#include "./utils.h"
#include "blas.h"
#include "assert.h"
#include <stdlib.h>
#include <time.h>

void cuda_set_device( int n )
{
	gpu_index = n;
	cudaError_t status = cudaSetDevice( n );
	check_error( status );
}

int cuda_get_device()
{
	int n = 0;
	cudaError_t status = cudaGetDevice( &n );
	check_error( status );
	return n;
}

void check_error( cudaError_t status )
{
	//cudaDeviceSynchronize();
	cudaError_t status2 = cudaGetLastError();
	if ( status != cudaSuccess )
	{
		const char *s = cudaGetErrorString( status );
		char buffer[256];
		printf( "CUDA Error: %s\n", s );
		assert( 0 );
		snprintf( buffer, 256, "CUDA Error: %s", s );
		error( buffer );
	}
	if ( status2 != cudaSuccess )
	{
		const char *s = cudaGetErrorString( status );
		char buffer[256];
		printf( "CUDA Error Prev: %s\n", s );
		assert( 0 );
		snprintf( buffer, 256, "CUDA Error Prev: %s", s );
		error( buffer );
	}
}

dim3 cuda_gridsize( size_t n )
{
	size_t k = (n-1) / BLOCK + 1;
	size_t x = k;
	size_t y = 1;

	if ( x > 65535 )
	{
		x = ceil( sqrt( k ) );
		y = (n-1)/(x*BLOCK) + 1;
	}

	dim3 d = { x, y, 1 };
	//printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
	return d;
}

#ifdef CUDNN
cudnnHandle_t cudnn_handle()
{
	static int init[16] = { 0 };
	static cudnnHandle_t handle[16];
	int i = cuda_get_device();

	if ( !init[i] )
	{
		cudnnCreate( &handle[i] );
		init[i] = 1;
	}
	return handle[i];
}
#endif

cublasHandle_t blas_handle()
{
	static int init[16] = { 0 };
	static cublasHandle_t handle[16];
	int i = cuda_get_device();

	if ( !init[i] )
	{
		cublasCreate( &handle[i] );
		init[i] = 1;
	}

	return handle[i];
}
// 장치에 실수형 메모리를 할당, 선정한 갑을 복사, 할당한 메모리의 주소를 반환한다
float *cuda_make_array( float *x, size_t GaeSu )
{
	float *x_gpu;
	// 장치에 할당할 메모리의 크기를 계산한다
	size_t Sul = sizeof( float ) * GaeSu;
	// 장치에 계산된 크기로 메모리를 할당한다
	cudaError_t status = cudaMalloc( (void **)&x_gpu, Sul );	// 장치에 메모리 할당
	check_error( status );

	// 계산에 필요한 사비값을 쥔장 메모리에서 장치의 메모리로 복사한다
	if ( x )
	{
		status = cudaMemcpy( x_gpu, x, Sul, cudaMemcpyHostToDevice );
		check_error( status );
	}

	if ( !x_gpu )
		error( "쿠다 메모리할당 실패\n" );	//error( "Cuda malloc failed\n" );

	return x_gpu;
}

void cuda_random( float *x_gpu, size_t n )
{
	static curandGenerator_t gen[16];
	static int init[16] = { 0 };
	int i = cuda_get_device();

	if ( !init[i] )
	{
		curandCreateGenerator( &gen[i], CURAND_RNG_PSEUDO_DEFAULT );
		curandSetPseudoRandomGeneratorSeed( gen[i], time( 0 ) );
		init[i] = 1;
	}

	curandGenerateUniform( gen[i], x_gpu, n );
	check_error( cudaPeekAtLastError() );
}

float cuda_compare( float *x_gpu, float *x, size_t n, char *s )
{
	float *tmp = calloc( n, sizeof( float ) );	// 쥔장메모리를 임시로 할당한다
	cuda_pull_array( x_gpu, tmp, n );			// 장치메모리의 값을 쥔장메모리로 복사한다
	//int i;
	//for(i = 0; i < n; ++i) printf("%f %f\n", tmp[i], x[i]);
	axpy_cpu( n, -1.0f, x, 1, tmp, 1 );	// 쥔장과 장치의 차를 계산, tmp += -1.0 * x

	float err = dot_cpu( n, tmp, 1, tmp, 1 );	// 모든 차의 제곱합, sum( tmp * tmp )
	printf( "Error %s: %f\n", s, sqrt( err/n ) );

	free( tmp );
	return err;
}
// 장치에 정수형 메모리를 할당하고 할당한 메모리의 주소를 반환한다
int *cuda_make_int_array( size_t GaeSu )
{
	int *x_gpu;
	// 장치에 할당할 메모리의 크기를 계산한다
	size_t Sul = sizeof( int ) * GaeSu;
	// 장치에 계산된 크기로 메모리를 할당한다
	cudaError_t status = cudaMalloc( (void **)&x_gpu, Sul );
	check_error( status );

	return x_gpu;
}

void cuda_free( float *x_gpu )
{
	cudaError_t status = cudaFree( x_gpu );
	check_error( status );
}
// 쥔장에서 장치로 값 복사
void cuda_push_array( float *x_gpu, float *x, size_t GaeSu )
{
	// 복사할 메모리의 크기를 계산한다
	size_t Sul = sizeof( float ) * GaeSu;
	// 쥔장에서 장치로 메모리 복사
	cudaError_t status = cudaMemcpy( x_gpu, x, Sul, cudaMemcpyHostToDevice );
	check_error( status );
}
// 장치에서 쥔장으로 값 복사
void cuda_pull_array( float *x_gpu, float *x, size_t GaeSu )
{
	// 복사할 메모리의 크기를 계산한다
	size_t Sul = sizeof( float ) * GaeSu;
	// 장치에서 쥔장으로 메모리 복사
	cudaError_t status = cudaMemcpy( x, x_gpu, Sul, cudaMemcpyDeviceToHost );
	check_error( status );
}

#endif
