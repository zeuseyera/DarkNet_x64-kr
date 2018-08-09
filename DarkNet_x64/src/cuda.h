
// 파일명: cuda.h

#ifndef CUDA_H
#define CUDA_H

#if defined(_MSC_VER) && _MSC_VER < 1900
	#define inline __inline
#endif

#ifdef GPU

#define BLOCK 512

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

#ifdef __cplusplus	// 선언하는 헤더파일은 포함하지 않는다
extern "C" {
#endif

extern int gpu_index;

#ifdef CUDNN
#include "cudnn.h"
#endif

void check_error( cudaError_t status );
cublasHandle_t blas_handle();

// 장치에 실수형 메모리를 할당, 선정한 갑을 복사, 할당한 메모리의 주소를 반환한다
float* cuda_make_array( float *x, size_t n );
// 장치에 정수형 메모리를 할당하고 할당한 메모리의 주소를 반환한다
int* cuda_make_int_array( size_t n);

void cuda_push_array( float *x_gpu, float *x, size_t GaeSu );	// 쥔장에서 장치로 값 복사
void cuda_pull_array( float *x_gpu, float *x, size_t GaeSu );	// 장치에서 쥔장으로 값 복사

void cuda_set_device( int n );

void cuda_free( float *x_gpu );

void cuda_random( float *x_gpu, size_t n );

float cuda_compare( float *x_gpu, float *x, size_t n, char *s );

dim3 cuda_gridsize( size_t n );

#ifdef CUDNN
cudnnHandle_t cudnn_handle();
#endif

#ifdef __cplusplus
}
#endif

#endif

#endif
