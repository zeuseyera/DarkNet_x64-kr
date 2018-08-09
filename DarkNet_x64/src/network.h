// Oh boy, why am I about to do this....
#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"

#include "image.h"
#include "data.h"
#include "tree.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
	CONSTANT
	, STEP
	, EXP
	, POLY
	, STEPS
	, SIG
	, RANDOM
} learning_rate_policy;

typedef struct network
{
	float	*workspace;
	int		n;			// 단(층)수
	int		batch;		// 사리수(한동이)
	int		*seen;		// 수련, 평가한 자료개수
	float	epoch;		// 세대수
	int		subdivisions;
	float	momentum;
	float	decay;
	layer	*layers;	// 신경망의 모든 단(층)
	int		outputs;	// 신경망 출력개수
	float	*output;	// 신경망 출력값
	learning_rate_policy policy;	// 학습율 정책

	float	learning_rate;
	float	gamma;
	float	scale;
	float	power;
	int		time_steps;
	int		step;
	int		max_batches;	// 동이개수
	float	*scales;
	int		*steps;
	int		num_steps;
	int		burn_in;

	int		adam;
	float	B1;
	float	B2;
	float	eps;

	int		inputs;	// 입력개수(가로x세로x판수)
	int		h;		// 세로
	int		w;		// 가로
	int		c;		// 판개수(채널)
	int		max_crop;
	int		min_crop;
	float	angle;
	float	aspect;
	float	exposure;
	float	saturation;
	float	hue;

	int		gpu_index;
	tree	*hierarchy;

#ifdef GPU
	float **input_gpu;
	float **truth_gpu;
	#endif
} network;

typedef struct network_state
{
	float *truth;
	float *input;		// 사비값에 대한 장치(GPU) 메모리의 첫번째 주소
	float *delta;
	float *workspace;	// 장치(GPU)의 작업공간 메모리에 대한 장치(GPU) 메모리의 첫번째 주소
	int train;
	int index;
	network net;
} network_state;

#ifdef GPU
float train_networks( network *nets, int n, data d, int interval );
void sync_nets( network *nets, int n, int interval );
float train_network_datum_gpu( network net, float *x, float *y );	// 쿠다로 신경망 수련
float *network_predict_gpu( network net, float *input );
float * get_network_output_gpu_layer( network net, int i );
float * get_network_delta_gpu_layer( network net, int i );
float *get_network_output_gpu( network net );
void forward_network_gpu( network net, network_state state );	//신경망 순방향 계산
void backward_network_gpu( network net, network_state state );	// 신경망 역방향 계산
void update_network_gpu( network *net );
#endif

float get_current_rate( network *net );
int get_current_batch( network *net );
void free_network( network net );
void compare_networks( network n1, network n2, data d );
char *get_layer_string( LAYER_TYPE a );

network make_network( int n );	// 신경망의 기본 메모리를 할당한다
void forward_network( network net, network_state state );
void backward_network( network net, network_state state );
void update_network( network net );

float train_network( network net, data d );	// 신경망에 자료를 전달하여 수련
float train_network_batch( network net, data d, int n );
float train_network_sgd( network net, data d, int n );
float train_network_datum( network net, float *x, float *y ); // 신경망에 입력, 목표값 전달 수련

matrix network_predict_data( network net, data test );
float *network_predict( network net, float *input );	// 입력값으로 신경망 출력값을 계산한다
float network_accuracy( network net, data d );
float *network_accuracies( network net, data d, int n );
float network_accuracy_multi( network net, data d, int n );
void top_predictions( network net, int n, int *index );
float *get_network_output( network net );
float *get_network_output_layer( network net, int i );
float *get_network_delta_layer( network net, int i );
float *get_network_delta( network net );
int get_network_output_size_layer( network net, int i );
int get_network_output_size( network net );	// 신경망 출력개수 반환
image get_network_image( network net );
image get_network_image_layer( network net, int i );
int get_predicted_class_network( network net );
void print_network( network net );
void visualize_network( network net );	// 신경망 가중값을 시각화하여 보여줌
int resize_network( network *net, int w, int h );
void set_batch_network( network *net, int b );
int get_network_input_size( network net );	// 신경망 입력개수 반환
float get_network_cost( network net );
int get_network_nuisance( network net );
int get_network_background( network net );

#ifdef __cplusplus
}
#endif

#endif

