#ifndef DARKNET_API
#define DARKNET_API

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

//#define HAVE_STRUCT_TIMESPEC	//과제속성에 설정, error C2011: 'timespec': 'struct' 형식 재정의
#include <pthread.h>

#define SECRET_NUM -1234
//extern int gpu_index;		// LNK2001 [7/3/2018 jobs]

#ifdef GPU
	#define BLOCK 512

	#include "cuda_runtime.h"
	#include "curand.h"
	#include "cublas_v2.h"

	#ifdef CUDNN
	#include "cudnn.h"
	#endif
#endif

#ifndef __cplusplus
	#ifdef OPENCV
	#include "opencv2/highgui/highgui_c.h"
	#include "opencv2/imgproc/imgproc_c.h"
	#include "opencv2/core/version.hpp"

	#if CV_MAJOR_VERSION == 3
	#include "opencv2/videoio/videoio_c.h"
	#include "opencv2/imgcodecs/imgcodecs_c.h"
	#endif
	#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern int gpu_index;	// LNK2001 [7/3/2018 jobs], extern "C" { 안에 있어야함

/// MNIST 자료관련
#define MNIST_IMAGE_MAGIC 2051	//MNIST 이미지파일 고유번호
#define MNIST_LABEL_MAGIC 2049	//MNIST 꼬리표파일 고유번호

#ifdef _MSC_VER
#define bswap( x ) _byteswap_ulong( x )
#else
#define bswap( x ) __builtin_bswap32( x )
#endif

// UNIFORM_ZERO_THRU_ONE 는 0(포함)과 1(제외)사이의 균일하게 분포된 수를 제공한다
#define UNIFORM_ZERO_THRU_ONE ( (double)(rand())/(RAND_MAX + 1 ) ) 
// UNIFORM_PLUS_MINUS_ONE 는 -1(포함)과 1(제외)사이의 균일하게 분포된 수를 제공한다
#define UNIFORM_PLUS_MINUS_ONE ( (double)(2.0 * rand())/RAND_MAX - 1.0 )

// MNIST 사비이미지 자료집합
//#pragma pack( push, 1 )
typedef struct
{
	unsigned int magic;	//고유 번호 (UBYTE_IMAGE_MAGIC).
	unsigned int GaeSu;	//자료집합에서 이미지의 갯수.
	unsigned int SeRo;	//각 이미지의 세로(높이).
	unsigned int GaRo;	//각 이미지의 가로(너비).

} MNIST_ImageHeader;

void MNIST_ImageSwap( MNIST_ImageHeader Img );

// MNIST 목표값 자료집합
typedef struct
{
	unsigned int magic;	//고유번호 (UBYTE_LABEL_MAGIC).
	unsigned int GaeSu;	//자료집합에서 이미지의 갯수.

} MNIST_LabelHeader;

void MNIST_LabelSwap( MNIST_LabelHeader lbl );
//#pragma pack( pop )

typedef struct
{
	int classes;
	char **names;
} metadata;

metadata get_metadata( char *file );

typedef struct
{
	int *leaf;
	int n;
	int *parent;
	int *child;
	int *group;
	char **name;

	int groups;			// 계층 무리개수
	int *group_size;	// 
	int *group_offset;
} tree;
tree *read_tree( char *filename );

typedef enum
{
	  LOGISTIC
	, RELU
	, RELIE
	, LINEAR
	, RAMP
	, TANH
	, PLSE
	, LEAKY
	, ELU
	, LOGGY
	, STAIR
	, HARDTAN
	, LHTAN
} ACTIVATION;

typedef enum
{
	  MULT
	, ADD
	, SUB
	, DIV
} BINARY_ACTIVATION;

typedef enum
{
	  CONVOLUTIONAL
	, DECONVOLUTIONAL
	, CONNECTED
	, MAXPOOL
	, SOFTMAX
	, DETECTION
	, DROPOUT
	, CROP
	, ROUTE
	, COST
	, NORMALIZATION
	, AVGPOOL
	, LOCAL
	, SHORTCUT
	, ACTIVE
	, RNN
	, GRU
	, LSTM
	, CRNN
	, BATCHNORM
	, NETWORK
	, XNOR
	, REGION
	, YOLO
	, REORG
	, UPSAMPLE
	, LOGXENT
	, L2NORM
	, BLANK
	, SOHAENG	// 예지행동
} LAYER_TYPE;

typedef enum
{
	  SSE		// 출력오차(편차)와 SSE손실을 계산한다
	, MASKED
	, L1		// 오차는 -1, +1 로 계산, 손실은 양수값으로 변환
	, SEG
	, SMOOTH
	, WGAN
} COST_TYPE;

typedef enum
{
	CLASSIFICATION_DATA
	, DETECTION_DATA
	, CAPTCHA_DATA
	, REGION_DATA
	, IMAGE_DATA
	, COMPARE_DATA
	, WRITING_DATA
	, SWAG_DATA
	, TAG_DATA
	, OLD_CLASSIFICATION_DATA
	, STUDY_DATA
	, DET_DATA
	, SUPER_DATA
	, LETTERBOX_DATA
	, REGRESSION_DATA
	, SEGMENTATION_DATA
	, INSTANCE_DATA
	, MNIST_DATA
	, BYEORIM_DATA
} data_type;	// 학습자료 유형(CIFAR 등)

typedef struct
{
	int batch;
	float learning_rate;
	float momentum;
	float decay;
	int adam;
	float B1;
	float B2;
	float eps;
	int t;
} update_args;	// 망 벼림 참여값 갱신

typedef struct
{
	int w;	//너비
	int h;	//높이
	int c;	//색
	float *data;	//값
} image;

struct network;
typedef struct network network;

struct layer;
typedef struct layer layer;

struct layer
{
	LAYER_TYPE type;		// 층 유형
	ACTIVATION activation;
	COST_TYPE cost_type;
	void ( *forward )		( struct layer, struct network );
	void ( *backward )		( struct layer, struct network );
	void ( *update )		( struct layer, update_args );
	void ( *forward_gpu )	( struct layer, struct network );
	void ( *backward_gpu )	( struct layer, struct network );
	void ( *update_gpu )	( struct layer, update_args );
	image* ( *BoJa_NaOnGab )	( struct layer, char*, image* );	// 출력값 보기
	image* ( *BoJa_MuGeGab )	( struct layer, char*, image* );	// 가중값 보기
	int batch_normalize;	// 실제 bool 로만 사용함
	int shortcut;
	int batch;		// 사리개수(한동이)
	int forced;
	int flipped;
	int inputs;		// 입력개수
	int outputs;	// 출력개수
	int nweights;	// 가중값개수
	int nbiases;	// 편향값개수
	int extra;
	int truths;		// 목표값 개수(욜로: 90*(4 + 1))
	int h;			// 사비 세로크기
	int w;			// 사비 가로크기
	int c;			// 사비 판(채널)개수
	int out_h;		// 출력 세로크기
	int out_w;		// 출력 가로크기
	int out_c;		// 출력 판 개수
	int n;			// 포집판(커널,필터) 또는 편향값 개수(출력개수)
	int max_boxes;	// 상자 최대개수
	int groups;		// 포집판 묶음 개수
	int size;		// 포집(커널,필터) 가로, 세로 크기
	int side;
	int stride;		// 보
	int reverse;
	int flatten;
	int spatial;
	int pad;		// 덧댐
	int sqrt;
	int flip;
	int index;
	int binary;
	int xnor;
	int steps;
	int hidden;
	int truth;		// 목표값층 표시용
	float smooth;
	float dot;
	float angle;
	float jitter;
	float saturation;
	float exposure;
	float shift;
	float ratio;
	float learning_rate_scale;
	float clip;
	int softmax;
	int classes;		// 분류개수
	int coords;
	int background;
	int rescore;
	int objectness;
	int joint;
	int noadjust;
	int reorg;
	int log;
	int tanh;
	int *mask;	// 마스크값 배열(욜로 3개: 3,4,5 또는 1,2,3)
	int total;
	/// 강화벼림 관련 변수
//	bool bSoHaeng;	// 소행단 여부

	float alpha;
	float beta;
	float kappa;

	float coord_scale;
	float object_scale;
	float noobject_scale;
	float mask_scale;
	float class_scale;
	int bias_match;
	int random;
	float ignore_thresh;
	float truth_thresh;
	float thresh;
	float focus;
	int classfix;
	int absolute;

	int onlyforward;
	int stopbackward;
	int dontload;
	int dontsave;
	int dontloadscales;

	float temperature;
	float probability;
	float scale;

	char  * cweights;
	int   * indexes;
	int   * input_layers;	// 각 사비단 번호
	int   * input_sizes;	// 각 사비경로의 입력개수
	int   * map;	// 분류 계층지도
	float * rand;
	float * cost;	// 코스트값(오직 하나의 값)
	float * state;
	float * prev_state;
	float * forgot_state;
	float * forgot_delta;
	float * state_delta;
	float * combine_cpu;
	float * combine_delta_cpu;

	float * concat;
	float * concat_delta;

	float * binary_weights;

	float * biases;			// 편향값(쥔장메모리)
	float * bias_updates;

	float * scales;
	float * scale_updates;

	float * weights;		// 가중값(쥔장메모리)
	float * weight_updates;

	float * delta;			// 오차값(쥔장메모리)
	float * output;			// 출력값(쥔장메모리)
	float * loss;
	float * squared;
	float * norms;

	float * spatial_mean;
	float * mean;
	float * variance;

	float * mean_delta;
	float * variance_delta;

	float * rolling_mean;
	float * rolling_variance;

	float * x;
	float * x_norm;

	float * m;
	float * v;

	float * bias_m;
	float * bias_v;
	float * scale_m;
	float * scale_v;


	float *z_cpu;
	float *r_cpu;
	float *h_cpu;
	float * prev_state_cpu;

	float *temp_cpu;
	float *temp2_cpu;
	float *temp3_cpu;

	float *dh_cpu;
	float *hh_cpu;
	float *prev_cell_cpu;
	float *cell_cpu;
	float *f_cpu;
	float *i_cpu;
	float *g_cpu;
	float *o_cpu;
	float *c_cpu;
	float *dc_cpu;

	float * binary_input;

	struct layer *input_layer;	// 입력층
	struct layer *self_layer;	// 자기층
	struct layer *output_layer;	// 출력층

	struct layer *reset_layer;
	struct layer *update_layer;
	struct layer *state_layer;

	struct layer *input_gate_layer;
	struct layer *state_gate_layer;
	struct layer *input_save_layer;
	struct layer *state_save_layer;
	struct layer *input_state_layer;
	struct layer *state_state_layer;

	struct layer *input_z_layer;
	struct layer *state_z_layer;

	struct layer *input_r_layer;
	struct layer *state_r_layer;

	struct layer *input_h_layer;
	struct layer *state_h_layer;

	struct layer *wz;
	struct layer *uz;
	struct layer *wr;
	struct layer *ur;
	struct layer *wh;
	struct layer *uh;
	struct layer *uo;
	struct layer *wo;
	struct layer *uf;
	struct layer *wf;
	struct layer *ui;
	struct layer *wi;
	struct layer *ug;
	struct layer *wg;

	tree *softmax_tree;

	size_t workspace_size;	// 작업에 필요한 메모리 크기(바이트 단위)

	#ifdef GPU
	int *indexes_gpu;

	float *z_gpu;
	float *r_gpu;
	float *h_gpu;

	float *temp_gpu;
	float *temp2_gpu;
	float *temp3_gpu;

	float *dh_gpu;
	float *hh_gpu;
	float *prev_cell_gpu;
	float *cell_gpu;
	float *f_gpu;
	float *i_gpu;
	float *g_gpu;
	float *o_gpu;
	float *c_gpu;
	float *dc_gpu;

	float *m_gpu;
	float *v_gpu;
	float *bias_m_gpu;
	float *scale_m_gpu;
	float *bias_v_gpu;
	float *scale_v_gpu;

	float * combine_gpu;
	float * combine_delta_gpu;

	float * prev_state_gpu;
	float * forgot_state_gpu;
	float * forgot_delta_gpu;
	float * state_gpu;
	float * state_delta_gpu;
	float * gate_gpu;
	float * gate_delta_gpu;
	float * save_gpu;
	float * save_delta_gpu;
	float * concat_gpu;
	float * concat_delta_gpu;

	float * binary_input_gpu;
	float * binary_weights_gpu;

	float * mean_gpu;
	float * variance_gpu;

	float * rolling_mean_gpu;
	float * rolling_variance_gpu;

	float * variance_delta_gpu;
	float * mean_delta_gpu;

	float * x_gpu;				// 입력값(장치 메모리)
	float * x_norm_gpu;
	float * weights_gpu;		// 가중값(장치 메모리)
	float * weight_updates_gpu;
	float * weight_change_gpu;

	float * biases_gpu;			// 편향값(장치 메모리)
	float * bias_updates_gpu;
	float * bias_change_gpu;

	float * scales_gpu;
	float * scale_updates_gpu;
	float * scale_change_gpu;

	float * output_gpu;			// 출력값(장치 메모리)
	float * loss_gpu;
	float * delta_gpu;			// 오차값(장치 메모리)
	float * rand_gpu;
	float * squared_gpu;
	float * norms_gpu;

	#ifdef CUDNN
	cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
	cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
	cudnnTensorDescriptor_t normTensorDesc;
	cudnnFilterDescriptor_t weightDesc;
	cudnnFilterDescriptor_t dweightDesc;
	cudnnConvolutionDescriptor_t	convDesc;
	cudnnConvolutionFwdAlgo_t		fw_algo;
	cudnnConvolutionBwdDataAlgo_t	bd_algo;
	cudnnConvolutionBwdFilterAlgo_t	bf_algo;
	#endif
	#endif
};

void free_layer( layer );

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
	int n;				// 단(층)수
	int batch;			// 사리수(한동이)
	int subdivisions;	// 재분할 개수(다중 GPU로 처리할때 사용하기 위해???)

	size_t *seen;		// 수련, 평가한 자료개수 추적, train_network_datum 에서 계수
	int *t;
	float epoch;		// 세대수

	layer *layers;		// 신경망의 모든 단(층)
	float *output;		// 신경망 출력값
	learning_rate_policy policy;	// 학습율 정책

	data_type JaRyoJong;	// 자료종류

	float learning_rate;
	float momentum;
	float decay;
	float gamma;
	float scale;
	float power;
	int time_steps;
	int step;
	int max_batches;	// 벼림 끝내기를 판단하기 위한 최대 동이개수
	float *scales;
	int   *steps;
	int num_steps;
	int burn_in;

	int adam;
	float B1;
	float B2;
	float eps;

	int inputs;		// 입력개수(가로x세로x판수)
	int outputs;	// 신경망 출력개수
	int truths;		// 목표값 개수 == 출력개수
	int notruth;
	int h;			// 세로
	int w;			// 가로
	int c;			// 판개수(채널)
	int max_crop;
	int min_crop;
	float max_ratio;
	float min_ratio;
	int center;
	float angle;
	float aspect;
	float exposure;
	float saturation;
	float hue;
	int random;
	/// 강화벼림 관련 변수
	int nSoHaengDan;	// 소행 출력(걸정)단

	int gpu_index;
	tree *hierarchy;

	float *input;		// 순방향 계산시 각 단의 입력값 주소를 잠시 담아두는 곳
	float *truth;		// 목표값???
	float *delta;		// 오차값???(목표값-출력값)
	float *workspace;
	int train;			// 벼림해야한다는 상태를 알림
	int index;			// 현재 진행중인 단 순번을 임시로 담아둔다
	float *cost;
	float clip;

	#ifdef GPU
	float *input_gpu;	// 순방향 계산시 각 단의 입력값 주소를 잠시 담아두는 곳
	float *truth_gpu;	// 목표값???
	float *delta_gpu;	// 오차값???(목표값-출력값)
	float *output_gpu;
	#endif

} network;

typedef struct
{
	int w;
	int h;
	float scale;
	float rad;
	float dx;
	float dy;
	float aspect;
} augment_args;

typedef struct
{
	float x;	//시작 가로
	float y;	//시작 세로
	float w;	//너비
	float h;	//높이
} box;	//상자정보

typedef struct detection
{
	box		bbox;		// 경계상자
	int		classes;	// 분류개수
	float	*prob;		// 신경망 예측(출력층 출력) 확률값
	float	*mask;
	float	objectness;	// 개체상태(가장큰 출력값)
	int		sort_class;
} detection;	// 검출정보

typedef struct matrix
{
	int rows;		//총 자료개수(행(가로), 자료개수)
	int cols;		//열(세로), w x h x c(자료하나 크기)
	float **vals;	//값
} matrix;

// 신경망 입력값, 목표값, 출력값 구조
typedef struct
{
	int w;			// 가로, 세로
	int h;			// 세로
	matrix X;		// 신경망 입력값
	matrix y;		// 목표값?
	int *labels;	// 목표값 딱지
	int shallow;	// 얄팍하다(할당한 메모리를 해제하기위한 표식)
	int *num_boxes;	// 경계구역 개수
	box **boxes;	// 경계구역
} data;

typedef struct load_args
{
	int threads;	// 쓰레드개수
	char **paths;	// 사비할 파일이름(char 형)
	char *path;
	int n;			// 입력자료 개수(사리수 x subdivisions x GPU 개수)
	int m;			// 자료(수련, 평가 등의) 총 개수
	char **labels;	// 딱지 목록
	int h;			// 사비 세로
	int w;			// 사비 가로
	int out_w;
	int out_h;
	int nh;
	int nw;
	int num_boxes;
	int min;		// 최소 사리수 crop min
	int max;		// 최대 사리수 crop max
	int size;
	int classes;	// 분류개수
	int background;
	int scale;
	int center;
	int coords;
	float jitter;
	float angle;
	float aspect;
	float saturation;
	float exposure;
	float hue;
	data *d;	// X: 사비값, y: 목표값
	image *im;
	image *resized;
	data_type type;	// 학습자료 유형(CIFAR 등)
	tree *hierarchy;
	data *JaRyo_MNIST;	// MNIST 자료
} load_args;	// 입출력 결정값 구조

typedef struct
{
	int id;
	float x, y, w, h;
	float left, right, top, bottom;
} box_label;


load_args get_base_args( network *net );

void free_data( data d );

typedef struct node
{
	void *val;			// 현재마디의 값
	struct node *next;	// 하위마디
	struct node *prev;	// 상위마디
} node;	// 마디

typedef struct list
{
	int size;
	node *front;	// 마디의 뿌리
	node *back;		// 마디의 가지
} list;	// 선택사항 목록

pthread_t load_data( load_args args );	// 자료를 탑재하는  쓰레드 하나
list *read_data_cfg( char *filename );
list *read_cfg( char *filename );
unsigned char *read_file( char *filename );
data resize_data( data orig, int w, int h );
data *tile_data( data orig, int divs, int size );
data select_data( data *orig, int *inds );

void forward_network( network *net );
void backward_network( network *net );
void update_network( network *net );


/// blas.c
// 자료값(X) 배열의 INCX 보위치 값과 출력값(Y)에 배열의 INCY 보위치 값을 곱한것을 모두 더하여 반환한다
float dot_cpu( int N, float *X, int INCX, float *Y, int INCY );
// 자료값(X) 배열의 INCX 보위치 값에 ALPHA 값을 곱한후 출력값(Y)에 배열의 INCY 보 위치값에 더한다
void axpy_cpu( int N, float ALPHA, float *X, int INCX, float *Y, int INCY );
// 자료값(X) 배열의 INCX 보위치 값을 출력값(Y)에 배열의 INCY 보위치 값으로 복사한다
void copy_cpu( int N, float *X, int INCX, float *Y, int INCY );
// 자료값(X) 배열의 INCX 보위치에 ALPHA 값으로 곱한다
void scal_cpu( int N, float ALPHA, float *X, int INCX );
// 자료값(X) 배열의 INCX 보위치에 ALPHA 값으로 채운다
void fill_cpu( int N, float ALPHA, float * X, int INCX );
void normalize_cpu( float *x, float *mean, float *variance, int batch, int filters, int spatial );
void softmax( float *input, int n, float temp, int stride, float *output );

int best_3d_shift_r( image a, image b, int min, int max );

#ifdef GPU
void axpy_gpu( int N, float ALPHA, float * X, int INCX, float * Y, int INCY );
void fill_gpu( int N, float ALPHA, float * X, int INCX );
void scal_gpu( int N, float ALPHA, float * X, int INCX );
void copy_gpu( int N, float * X, int INCX, float * Y, int INCY );

/// cuda.h
void cuda_set_device( int n );	// c2732 에러 [6/28/2018 jobs]
void cuda_free( float *x_gpu );
float *cuda_make_array( float *x, size_t n );	// 장치에 지정한 크기의 메모리 할당
// 쥔장의 지정한 메모리에서 장치의 지정한 메모리로 지정한 개수의 값을 밀어넣는다
void cuda_push_array( float *x_gpu, float *x, size_t n );
// 장치의 지정한 메모리에서 쥔장의 지정한 메모리로 지정한 개수의 값을 긁어온다
void cuda_pull_array( float *x_gpu, float *x, size_t n );
float cuda_mag_array( float *x_gpu, size_t n );

void forward_network_gpu( network *net );
void backward_network_gpu( network *net );
void update_network_gpu( network *net );

float train_networks( network **nets, int n, data d, int interval );
void sync_nets( network **nets, int n, int interval );
void harmless_update_network_gpu( network *net );
#endif

image get_label( image **characters, char *string, int size );
void draw_label( image a, int r, int c, image label, const float *rgb );
void save_image_png( image im, const char *name );
void get_next_batch( data d, int n, int offset, float *X, float *y );	// 입력값과 목표값을 가져온다
void grayscale_image_3c( image im );
void matrix_to_csv( matrix m );
float train_network_sgd( network *net, data d, int n );
void rgbgr_image( image im );
data copy_data( data d );
data concat_data( data d1, data d2 );
data load_cifar10_data( char *filename );
float matrix_topk_accuracy( matrix truth, matrix guess, int k );
void matrix_add_matrix( matrix from, matrix to );
void scale_matrix( matrix m, float scale );
matrix csv_to_matrix( char *filename );
float *network_accuracies( network *net, data d, int n );
float train_network_datum( network *net );
image make_random_image( int w, int h, int c );

/// layer_connected.h
void denormalize_connected_layer( layer l );				// c2732 에러 [6/28/2018 jobs]
void statistics_connected_layer( layer l );					// c2732 에러 [6/28/2018 jobs]
/// layer_convolutional.h
void denormalize_convolutional_layer( layer l );			// c2732 에러 [6/28/2018 jobs]
void rgbgr_weights( layer l );								// c2732 에러 [6/28/2018 jobs]
void rescale_weights( layer l, float scale, float trans );	// c2732 에러 [6/28/2018 jobs]
image *pull_convolutional_weights( layer l );				// c2732 에러 [6/28/2018 jobs]

void demo( char *cfgfile
		, char *weightfile
		, float thresh
		, int cam_index
		, const char *filename
		, char **names
		, int classes
		, int frame_skip
		, char *prefix
		, int avg
		, float hier_thresh		// 계층 문턱값
		, int w
		, int h
		, int fps
		, int fullscreen );
void get_detection_detections( layer l, int w, int h, float thresh, detection *dets );

char *option_find_str( list *l, char *key, char *def );
int option_find_int( list *l, char *key, int def );
int option_find_int_quiet( list *l, char *key, int def );

/// parser.h
network *parse_network_cfg( char *filename );		// c2732 에러 [6/28/2018 jobs]
void save_weights( network *net, char *filename );	// c2732 에러 [6/28/2018 jobs]
void load_weights( network *net, char *filename );	// c2732 에러 [6/28/2018 jobs]
void save_weights_upto( network *net, char *filename, int cutoff );	// c2732 에러 [6/28/2018 jobs]
void load_weights_upto( network *net, char *filename, int start, int cutoff );	// c2732 에러 [6/28/2018 jobs]

void zero_objectness( layer l );
void get_region_detections( layer l
						, int w
						, int h
						, int netw
						, int neth
						, float thresh
						, int *map
						, float tree_thresh
						, int relative
						, detection *dets );
int get_yolo_detections( layer l
						, int w
						, int h
						, int netw
						, int neth
						, float thresh
						, int *map
						, int relative
						, detection *dets );
void free_network( network *net );
void set_temp_network( network *net, float t );

/// network.c
network *load_network( char *cfg, char *weights, int clear );	// c2732 에러 [6/28/2018 jobs]
void set_batch_network( network *net, int b );					// c2732 에러 [6/28/2018 jobs]
float *network_predict( network *net, float *input );			// c2732 에러 [6/28/2018 jobs]
int resize_network( network *net, int w, int h );				// c2732 에러 [6/28/2018 jobs]
//void visualize_network( network *net );							// c2732 에러 [6/28/2018 jobs]

/// util.c
double what_time_is_it_now();									// c2732 에러 [6/28/2018 jobs]
int find_arg( int argc, char* argv[], char *arg );				// c2732 에러 [6/28/2018 jobs]
int find_int_arg( int argc, char **argv, char *arg, int def );			// c2732 에러 [6/28/2018 jobs]
float find_float_arg( int argc, char **argv, char *arg, float def );	// c2732 에러 [6/28/2018 jobs]
char *find_char_arg( int argc, char **argv, char *arg, char *def );		// c2732 에러 [6/28/2018 jobs]

/// image.c
image make_image( int w, int h, int c );		// 이미지배열의 메모리 할당
// 하나의 그림파일 자료에서 image 구조자료를 지정한 채널로 탑재한다
image load_image( char *filename, int w, int h, int c );
// 하나의 그림파일 자료에서 image 구조자료를 3채널로 탑재한다
image load_image_color( char *filename, int w, int h );
void test_resize( char *filename );				// c2732 에러 [6/28/2018 jobs]
void save_image( image p, const char *name );	// c2732 에러 [6/28/2018 jobs]
image copy_image( image p );					// 담아둘 메모리를 할당하고, 자료값을 복사하고, 담아둔 주소를 반환한다
void composite_3d( char *f1, char *f2, char *out, int delta );	// c2732 에러 [6/28/2018 jobs]
void random_distort_image( image im, float hue, float saturation, float exposure );	// c2732 에러 [6/28/2018 jobs]
void ghost_image( image source, image dest, int dx, int dy );	// c2732 에러 [6/28/2018 jobs]
void fill_image( image m, float s );			// 이미지 자료값 지정한 값으로 채운다
void rotate_image_cw( image im, int times );	// c2732 에러 [6/28/2018 jobs]
void free_image( image m );						// c2732 에러 [6/28/2018 jobs]
void normalize_image( image p );						// 이미지하나에서 최대값과 최소값을 알아내고 고르기를 한다
void normalize_image_MuRi( image *pImage, int iPanSu );	// 이미지무리에서 최대값과 최소값을 알아내고 고르기를 한다
image **load_alphabet();								// 그림파일로 만들어진 문자이미지를 탑재한다

image resize_image( image im, int w, int h );
void censor_image( image im, int dx, int dy, int w, int h );
image letterbox_image( image im, int w, int h );
image crop_image( image im, int dx, int dy, int w, int h );
image center_crop_image( image im, int w, int h );
image resize_min( image im, int min );
image resize_max( image im, int max );
image threshold_image( image im, float thresh );
image mask_to_rgb( image mask );
void free_matrix( matrix m );
void show_image( image p, const char *name );
void draw_box_width( image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b );
float get_current_rate( network *net );
data load_data_old( char **paths, int n, int m, char **labels, int k, int w, int h );
size_t get_current_batch( network *net );
void constrain_image( image im );
image get_network_image_layer( network *net, int i );
layer get_network_output_layer( network *net );
void top_predictions( network *net, int n, int *index );
void flip_image( image a );
image float_to_image( int w, int h, int c, float *data );	// 자료배열을 이미지로 변경(채널, 가로, 세로 정보를 추가함)
float network_accuracy( network *net, data d );
image grayscale_image( image im );
image rotate_image( image m, float rad );
float box_iou( box a, box b );
data load_all_cifar10();
box_label *read_boxes( char *filename, int *n );
box float_to_box( float *f, int stride );
void draw_detections( image im
					, detection *dets
					, int num
					, float thresh
					, char **names
					, image **alphabet
					, int classes );

matrix network_predict_data( network *net, data test );
image get_network_image( network *net );

int network_width( network *net );
int network_height( network *net );
float *network_predict_image( network *net, image im );
void network_detect( network *net
				, image im
				, float thresh
				, float hier_thresh	// 계층 문턱값
				, float nms
				, detection *dets );
detection *get_network_boxes( network *net
							, int w
							, int h
							, float thresh
							, float hier	// 계층 문턱값
							, int *map
							, int relative
							, int *num );
void free_detections( detection *dets, int n );

void reset_network_state( network *net, int b );

char **get_labels( char *filename );	// 파일에서 분류목록을 추출한다
void do_nms_obj( detection *dets, int total, int classes, float thresh );
void do_nms_sort( detection *dets, int total, int classes, float thresh );

matrix make_matrix( int rows, int cols );

#ifndef __cplusplus
#ifdef OPENCV
image get_image_from_stream( CvCapture *cap );
#endif
#endif

float train_network( network *net, data d );
pthread_t load_data_in_thread( load_args args );
void load_data_blocking( load_args args );
list *get_paths( char *filename );	// 파일에서 목록을 추출하여 추출한 모든 목록을 반환한다
void hierarchy_predictions( float *predictions
						, int n
						, tree *hier
						, int only_leaves
						, int stride );
void change_leaves( tree *t, char *leaf_list );

char *basecfg( char *cfgfile );
void find_replace( char *str, char *orig, char *rep, char *output );
void free_ptrs( void **ptrs, int n );
char *fgetl( FILE *fp );
void strip( char *s );
float sec( clock_t clocks );
/// list.c
void **list_to_array( list *l );	// 목록 배열메모리 할당
void top_k( float *a, int n, int k, int *index );
int *read_map( char *filename );
void error( const char *s );
int max_index( float *a, int n );
int max_int_index( int *a, int n );
int sample_array( float *a, int n );
int *random_index_order( int min, int max );
void free_list( list *l );
float mse_array( float *a, int n );
float variance_array( float *a, int n );
float mag_array( float *a, int n );
void scale_array( float *a, int n, float s );
float mean_array( float *a, int n );
float sum_array( float *a, int n );
void normalize_array( float *a, int n );
int *read_intlist( char *s, int *n, int d );
size_t rand_size_t();
float rand_normal();
float rand_uniform( float min, float max );

/// 실행명령
void run_visualize( int argc, char **argv );	// network.c
void run_yolo( int argc, char **argv );			// yolo.c
void test_detector( char *datacfg
				, char *cfgfile
				, char *weightfile
				, char *filename
				, float thresh
				, float hier_thresh
				, char *outfile
				, int fullscreen );				// detector.c
void run_detector( int argc, char **argv );		// detector.c
void run_coco( int argc, char **argv );			// coco.c
void run_captcha( int argc, char **argv );		// captcha.c
void run_nightmare( int argc, char **argv );	// nightmare.c
void predict_classifier( char *datacfg
				, char *cfgfile
				, char *weightfile
				, char *filename
				, int top );					// classifier.c
void run_classifier( int argc, char **argv );	// classifier.c
void run_regressor( int argc, char **argv );	// regressor.c
void run_segmenter( int argc, char **argv );	// segmenter.c
void run_char_rnn( int argc, char **argv );		// rnn.c
void run_tag( int argc, char **argv );			// tag.c
void run_cifar( int argc, char **argv );		// cifar.c
void run_go( int argc, char **argv );			// go.c
void run_art( int argc, char **argv );			// art.c
void run_super( int argc, char **argv );		// super.c
void run_lsd( int argc, char **argv );			// lsd.c
void run_voxel( int argc, char **argv );		// voxel.c
void run_compare( int argc, char **argv );		// compare.c
void run_dice( int argc, char **argv );			// dice.c
void run_writing( int argc, char **argv );		// writing.c
void run_vid_rnn( int argc, char **argv );		// rnn_vid.c

#ifdef __cplusplus
}
#endif

#endif
