#ifndef DATA_H
#define DATA_H

#define HAVE_STRUCT_TIMESPEC	//error C2011: 'timespec': 'struct' 형식 재정의
#include <pthread.h>

#if defined(_MSC_VER) && _MSC_VER < 1900
#define inline __inline
#endif

#include "matrix.h"
#include "list.h"
#include "image.h"
#include "tree.h"

static inline float distance_from_edge( int x, int max )
{
	int dx = (max/2) - x;
	if ( dx < 0 ) dx = -dx;
	dx = (max/2) + 1 - dx;
	dx *= 2;
	float dist = (float)dx/max;
	if ( dist > 1 ) dist = 1;
	return dist;
}
// 신경망 입력값, 목표값, 출력값 구조
typedef struct
{
	int w, h;	// 가로, 세로
	matrix X;	// 신경망 입력값
	matrix y;	// 목표값?
	int shallow;
	int *num_boxes;	// 경계구역 개수
	box **boxes;	// 경계구역
} data;

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
} data_type;

typedef struct load_args
{
	int threads;	// 쓰레드개수
	char **paths;	// 사비할 파일이름(char 형)
	char *path;
	int n;			// 사리수 x subdivisions x GPU 개수
	int m;			// 자료(수련, 평가 등의) 개수
	char **labels;	// 딱지 목록
	int h;		// 사비 세로
	int w;		// 사비 가로
	int out_w;
	int out_h;
	int nh;
	int nw;
	int num_boxes;
	int min;	// crop min
	int max;	// crop max
	int size;
	int classes;	// 분류개수
	int background;
	int scale;
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
} load_args;	// 결정값 구조

typedef struct
{
	int id;
	float x, y, w, h;
	float left, right, top, bottom;
} box_label;

void free_data( data d );

pthread_t load_data( load_args args );	// 자료를 탑재하는  쓰레드 하나

pthread_t load_data_in_thread( load_args args );

void print_letters( float *pred, int n );
data load_data_captcha( char **paths, int n, int m, int k, int w, int h );
data load_data_captcha_encode( char **paths, int n, int m, int w, int h );
data load_data_old( char **paths, int n, int m, char **labels, int k, int w, int h );
data load_data_detection( int n
				, char **paths
				, int m
				, int w
				, int h
				, int boxes
				, int classes
				, float jitter
				, float hue
				, float saturation
				, float exposure );
data load_data_tag( char **paths
				, int n
				, int m
				, int k
				, int min
				, int max
				, int size
				, float angle
				, float aspect
				, float hue
				, float saturation
				, float exposure );
matrix load_image_augment_paths( char **paths
				, int n
				, int min
				, int max
				, int size
				, float angle
				, float aspect
				, float hue
				, float saturation
				, float exposure );
data load_data_super( char **paths, int n, int m, int w, int h, int scale );
data load_data_augment( char **paths
					, int n
					, int m
					, char **labels
					, int k
					, tree *hierarchy
					, int min
					, int max
					, int size
					, float angle
					, float aspect
					, float hue
					, float saturation
					, float exposure );
data load_go( char *filename );

box_label *read_boxes( char *filename, int *n );
data load_cifar10_data( char *filename );
data load_all_cifar10();

data load_data_writing( char **paths, int n, int m, int w, int h, int out_w, int out_h );

list *get_paths( char *filename );	// 파일에서 목록을 추출하여 추출한 모든 목록을 반환한다
char **get_labels( char *filename );	// 파일에서 분류목록을 추출한다
void get_random_batch( data d, int n, float *X, float *y );
data get_data_part( data d, int part, int total );
data get_random_data( data d, int num );
void get_next_batch( data d, int n, int offset, float *X, float *y );	// 입력값과 목표값을 가져온다
data load_categorical_data_csv( char *filename, int target, int k );
void normalize_data_rows( data d );
void scale_data_rows( data d, float s );
void translate_data_rows( data d, float s );
void randomize_data( data d );
data *split_data( data d, int part, int total );
data concat_data( data d1, data d2 );
data concat_datas( data *d, int n );
void fill_truth( char *path, char **labels, int k, float *truth );

#endif
