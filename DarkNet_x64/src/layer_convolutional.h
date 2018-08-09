#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

typedef layer convolutional_layer;

//void denormalize_convolutional_layer( layer l );			// c2732 에러 [6/28/2018 jobs]
//void rgbgr_weights( layer l );								// c2732 에러 [6/28/2018 jobs]
//void rescale_weights( layer l, float scale, float trans );	// c2732 에러 [6/28/2018 jobs]
//image *get_weights( layer l );								// c2732 에러 [6/28/2018 jobs]

#ifdef GPU
void forward_convolutional_layer_gpu( convolutional_layer layer, network net );
void backward_convolutional_layer_gpu( convolutional_layer layer, network net );
void update_convolutional_layer_gpu( convolutional_layer layer, update_args a );

void push_convolutional_layer( convolutional_layer layer );
void pull_convolutional_layer( convolutional_layer layer );

void add_bias_gpu( float *output, float *biases, int batch, int n, int size );
void backward_bias_gpu( float *bias_updates, float *delta, int batch, int n, int size );
void adam_update_gpu( float *w
					, float *d
					, float *m
					, float *v
					, float B1
					, float B2
					, float eps
					, float decay
					, float rate
					, int n
					, int batch
					, int t );
#ifdef CUDNN
void cudnn_convolutional_setup( layer *l );
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
											, int adam );
void resize_convolutional_layer( convolutional_layer *layer, int w, int h );
void forward_convolutional_layer( const convolutional_layer layer, network net );
void update_convolutional_layer( convolutional_layer layer, update_args a );
// 나선층 가중값 시각화 [7/15/2018 jobs]
image *visualize_convolutional_layer_output( convolutional_layer Lyr, char *window, image *prev_out );
image *visualize_convolutional_layer_weight( convolutional_layer Lyr, char *window, image *prev_weights );
void binarize_weights( float *weights, int n, int size, float *binary );
void swap_binary( convolutional_layer *l );
void binarize_weights2( float *weights, int n, int size, char *binary, float *scales );

void backward_convolutional_layer( convolutional_layer layer, network net );

void add_bias( float *output, float *biases, int batch, int n, int size );
void backward_bias( float *bias_updates, float *delta, int batch, int n, int size );

image get_convolutional_image( convolutional_layer layer );
image get_convolutional_delta( convolutional_layer layer );
image get_convolutional_weight( convolutional_layer layer, int i );

int convolutional_out_height( convolutional_layer layer );
int convolutional_out_width( convolutional_layer layer );

#endif

