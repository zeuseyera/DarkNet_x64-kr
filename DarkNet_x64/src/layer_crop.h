#ifndef CROP_LAYER_H
#define CROP_LAYER_H

#include "image.h"
#include "layer.h"
#include "network.h"

typedef layer crop_layer;

image get_crop_image( crop_layer l );
crop_layer make_crop_layer( int batch
						, int h
						, int w
						, int c
						, int crop_height
						, int crop_width
						, int flip
						, float angle
						, float saturation
						, float exposure );
void forward_crop_layer( const crop_layer l, network net );
void resize_crop_layer( layer *l, int w, int h );
// 크롭층 나온값 시각화
image *visualize_crop_layer_output( crop_layer Lyr, char *window, image *prev_out );

#ifdef GPU
void forward_crop_layer_gpu( crop_layer l, network net );
#endif

#endif

