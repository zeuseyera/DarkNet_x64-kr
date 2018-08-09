#include "layer_activation.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

layer make_activation_layer( int batch, int inputs, ACTIVATION activation )
{
	layer Lyr = { 0 };
	Lyr.type	= ACTIVE;

	Lyr.inputs	= inputs;
	Lyr.outputs	= inputs;
	Lyr.batch		= batch;

	Lyr.output	= calloc( batch*inputs, sizeof( float* ) );
	Lyr.delta	= calloc( batch*inputs, sizeof( float* ) );

	Lyr.forward		= forward_activation_layer;
	Lyr.backward	= backward_activation_layer;

	#ifdef GPU
	Lyr.forward_gpu		= forward_activation_layer_gpu;
	Lyr.backward_gpu	= backward_activation_layer_gpu;

	Lyr.output_gpu	= cuda_make_array( Lyr.output, inputs*batch );
	Lyr.delta_gpu	= cuda_make_array( Lyr.delta, inputs*batch );
	#endif

	Lyr.activation	= activation;

	fprintf( stderr, "Activation Layer: %d inputs\n", inputs );

	return Lyr;
}

void forward_activation_layer( layer Lyr, network net )
{
	copy_cpu( Lyr.outputs*Lyr.batch, net.input, 1, Lyr.output, 1 );
	activate_array( Lyr.output, Lyr.outputs*Lyr.batch, Lyr.activation );
}

void backward_activation_layer( layer Lyr, network net )
{
	gradient_array( Lyr.output, Lyr.outputs*Lyr.batch, Lyr.activation, Lyr.delta );
	copy_cpu( Lyr.outputs*Lyr.batch, Lyr.delta, 1, net.delta, 1 );
}

#ifdef GPU

void forward_activation_layer_gpu( layer Lyr, network net )
{
	copy_gpu( Lyr.outputs*Lyr.batch, net.input_gpu, 1, Lyr.output_gpu, 1 );
	activate_array_gpu( Lyr.output_gpu, Lyr.outputs*Lyr.batch, Lyr.activation );
}

void backward_activation_layer_gpu( layer Lyr, network net )
{
	gradient_array_gpu( Lyr.output_gpu, Lyr.outputs*Lyr.batch, Lyr.activation, Lyr.delta_gpu );
	copy_gpu( Lyr.outputs*Lyr.batch, Lyr.delta_gpu, 1, net.delta_gpu, 1 );
}
#endif

