#include "layer_route.h"
#include "cuda.h"
#include "blas.h"

#include <stdio.h>

route_layer make_route_layer( int batch, int nn, int *input_layers, int *input_sizes )
{
	fprintf( stderr, "route " );
	route_layer rLyr = { 0 };
	rLyr.type	= ROUTE;
	rLyr.batch	= batch;
	rLyr.n		= nn;
	rLyr.input_layers	= input_layers;
	rLyr.input_sizes	= input_sizes;

	int ii;
	int outputs = 0;
	for ( ii=0; ii < nn; ++ii )
	{
		fprintf( stderr, " %d", input_layers[ii] );
		outputs += input_sizes[ii];
	}
	fprintf( stderr, "\n" );

	rLyr.outputs	= outputs;	// 출력개수
	rLyr.inputs		= outputs;	// 모든 경로의 출력개수
	rLyr.delta		= calloc( outputs*batch, sizeof( float ) );
	rLyr.output		= calloc( outputs*batch, sizeof( float ) );;

	rLyr.forward	= forward_route_layer;
	rLyr.backward	= backward_route_layer;

	#ifdef GPU
	rLyr.forward_gpu	= forward_route_layer_gpu;
	rLyr.backward_gpu	= backward_route_layer_gpu;

	rLyr.delta_gpu		= cuda_make_array( rLyr.delta, outputs*batch );
	rLyr.output_gpu		= cuda_make_array( rLyr.output, outputs*batch );
	#endif

	return rLyr;
}

void resize_route_layer( route_layer *rLyr, network *net )
{
	int ii;
	layer first = net->layers[rLyr->input_layers[0]];

	rLyr->out_w = first.out_w;
	rLyr->out_h = first.out_h;
	rLyr->out_c = first.out_c;

	rLyr->outputs = first.outputs;
	rLyr->input_sizes[0] = first.outputs;

	for ( ii=1; ii < rLyr->n; ++ii )
	{
		int index	= rLyr->input_layers[ii];
		layer next	= net->layers[index];

		rLyr->outputs			+= next.outputs;
		rLyr->input_sizes[ii]	= next.outputs;

		if ( next.out_w == first.out_w && next.out_h == first.out_h )
		{
			rLyr->out_c += next.out_c;
		}
		else
		{
			printf( "%d %d, %d %d\n", next.out_w, next.out_h, first.out_w, first.out_h );
			rLyr->out_h = rLyr->out_w = rLyr->out_c = 0;
		}
	}

	rLyr->inputs	= rLyr->outputs;
	rLyr->delta		= realloc( rLyr->delta, rLyr->outputs*rLyr->batch*sizeof( float ) );
	rLyr->output	= realloc( rLyr->output, rLyr->outputs*rLyr->batch*sizeof( float ) );

	#ifdef GPU
	cuda_free( rLyr->output_gpu );
	cuda_free( rLyr->delta_gpu );
	rLyr->output_gpu	= cuda_make_array( rLyr->output, rLyr->outputs*rLyr->batch );
	rLyr->delta_gpu		= cuda_make_array( rLyr->delta, rLyr->outputs*rLyr->batch );
	#endif

}

void forward_route_layer( const route_layer rLyr, network net )
{
	int ii, jj;
	int offset = 0;
	for ( ii=0; ii < rLyr.n; ++ii )
	{
		int index		= rLyr.input_layers[ii];
		float *input	= net.layers[index].output;
		int input_size	= rLyr.input_sizes[ii];

		for ( jj=0; jj < rLyr.batch; ++jj )
		{
			copy_cpu( input_size
					, input + jj*input_size
					, 1
					, rLyr.output + offset + jj*rLyr.outputs
					, 1 );
		}

		offset += input_size;
	}
}

void backward_route_layer( const route_layer rLyr, network net )
{
	int ii, jj;
	int offset = 0;
	for ( ii=0; ii < rLyr.n; ++ii )
	{
		int index		= rLyr.input_layers[ii];
		float *delta	= net.layers[index].delta;
		int input_size	= rLyr.input_sizes[ii];

		for ( jj=0; jj < rLyr.batch; ++jj )
		{
			axpy_cpu( input_size
					, 1
					, rLyr.delta + offset + jj*rLyr.outputs
					, 1
					, delta + jj*input_size
					, 1 );
		}

		offset += input_size;
	}
}

#ifdef GPU
void forward_route_layer_gpu( const route_layer rLyr, network net )
{
	int ii, jj;
	int offset = 0;
	for ( ii = 0; ii < rLyr.n; ++ii )
	{
		int index		= rLyr.input_layers[ii];
		float *input	= net.layers[index].output_gpu;
		int input_size	= rLyr.input_sizes[ii];

		for ( jj = 0; jj < rLyr.batch; ++jj )
		{
			copy_gpu( input_size
					, input + jj*input_size
					, 1
					, rLyr.output_gpu + offset + jj*rLyr.outputs
					, 1 );
		}

		offset += input_size;
	}
}

void backward_route_layer_gpu( const route_layer rLyr, network net )
{
	int ii, jj;
	int offset = 0;
	for ( ii=0; ii < rLyr.n; ++ii )
	{
		int index		= rLyr.input_layers[ii];
		float *delta	= net.layers[index].delta_gpu;
		int input_size	= rLyr.input_sizes[ii];

		for ( jj=0; jj < rLyr.batch; ++jj )
		{
			axpy_gpu( input_size
					, 1
					, rLyr.delta_gpu + offset + jj*rLyr.outputs
					, 1
					, delta + jj*input_size
					, 1 );
		}

		offset += input_size;
	}
}
#endif
