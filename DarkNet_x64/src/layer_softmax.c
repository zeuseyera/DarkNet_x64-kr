#include "layer_softmax.h"
#include "blas.h"
#include "cuda.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

softmax_layer make_softmax_layer( int batch, int inputs, int groups )
{
	assert( inputs%groups == 0 );
	fprintf( stderr, "softmax                                        %4d\n", inputs );
	softmax_layer Lyr = { 0 };
	Lyr.type	= SOFTMAX;
	Lyr.batch	= batch;
	Lyr.groups	= groups;
	Lyr.inputs	= inputs;
	Lyr.outputs	= inputs;
	Lyr.loss	= calloc( inputs*batch, sizeof( float ) );
	Lyr.output	= calloc( inputs*batch, sizeof( float ) );
	Lyr.delta	= calloc( inputs*batch, sizeof( float ) );
	Lyr.cost	= calloc( 1, sizeof( float ) );

	Lyr.forward			= forward_softmax_layer;
	Lyr.backward		= backward_softmax_layer;
	Lyr.BoJa_NaOnGab	= visualize_softmax_layer_output;

	#ifdef GPU
	Lyr.forward_gpu		= forward_softmax_layer_gpu;
	Lyr.backward_gpu	= backward_softmax_layer_gpu;

	Lyr.output_gpu		= cuda_make_array( Lyr.output, inputs*batch );
	Lyr.loss_gpu		= cuda_make_array( Lyr.loss, inputs*batch );
	Lyr.delta_gpu		= cuda_make_array( Lyr.delta, inputs*batch );
	#endif

	return Lyr;
}

void forward_softmax_layer( const softmax_layer Lyr, network net )
{
	if ( Lyr.softmax_tree )
	{
		int ii;
		int count = 0;

		for ( ii=0; ii < Lyr.softmax_tree->groups; ++ii )
		{
			int group_size = Lyr.softmax_tree->group_size[ii];
			softmax_cpu( net.input + count
					, group_size
					, Lyr.batch, Lyr.inputs
					, 1
					, 0
					, 1
					, Lyr.temperature
					, Lyr.output + count );
			count += group_size;
		}
	}
	else
	{
		softmax_cpu( net.input
				, Lyr.inputs/Lyr.groups
				, Lyr.batch
				, Lyr.inputs
				, Lyr.groups
				, Lyr.inputs/Lyr.groups
				, 1
				, Lyr.temperature
				, Lyr.output );
	}

	if ( net.truth )
	{
		softmax_x_ent_cpu( Lyr.batch*Lyr.inputs, Lyr.output, net.truth, Lyr.delta, Lyr.loss );
		Lyr.cost[0] = sum_array( Lyr.loss, Lyr.batch*Lyr.inputs );
	}
}

void backward_softmax_layer( const softmax_layer Lyr, network net )
{
	axpy_cpu( Lyr.inputs*Lyr.batch, 1, Lyr.delta, 1, net.delta, 1 );
}

#ifdef GPU

void pull_softmax_layer_output( const softmax_layer Lyr )
{
	cuda_pull_array( Lyr.output_gpu, Lyr.output, Lyr.inputs*Lyr.batch );
}

void forward_softmax_layer_gpu( const softmax_layer Lyr, network net )
{
	if ( Lyr.softmax_tree )
	{
		softmax_tree( net.input_gpu
					, 1
					, Lyr.batch
					, Lyr.inputs
					, Lyr.temperature
					, Lyr.output_gpu
					, *Lyr.softmax_tree );
		/*
		int i;
		int count = 0;
		for ( i = 0; i < Lyr.softmax_tree->groups; ++i )
		{
			int group_size = Lyr.softmax_tree->group_size[i];
			softmax_gpu( net.input_gpu + count
					, group_size
					, Lyr.batch
					, Lyr.inputs
					, 1
					, 0
					, 1
					, Lyr.temperature
					, Lyr.output_gpu + count );
			count += group_size;
		}
		*/
	}
	else
	{
		if ( Lyr.spatial )
		{
			softmax_gpu( net.input_gpu
				, Lyr.c
				, Lyr.batch*Lyr.c
				, Lyr.inputs/Lyr.c
				, Lyr.w*Lyr.h
				, 1
				, Lyr.w*Lyr.h
				, 1
				, Lyr.output_gpu );
		}
		else
		{
			softmax_gpu( net.input_gpu
				, Lyr.inputs/Lyr.groups
				, Lyr.batch
				, Lyr.inputs
				, Lyr.groups
				, Lyr.inputs/Lyr.groups
				, 1
				, Lyr.temperature
				, Lyr.output_gpu );
		}
	}

	if ( net.truth )
	{
		softmax_x_ent_gpu( Lyr.batch*Lyr.inputs, Lyr.output_gpu, net.truth_gpu, Lyr.delta_gpu, Lyr.loss_gpu );

		if ( Lyr.softmax_tree )
		{
			mask_gpu( Lyr.batch*Lyr.inputs, Lyr.delta_gpu, SECRET_NUM, net.truth_gpu, 0 );
			mask_gpu( Lyr.batch*Lyr.inputs, Lyr.loss_gpu, SECRET_NUM, net.truth_gpu, 0 );
		}

		cuda_pull_array( Lyr.loss_gpu, Lyr.loss, Lyr.batch*Lyr.inputs );
		Lyr.cost[0] = sum_array( Lyr.loss, Lyr.batch*Lyr.inputs );
	}
}

void backward_softmax_layer_gpu( const softmax_layer Lyr, network net )
{
	axpy_gpu( Lyr.batch*Lyr.inputs, 1, Lyr.delta_gpu, 1, net.delta_gpu, 1 );
}

#endif

// �ϳ��� ���°� �ּҸ� �̹����迭�� �ּҸ� ������
image pull_softmax_out( softmax_layer Lyr, int nn )
{
	int ww = Lyr.w;		// ��� �ʺ�
	int hh = Lyr.h;		// ��� ����
	int cc = 1;				// 
	int bo = ww*hh*nn;		// ������ ��
	return float_to_image( ww, hh, cc, Lyr.output + bo );
}
// �޸𸮸� �Ҵ��ϰ� �Ҵ��� �޸𸮿� ���°��� �����Ѵ�
image *pull_softmax_image_out( softmax_layer Lyr )
{
	image *out = calloc( Lyr.outputs, sizeof( image ) );	// ������ ������ŭ �̹����޸� �Ҵ�

	int ii;
	// �����̹��� �ǰ��� �ݺ�
	for ( ii=0; ii < Lyr.outputs; ++ii )
	{
		// ��Ƶ� �޸𸮸� �Ҵ��ϰ�, �ڷᰪ�� �����ϰ�, ��Ƶ� �ּҸ� ��ȯ�Ѵ�
		out[ii] = copy_image( pull_softmax_out( Lyr, ii ) );
	}

	normalize_image_MuRi( out, Lyr.outputs );	//�̹������� ��ü�� �����Ѵ�

	return out;
}
// Ȱ���� ���°� �ð�ȭ
image *visualize_softmax_layer_output( softmax_layer Lyr, char *window, image *prev_out )
{
	// ������ ������ŭ �̹����޸𸮸� �Ҵ��ϰ� ��Ƶ� �ּҸ� �����Ѵ�
	image *single_out = pull_softmax_image_out( Lyr );
	// �̹����� ���簢������ �迭�� �����ϰ� ȭ�鿡 �����ش�
	show_images( single_out, Lyr.outputs, window );

	char buff[256];

	sprintf_s( buff, 256, "%s: Output", window );

	return single_out;
}
