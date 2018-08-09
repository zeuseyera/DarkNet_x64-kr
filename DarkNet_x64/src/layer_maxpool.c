#include "layer_maxpool.h"
#include "cuda.h"
#include <stdio.h>

image get_maxpool_image( maxpool_layer Lyr )
{
	int h = Lyr.out_h;
	int w = Lyr.out_w;
	int c = Lyr.c;
	return float_to_image( w, h, c, Lyr.output );
}

image get_maxpool_delta( maxpool_layer Lyr )
{
	int h = Lyr.out_h;
	int w = Lyr.out_w;
	int c = Lyr.c;
	return float_to_image( w, h, c, Lyr.delta );
}

maxpool_layer make_maxpool_layer( int batch, int h, int w, int c, int size, int stride, int padding )
{
	maxpool_layer Lyr = { 0 };
	Lyr.type = MAXPOOL;
	Lyr.batch = batch;
	Lyr.h = h;
	Lyr.w = w;
	Lyr.c = c;
	Lyr.pad		= padding;
	Lyr.out_w	= (w + 2*padding)/stride;
	Lyr.out_h	= (h + 2*padding)/stride;
	Lyr.out_c	= c;
	Lyr.outputs	= Lyr.out_h * Lyr.out_w * Lyr.out_c;
	Lyr.inputs	= h*w*c;
	Lyr.size	= size;
	Lyr.stride	= stride;
	int output_size = Lyr.out_h * Lyr.out_w * Lyr.out_c * batch;
	Lyr.indexes	= calloc( output_size, sizeof( int ) );
	Lyr.output	= calloc( output_size, sizeof( float ) );
	Lyr.delta	= calloc( output_size, sizeof( float ) );
	Lyr.forward	= forward_maxpool_layer;
	Lyr.backward = backward_maxpool_layer;
	Lyr.BoJa_NaOnGab = visualize_maxpool_layer_output;

	#ifdef GPU
	Lyr.forward_gpu	= forward_maxpool_layer_gpu;
	Lyr.backward_gpu = backward_maxpool_layer_gpu;
	Lyr.indexes_gpu	= cuda_make_int_array( 0, output_size );
	Lyr.output_gpu	= cuda_make_array( Lyr.output, output_size );
	Lyr.delta_gpu	= cuda_make_array( Lyr.delta, output_size );
	#endif

	fprintf( stderr
		, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n"
		, size, size, stride, w, h, c, Lyr.out_w, Lyr.out_h, Lyr.out_c );

	return Lyr;
}

void resize_maxpool_layer( maxpool_layer *Lyr, int w, int h )
{
	Lyr->h = h;
	Lyr->w = w;
	Lyr->inputs	= h*w*Lyr->c;

	Lyr->out_w	= (w + 2*Lyr->pad)/Lyr->stride;
	Lyr->out_h	= (h + 2*Lyr->pad)/Lyr->stride;
	Lyr->outputs	= Lyr->out_w * Lyr->out_h * Lyr->c;
	int output_size = Lyr->outputs * Lyr->batch;

	Lyr->indexes	= realloc( Lyr->indexes, output_size * sizeof( int ) );
	Lyr->output	= realloc( Lyr->output, output_size * sizeof( float ) );
	Lyr->delta	= realloc( Lyr->delta, output_size * sizeof( float ) );

	#ifdef GPU
	cuda_free( (float *)Lyr->indexes_gpu );
	cuda_free( Lyr->output_gpu );
	cuda_free( Lyr->delta_gpu );
	Lyr->indexes_gpu = cuda_make_int_array( 0, output_size );
	Lyr->output_gpu  = cuda_make_array( Lyr->output, output_size );
	Lyr->delta_gpu   = cuda_make_array( Lyr->delta, output_size );
	#endif
}

void forward_maxpool_layer( const maxpool_layer Lyr, network net )
{
	int b, i, j, k, m, n;
	int w_offset = -Lyr.pad;
	int h_offset = -Lyr.pad;

	int h = Lyr.out_h;
	int w = Lyr.out_w;
	int c = Lyr.c;

	for ( b = 0; b < Lyr.batch; ++b )
	{
		for ( k = 0; k < c; ++k )
		{
			for ( i = 0; i < h; ++i )
			{
				for ( j = 0; j < w; ++j )
				{
					int out_index = j + w*(i + h*(k + c*b));
					float max = -FLT_MAX;
					int max_i = -1;

					for ( n = 0; n < Lyr.size; ++n )
					{
						for ( m = 0; m < Lyr.size; ++m )
						{
							int cur_h = h_offset + i*Lyr.stride + n;
							int cur_w = w_offset + j*Lyr.stride + m;
							int index = cur_w + Lyr.w*(cur_h + Lyr.h*(k + b*Lyr.c));
							int valid = ( cur_h >= 0 && cur_h < Lyr.h &&
										  cur_w >= 0 && cur_w < Lyr.w );
							float val = (valid != 0) ? net.input[index] : -FLT_MAX;
							max_i = (val > max) ? index : max_i;
							max   = (val > max) ? val : max;
						}
					}

					Lyr.output[out_index] = max;
					Lyr.indexes[out_index] = max_i;
				}
			}
		}
	}
}

void backward_maxpool_layer( const maxpool_layer Lyr, network net )
{
	int i;
	int h = Lyr.out_h;
	int w = Lyr.out_w;
	int c = Lyr.c;
	for ( i = 0; i < h*w*c*Lyr.batch; ++i )
	{
		int index = Lyr.indexes[i];
		net.delta[index] += Lyr.delta[i];
	}
}
// �ϳ��� �������߰� �ּҸ� �̹����迭�� �ּҸ� ������
image pull_maxpool_out( maxpool_layer Lyr, int nn )
{
	int ww = Lyr.out_w;		// ��� �ʺ�
	int hh = Lyr.out_h;		// ��� ����
	int cc = 1;				// 
	int bo = ww*hh*nn;		// ������ ��
	return float_to_image( ww, hh, cc, Lyr.output + bo );
}
// �޸𸮸� �Ҵ��ϰ� �Ҵ��� �޸𸮿� ���߰��� �����Ѵ�
image *pull_maxpool_image_out( maxpool_layer Lyr )
{
	image *out = calloc( Lyr.out_c, sizeof( image ) );	// ������ ������ŭ �̹����޸� �Ҵ�

	int ii;
	// �����̹��� �ǰ��� �ݺ�
	for ( ii=0; ii < Lyr.out_c; ++ii )
	{
		// ��Ƶ� �޸𸮸� �Ҵ��ϰ�, �ڷᰪ�� �����ϰ�, ��Ƶ� �ּҸ� ��ȯ�Ѵ�
		out[ii] = copy_image( pull_maxpool_out( Lyr, ii ) );
		normalize_image( out[ii] );	//�����Ǻ��� ���⸦ �ϸ� Ư¡ǥ���� �Ǵ°�???
	}

	//normalize_image_MuRi( out, Lyr.n );	//�̹������� ��ü�� �����Ѵ�

	return out;
}
// ��δ� ���°� �ð�ȭ
image *visualize_maxpool_layer_output( maxpool_layer Lyr, char *window, image *prev_out )
{
	// ������ ������ŭ �̹����޸𸮸� �Ҵ��ϰ� ��Ƶ� �ּҸ� �����Ѵ�
	image *single_out = pull_maxpool_image_out( Lyr );
	// �̹����� ���簢������ �迭�� �����ϰ� ȭ�鿡 �����ش�
	show_images( single_out, Lyr.out_c, window );

	char buff[256];

	sprintf_s( buff, 256, "%s: Output", window );

	return single_out;
}
