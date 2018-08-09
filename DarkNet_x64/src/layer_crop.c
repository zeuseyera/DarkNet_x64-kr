#include "layer_crop.h"
#include "cuda.h"
#include <stdio.h>

void backward_crop_layer( const crop_layer l, network net )
{
}
void backward_crop_layer_gpu( const crop_layer l, network net )
{
}

crop_layer make_crop_layer( int batch
						, int h
						, int w
						, int c
						, int crop_height
						, int crop_width
						, int flip
						, float angle
						, float saturation
						, float exposure )
{
	fprintf( stderr
		, "Crop Layer: %d x %d -> %d x %d x %d image\n"
		, h, w, crop_height, crop_width, c );

	crop_layer Lyr = { 0 };
	Lyr.type	= CROP;
	Lyr.batch	= batch;
	Lyr.h		= h;
	Lyr.w		= w;
	Lyr.c		= c;
	Lyr.scale		= (float)crop_height / h;
	Lyr.flip		= flip;
	Lyr.angle		= angle;
	Lyr.saturation	= saturation;
	Lyr.exposure	= exposure;
	Lyr.out_w		= crop_width;
	Lyr.out_h		= crop_height;
	Lyr.out_c		= c;
	Lyr.inputs		= Lyr.w * Lyr.h * Lyr.c;
	Lyr.outputs		= Lyr.out_w * Lyr.out_h * Lyr.out_c;
	Lyr.output		= calloc( Lyr.outputs*batch, sizeof( float ) );
	Lyr.forward		= forward_crop_layer;
	Lyr.backward	= backward_crop_layer;
	Lyr.BoJa_NaOnGab = visualize_crop_layer_output;

	#ifdef GPU
	Lyr.forward_gpu	= forward_crop_layer_gpu;
	Lyr.backward_gpu = backward_crop_layer_gpu;
	Lyr.output_gpu	= cuda_make_array( Lyr.output, Lyr.outputs*batch );
	Lyr.rand_gpu	= cuda_make_array( 0, Lyr.batch*8 );
	#endif

	return Lyr;
}

void resize_crop_layer( layer *Lyr, int w, int h )
{
	Lyr->w = w;
	Lyr->h = h;

	Lyr->out_w =  Lyr->scale*w;
	Lyr->out_h =  Lyr->scale*h;

	Lyr->inputs = Lyr->w * Lyr->h * Lyr->c;
	Lyr->outputs = Lyr->out_h * Lyr->out_w * Lyr->out_c;

	Lyr->output = realloc( Lyr->output, Lyr->batch*Lyr->outputs*sizeof( float ) );

	#ifdef GPU
	cuda_free( Lyr->output_gpu );
	Lyr->output_gpu = cuda_make_array( Lyr->output, Lyr->outputs*Lyr->batch );
	#endif
}

void forward_crop_layer( const crop_layer Lyr, network net )
{
	int i, j, c, b, row, col;
	int index;
	int count = 0;
	int flip = (Lyr.flip && rand()%2);
	int dh = rand()%(Lyr.h - Lyr.out_h + 1);
	int dw = rand()%(Lyr.w - Lyr.out_w + 1);
	float scale = 2;
	float trans = -1;

	if ( Lyr.noadjust )
	{
		scale = 1;
		trans = 0;
	}

	if ( !net.train )
	{
		flip = 0;
		dh = (Lyr.h - Lyr.out_h)/2;
		dw = (Lyr.w - Lyr.out_w)/2;
	}

	for ( b = 0; b < Lyr.batch; ++b )
	{
		for ( c = 0; c < Lyr.c; ++c )
		{
			for ( i = 0; i < Lyr.out_h; ++i )
			{
				for ( j = 0; j < Lyr.out_w; ++j )
				{
					if ( flip )
					{
						col = Lyr.w - dw - j - 1;
					}
					else
					{
						col = j + dw;
					}

					row = i + dh;
					index = col+Lyr.w*(row+Lyr.h*(c + Lyr.c*b));
					Lyr.output[count++] = net.input[index]*scale + trans;
				}
			}
		}
	}
}

image get_crop_image( crop_layer Lyr )
{
	int ww = Lyr.out_w;
	int hh = Lyr.out_h;
	int cc = Lyr.out_c;
	return float_to_image( ww, hh, cc, Lyr.output );
}

// ������ ��°� �ּҸ� �̹����迭�� �ּҸ� ������
image pull_crop_out( crop_layer Lyr, int nn )
{
	int ww = Lyr.out_w;		// ��� �ʺ�
	int hh = Lyr.out_h;		// ��� ����
	int cc = 1;				// 
	int bo = ww*hh*nn;		// ������ ��
	return float_to_image( ww, hh, cc, Lyr.output + bo );
}
// �޸𸮸� �Ҵ��ϰ� �Ҵ��� �޸𸮿� ���߰��� �����Ѵ�
image *pull_crop_image_out( crop_layer Lyr )
{
	image *out = calloc( Lyr.out_c, sizeof( image ) );	// ������ ������ŭ �̹����޸� �Ҵ�

	int ii;
	// �����̹��� �ǰ��� �ݺ�
	for ( ii=0; ii < Lyr.out_c; ++ii )
	{
		// ��Ƶ� �޸𸮸� �Ҵ��ϰ�, �ڷᰪ�� �����ϰ�, ��Ƶ� �ּҸ� ��ȯ�Ѵ�
		out[ii] = copy_image( pull_crop_out( Lyr, ii ) );
		//normalize_image( weights[ii] );	//�����Ǻ��� ���⸦ �ϸ� Ư¡ǥ���� �Ǵ°�???
	}

	normalize_image_MuRi( out, Lyr.out_c );	//�̹������� ��ü�� �����Ѵ�

	return out;
}
// ������ ���°� �ð�ȭ
image *visualize_crop_layer_output( crop_layer Lyr, char *window, image *prev_out )
{
	// ������ ������ŭ �̹����޸𸮸� �Ҵ��ϰ� ��Ƶ� �ּҸ� �����Ѵ�
	image *single_out = pull_crop_image_out( Lyr );
	// �̹����� ���簢������ �迭�� �����ϰ� ȭ�鿡 �����ش�
	show_images( single_out, Lyr.out_c, window );

	char buff[256];

	sprintf_s( buff, 256, "%s: Output", window );

	return single_out;
}
