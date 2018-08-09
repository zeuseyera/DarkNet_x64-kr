#include "image.h"
#include "utils.h"
#include "blas.h"
#include "cuda.h"
#include <stdio.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int windows = 0;

float colors[6][3] =
{	//  B, G, R
	  { 1, 0, 1 }
	, { 0, 0, 1 }
	, { 0, 1, 1 }
	, { 0, 1, 0 }
	, { 1, 1, 0 }
	, { 1, 0, 0 }
};
// colors �迭�� ���� ������ ǥ�� ���� ������ ���� ����������� ��ȯ�Ѵ�
float get_color( int cc, int xx, int max )	// ��, ��, �ִ밪
{
	// R 
	float ratio = ((float)xx/max) * 5;
	int ii = floor( ratio );	// ����
	int jj = ceil( ratio );		// �ø�
	ratio -= ii;
	float rr = (1-ratio) * colors[ii][cc] + ratio*colors[jj][cc];
	//printf("%f\n", r);
	return rr;
}

image mask_to_rgb( image mask )
{
	int n = mask.c;
	image im = make_image( mask.w, mask.h, 3 );
	int i, j;
	for ( j = 0; j < n; ++j )
	{
		int offset = j*123457 % n;
		float red = get_color( 2, offset, n );
		float green = get_color( 1, offset, n );
		float blue = get_color( 0, offset, n );

		for ( i = 0; i < im.w*im.h; ++i )
		{
			im.data[i + 0*im.w*im.h] += mask.data[j*im.h*im.w + i]*red;
			im.data[i + 1*im.w*im.h] += mask.data[j*im.h*im.w + i]*green;
			im.data[i + 2*im.w*im.h] += mask.data[j*im.h*im.w + i]*blue;
		}
	}
	return im;
}

// ȭ���� ���� �����´�
static float get_pixel( image img, int xx, int yy, int cc )
{
	assert( xx < img.w && yy < img.h && cc < img.c );

	return img.data[cc*img.h*img.w + yy*img.w + xx];
}
// ȭ���� ���� �����´�
static float get_pixel_extend( image img, int xx, int yy, int cc )
{
	if ( xx < 0 || xx >= img.w || yy < 0 || yy >= img.h ) return 0;

/*	if(x < 0) x = 0;
	if(x >= img.w) x = img.w-1;
	if(y < 0) y = 0;
	if(y >= img.h) y = img.h-1;
*/
	if ( cc < 0 || cc >= img.c ) return 0;
	return get_pixel( img, xx, yy, cc );
}
// ������ ȭ�ҿ� ���� �����Ѵ�
static void set_pixel( image img, int xx, int yy, int cc, float val )
{
	assert( xx < img.w && yy < img.h && cc < img.c );

	if ( xx < 0 || yy < 0 || cc < 0 || xx >= img.w || yy >= img.h || cc >= img.c ) return;

	img.data[cc*img.h*img.w + yy*img.w + xx] = val;
}
// ������ ȭ�ҿ� ���� ���Ѵ�
static void add_pixel( image img, int xx, int yy, int cc, float val )
{
	assert( xx < img.w && yy < img.h && cc < img.c );

	img.data[cc*img.h*img.w + yy*img.w + xx] += val;
}

// ������ �Ǽ���ǥ�� ���� ������ǥ�� ���߼��� �����Ͽ� ���� ����
static float bilinear_interpolate( image im, float xx, float yy, int cc )
{
	int ix = (int)floorf( xx );
	int iy = (int)floorf( yy );

	float dx = xx - ix;
	float dy = yy - iy;

	float val = (1-dy) * (1-dx) * get_pixel_extend( im, ix, iy, cc ) +
				dy     * (1-dx) * get_pixel_extend( im, ix, iy+1, cc ) +
				(1-dy) *   dx   * get_pixel_extend( im, ix+1, iy, cc ) +
				dy     *   dx   * get_pixel_extend( im, ix+1, iy+1, cc );
	return val;
}

// ����ȭ�ҿ� �纻ȭ�Ҹ� ���Ͽ� �纻�̹��� ȭ�Ұ��� �����Ѵ�
void composite_image( image source, image dest, int xs, int ys )
{
	int xx, yy, cc;
	// �� �ݺ�
	for ( cc=0; cc < source.c; ++cc )
	{
		// ���� �ݺ�
		for ( yy=0; yy < source.h; ++yy )
		{
			// ���� �ݺ�
			for ( xx=0; xx < source.w; ++xx )
			{
				float val	= get_pixel( source, xx, yy, cc );
				float val2	= get_pixel_extend( dest, xs+xx, ys+yy, cc );
				set_pixel( dest, xs+xx, ys+yy, cc, val * val2 );
			}
		}
	}
}
// �̹����� �׵θ��� �߰��� �� �̹����� ��ȯ�Ѵ�
image border_image( image src, int border )
{
	// �׵θ��� �߰��� �޸� �Ҵ�
	image sae = make_image( src.w + 2*border, src.h + 2*border, src.c );

	int xx, yy, cc;

	for ( cc=0; cc < sae.c; ++cc )
	{
		for ( yy=0; yy < sae.h; ++yy )
		{
			for ( xx=0; xx < sae.w; ++xx )
			{
				float val = get_pixel_extend( src, xx - border, yy - border, cc );

				if ( xx - border < 0		||
					 xx - border >= src.w	||
					 yy - border < 0		||
					 yy - border >= src.h )
				{
					val = 1;
				}

				set_pixel( sae, xx, yy, cc, val );
			}
		}
	}

	return sae;
}
// img_a �� img_b �� ���� �� �̹����� ��ȯ�Ѵ�
image tile_images( image img_a, image img_b, int xs )
{
	// ������ �ڷᰡ ������ �纻�̹����� �����Ͽ� ��ȯ�Ѵ�
	if ( img_a.w == 0 ) return copy_image( img_b );
	// ������ �̹����� ���� ũ���� �޸𸮸� �ٽ� �Ҵ��Ѵ�
	image sae = make_image( img_a.w + img_b.w + xs
						 , (img_a.h > img_b.h) ? img_a.h : img_b.h
						 , (img_a.c > img_b.c) ? img_a.c : img_b.c );

	fill_cpu( sae.w*sae.h*sae.c, 1.0f, sae.data, 1 );	// composite_image �� ����ϱ� ���� 1.0 ���� ä��
	embed_image( img_a, sae, 0, 0 );					// ���̹����� img_a �� �߰�
	composite_image( img_b, sae, img_a.w + xs, 0 );		// ���̹����� img_b �� �߰�(embed_image �� ����ص� ��)

	return sae;
}
// ���ڿ��� ������ ũ���� �̹����� ��ȯ�ϰ� �׵θ� ������ �߰��Ͽ� �̹����� ��ȯ
image get_label( image **characters, char *string, int size )
{
	size = size/10;	// �����̹��� ũ�� ����
	if ( size > 7 ) size = 7;

	image label = make_empty_image( 0, 0, 0 );

	while ( *string )
	{
		// ���ڱ׸� �迭���� �����ϳ��� �ش��ϴ� �̹����� �����Ѵ�
		image lbl = characters[size][(int)*string];
		// �ΰ��� �̹����� ���Ѵ�
		image sae = tile_images( label, lbl, -size - 1 + (size+1)/2 );
		free_image( label );	// ���� �Ҵ��� �̹����� ����ϱ� ���� ������ �Ҵ��� �޸� ����
		label = sae;
		++string;
	}

	image sae = border_image( label, label.h*0.25 );
	free_image( label );

	return sae;
}
// �̹����� ������ �߰��Ѵ�		// xs, ys ��ġ�ٲ� [7/14/2018 jobs]
void draw_label( image src, int xs, int ys, image label, const float *rgb )
{
	int ww = label.w;
	int hh = label.h;
	if ( ys - hh >= 0 ) ys = ys - hh;

	int xx, yy, cc;
	for ( yy=0; yy < hh && ys + yy < src.h; ++yy )
	{
		for ( xx=0; xx < ww && xs + xx < src.w; ++xx )
		{
			for ( cc=0; cc < label.c; ++cc )
			{
				float val = get_pixel( label, xx, yy, cc );
				set_pixel( src, xs+xx, ys+yy, cc, rgb[cc] * val );
			}
		}
	}
}
// �̹����� ������ ũ��� ������ ���ڸ� �׸���
void draw_box( image img, int xs, int ys, int xe, int ye, float r, float g, float b )
{
	//normalize_image(a);
	int ii;
	if ( xs < 0 )		xs = 0;
	if ( xs >= img.w )	xs = img.w-1;
	if ( xe < 0 )		xe = 0;
	if ( xe >= img.w )	xe = img.w-1;

	if ( ys < 0 )		ys = 0;
	if ( ys >= img.h )	ys = img.h-1;
	if ( ye < 0 )		ye = 0;
	if ( ye >= img.h )	ye = img.h-1;
	// ���μ� �׸���
	for ( ii=xs; ii <= xe; ++ii )
	{
		img.data[ii + ys*img.w + 0*img.w*img.h] = r;
		img.data[ii + ye*img.w + 0*img.w*img.h] = r;

		img.data[ii + ys*img.w + 1*img.w*img.h] = g;
		img.data[ii + ye*img.w + 1*img.w*img.h] = g;

		img.data[ii + ys*img.w + 2*img.w*img.h] = b;
		img.data[ii + ye*img.w + 2*img.w*img.h] = b;
	}
	// ���μ� �׸���
	for ( ii=ys; ii <= ye; ++ii )
	{
		img.data[xs + ii*img.w + 0*img.w*img.h] = r;
		img.data[xe + ii*img.w + 0*img.w*img.h] = r;

		img.data[xs + ii*img.w + 1*img.w*img.h] = g;
		img.data[xe + ii*img.w + 1*img.w*img.h] = g;

		img.data[xs + ii*img.w + 2*img.w*img.h] = b;
		img.data[xe + ii*img.w + 2*img.w*img.h] = b;
	}
}
// �̹����� ������ ũ��, �β�, ������ ���ڸ� �׸���
void draw_box_width( image img, int xs, int ys, int xe, int ye, int w, float r, float g, float b )
{
	int ii;
	for ( ii=0; ii < w; ++ii )
	{
		draw_box( img, xs+ii, ys+ii, xe-ii, ye-ii, r, g, b );
	}
}
// �̹����� ������ ������ũ��, �β�, ������ ���ڸ� �׸���
void draw_bbox( image img, box bbox, int w, float r, float g, float b )
{
	int left  = ( bbox.x - bbox.w/2 ) * img.w;
	int right = ( bbox.x + bbox.w/2 ) * img.w;
	int top   = ( bbox.y - bbox.h/2 ) * img.h;
	int bot   = ( bbox.y + bbox.h/2 ) * img.h;

	int ii;
	for ( ii=0; ii < w; ++ii )
	{
		draw_box( img, left+ii, top+ii, right-ii, bot-ii, r, g, b );
	}
}
// �׸����Ϸ� ������� �����̹����� ž���Ѵ�
image **load_alphabet()
{
	int i, j;
	const int nsize = 8;
	image **alphabets = calloc( nsize, sizeof( image ) );
	for ( j = 0; j < nsize; ++j )
	{
		alphabets[j] = calloc( 128, sizeof( image ) );
		for ( i = 32; i < 127; ++i )
		{
			char buff[256];
			//sprintf( buff, "data/labels/%d_%d.png", i, j );
			//sprintf_s( buff, 256, "data/labels/%d_%d.png", i, j );	//  [7/12/2018 jobs]
			sprintf_s( buff, 256, "data/labels_image/%d_%d.png", i, j );	//  [7/12/2018 jobs]
			alphabets[j][i] = load_image_color( buff, 0, 0 );
		}
	}
	return alphabets;
}

void draw_detections( image im
					, detection *dets	// ������ ���
					, int num			// ������ ���ڰ���
					, float thresh		// ���ΰ�
					, char **names		// ���� �з��̸� ���
					, image **alphabet	// ���ڱ׸�
					, int classes )		// �з�����
{
	int ii, jj;
	// ������ ���ڰ��� �ݺ�
	for ( ii=0; ii < num; ++ii )
	{
		char labelstr[4096] = { 0 };
		int class = -1;
		// �з����� �ݺ�
		for ( jj=0; jj < classes; ++jj )
		{
			if ( dets[ii].prob[jj] > thresh )
			{
				if ( class < 0 )
				{
					//strcat( labelstr, names[j] );
					strcat_s( labelstr, 4096, names[jj] );
					class = jj;
				}
				else		// �������� ���ΰ����� ū���
				{
					//strcat( labelstr, ", " );
					//strcat( labelstr, names[j] );
					strcat_s( labelstr, 4096, ", " );
					strcat_s( labelstr, 4096, names[jj] );
				}

				//printf( "%s: %.0f%%\n", names[jj], dets[ii].prob[jj]*100 );	//  [7/14/2018 jobs]
				printf( "%s: ����Ȯ�� %.0f%%\n", names[jj], dets[ii].prob[jj]*100 );	//  [7/14/2018 jobs]
			}
		}

		if ( class >= 0 )
		{
			int width = im.h * 0.006f;

/*			if( 0 )
			{
				width = pow(prob, 1./2.)*10+1;
				alphabet = 0;
			}
*/
			//printf( "%d %s: %.0f%%\n", i, names[class], prob*100 );
			int offset	= class*123457 % classes;	// �з������� ���� ��������
			float red	= get_color( 2, offset, classes );
			float green	= get_color( 1, offset, classes );
			float blue	= get_color( 0, offset, classes );
			float rgb[3];

			//width = prob*20+2;

			rgb[0] = red;
			rgb[1] = green;
			rgb[2] = blue;
			box bx = dets[ii].bbox;
			//printf("%f %f %f %f\n", bx.x, bx.y, bx.w, bx.h);

			int left  = ( bx.x - bx.w/2.0f )*im.w;	// ���� ����
			int right = ( bx.x + bx.w/2.0f )*im.w;	// ���� ��
			int top   = ( bx.y - bx.h/2.0f )*im.h;	// ���� ����
			int bot   = ( bx.y + bx.h/2.0f )*im.h;	// ���� ��

			if ( left < 0 )			left	= 0;
			if ( right > im.w-1 )	right	= im.w-1;
			if ( top < 0 )			top		= 0;
			if ( bot > im.h-1 )		bot		= im.h-1;

			draw_box_width( im, left, top, right, bot, width, red, green, blue );

			// ���ڿ��� �ش��ϴ� ���ڱ׸��� ã�Ƽ� �߰���
			if ( alphabet )
			{
				image label = get_label( alphabet, labelstr, (int)(im.h*0.03f) );
				//draw_label( im, top + width, left, label, rgb );	//  [7/14/2018 jobs]
				draw_label( im, left, top + width, label, rgb );	//  [7/14/2018 jobs]
				free_image( label );
			}

			if ( dets[ii].mask )
			{
				image mask			= float_to_image( 14, 14, 1, dets[ii].mask );
				image resized_mask	= resize_image( mask, bx.w*im.w, bx.h*im.h );
				image tmask			= threshold_image( resized_mask, 0.5f );
				embed_image( tmask, im, left, top );
				free_image( mask );
				free_image( resized_mask );
				free_image( tmask );
			}
		}
	}
}

// �̹����� �밢������(ȸ����ȯ) �Ѵ�
void transpose_image( image im )
{
	assert( im.w == im.h );
	int nn, mm;
	int cc;
	// �� �ݺ�(ä��)
	for ( cc=0; cc < im.c; ++cc )
	{
		int cc_Bo	= im.w*im.h*cc;	// �� ����

		// �� �ݺ�
		for ( nn=0; nn< im.w-1; ++nn )
		{
			// �� �ݺ�
			for ( mm=nn + 1; mm < im.w; ++mm )
			{
				int hw	= cc_Bo + im.w*nn + mm;	//������ġ
				int dw	= cc_Bo + im.w*mm + nn;	//������ġ
				//int hw	= mm + im.w*(nn + im.h*cc);	//������ġ
				//int dw	= nn + im.w*(mm + im.h*cc);	//������ġ

				float swap = im.data[ hw ];		//������ġ ���� ��Ƶд�
				im.data[ hw ] = im.data[ dw ];	//������ġ ���� ������ġ ������ �����Ѵ�
				im.data[ dw ] = swap;			//������ġ ���� ������ġ ������ �����Ѵ�

				//float swap = im.data[ mm + im.w*(nn + im.h*cc) ];
				//im.data[ mm + im.w*(nn + im.h*cc) ] = im.data[ nn + im.w*(mm + im.h*cc) ];
				//im.data[ nn + im.w*(mm + im.h*cc) ] = swap;
			}
		}
	}
}

// �̹����� �ð�������� ȸ���Ѵ�
void rotate_image_cw( image im, int times )
{
	assert( im.w == im.h );
	times = (times + 400) % 4;	//ȸ��Ƚ��
	int ii, yy, xx, cc;
	int nn = im.w;
	// ȸ�� �ݺ�
	for ( ii=0; ii < times; ++ii )
	{
		// �� �ݺ�(ä��)
		for ( cc=0; cc < im.c; ++cc )
		{
			int cc_Bo	= im.w*im.h*cc;	// �� ����

			// ���� �ݺ�(�������� �ݸ� �ݺ�)
			for ( yy=0; yy < nn/2; ++yy )
			{
				// ���� �ݺ�(�������� �ݸ� �ݺ�)
				for ( xx=0; xx < (nn-1)/2 + 1; ++xx )
				{
					// ���� ����� �̹����迭�� ���� ������ �Ǿ �ݽð�������� ������ �ð�������� ȸ����
					int cw000	= cc_Bo + im.w*yy + xx;						//������ġ   0�� ��ġ
					int cw090	= cc_Bo + im.w*xx + ( nn-1-yy );			//�ð����  90�� ��ġ
					int cw180	= cc_Bo + im.w*( nn-1-yy ) + ( nn-1-xx );	//�ð���� 180�� ��ġ
					int cw270	= cc_Bo + im.w*( nn-1-xx ) + yy;			//�ð���� 270�� ��ġ
					//int cw000	= xx + im.w*(yy + im.h*cc);				//������ġ   0�� ��ġ
					//int cw090	= nn-1-yy + im.w*(xx + im.h*cc);		//�ð����  90�� ��ġ
					//int cw180	= nn-1-xx + im.w*(nn-1-yy + im.h*cc);	//�ð���� 180�� ��ġ
					//int cw270	= yy + im.w*(nn-1-xx + im.h*cc);		//�ð���� 270�� ��ġ

					float temp		 = im.data[ cw000 ];	//  0�� ȭ�Ұ��� ��Ƶд�
					im.data[ cw000 ] = im.data[ cw090 ];	//  0�� ȭ�Ұ���  90�� ȭ�Ұ����� ����
					im.data[ cw090 ] = im.data[ cw180 ];	// 90�� ȭ�Ұ��� 180�� ȭ�Ұ����� ����
					im.data[ cw180 ] = im.data[ cw270 ];	//180�� ȭ�Ұ��� 270�� ȭ�Ұ����� ����
					im.data[ cw270 ] = temp;				//270�� ȭ�Ұ���   0�� ȭ�Ұ����� ����

					//float temp = im.data[ xx + im.w*(yy + im.h*cc) ];
					//im.data[ xx + im.w*(yy + im.h*cc) ] = im.data[ nn-1-yy + im.w*(xx + im.h*cc) ];
					//im.data[ nn-1-yy + im.w*(xx + im.h*cc) ] = im.data[ nn-1-xx + im.w*(nn-1-yy + im.h*cc) ];
					//im.data[ nn-1-xx + im.w*(nn-1-yy + im.h*cc) ] = im.data[ yy + im.w*(nn-1-xx + im.h*cc) ];
					//im.data[ yy + im.w*(nn-1-xx + im.h*cc) ] = temp;
				}
			}
		}
	}
}

// �̹����� �¿�� �����´�
void flip_image( image im )
{
	int yy, xx, cc;

	// �� �ݺ�(ä��)
	for ( cc=0; cc < im.c; ++cc )
	{
		int cc_Bo	= im.w*im.h*cc;	// �� ����

		// ���� �ݺ�
		for ( yy=0; yy < im.h; ++yy )
		{
			// ���� �ݺ�(�������� �ݸ� �ݺ�)
			for ( xx=0; xx < im.w/2; ++xx )
			{
				int hw	= cc_Bo + im.w*yy + xx;					//����ȭ��(������) ��ġ
				int dw	= cc_Bo + im.w*yy + (im.w - xx - 1);	//����ȭ��(�ݴ���) ��ġ
				//int hw	= xx + im.w*(yy + im.h*(cc));				//����ȭ��(������) ��ġ
				//int dw	= (im.w - xx - 1) + im.w*(yy + im.h*(cc));	//����ȭ��(�ݴ���) ��ġ

				float swap	= im.data[ dw ];	//�ݴ��� ȭ�Ұ� ��Ƶα�
				im.data[ dw ] = im.data[ hw ];	//������ ȭ�Ұ����� �ݴ��� ȸ�Ұ��� ����
				im.data[ hw ] = swap;			//��Ƶ� ȭ�Ұ����� ������ ȭ�Ұ��� ����
			}
		}
	}
}

image image_distance( image im_a, image im_b )
{
	int i, j;
	image dist = make_image( im_a.w, im_a.h, 1 );

	for ( i = 0; i < im_a.c; ++i )
	{
		for ( j = 0; j < im_a.h*im_a.w; ++j )
		{
			dist.data[j] += pow( im_a.data[i*im_a.h*im_a.w+j]-im_b.data[i*im_a.h*im_a.w+j], 2 );
		}
	}

	for ( j = 0; j < im_a.h*im_a.w; ++j )
	{
		dist.data[j] = sqrt( dist.data[j] );
	}

	return dist;
}

void ghost_image( image src, image dst, int xs, int ys )
{
	int xx, yy, cc;
	float max_dist = (float)sqrt( (-src.w/2.0 + 0.5)*(-src.w/2.0 + 0.5) );

	for ( cc=0; cc < src.c; ++cc )
	{
		for ( yy=0; yy < src.h; ++yy )
		{
			for ( xx=0; xx < src.w; ++xx )
			{
				float dist = sqrt( (xx - src.w/2.0 + 0.5)*(xx - src.w/2.0 + 0.5) +
								   (yy - src.h/2.0 + 0.5)*(yy - src.h/2.0 + 0.5) );

				float alpha = (1 - dist/max_dist);

				if ( alpha < 0 ) alpha = 0;

				float v1 = get_pixel( src, xx, yy, cc );
				float v2 = get_pixel( dst, xs+xx, ys+yy, cc );
				float val = alpha*v1 + (1-alpha)*v2;

				set_pixel( dst, xs+xx, ys+yy, cc, val );
			}
		}
	}
}

// ��� �̹���
void blocky_image( image im, int s )
{
	int xx, yy, cc;

	for ( cc=0; cc < im.c; ++cc )
	{
		int cc_Bo	= im.w*im.h*cc;	// �� ����

		for ( yy=0; yy < im.h; ++yy )
		{
			for ( xx= 0; xx < im.w; ++xx )
			{
				im.data[ cc_Bo + im.w*yy + xx ] = im.data[ cc_Bo + im.w*( yy / (s*s) ) + xx/(s*s) ];
				//im.data[xx + im.w*(yy + im.h*cc)] = im.data[xx/(s*s) + im.w*(yy/(s*s) + im.h*cc)];
			}
		}
	}
}

// �˿� �̹���
void censor_image( image im, int dx, int dy, int w, int h )
{
	int xx, yy, cc;
	int s = 32;
	if ( dx < 0 ) dx = 0;
	if ( dy < 0 ) dy = 0;

	for ( cc= 0; cc < im.c; ++cc )
	{
		int cc_Bo	= im.w*im.h*cc;	// �� ����

		for ( yy=dy; yy < dy + h && yy < im.h; ++yy )
		{
			for ( xx=dx; xx < dx + w && xx < im.w; ++xx )
			{
				im.data[ cc_Bo + im.w*yy + xx ] = im.data[ cc_Bo + im.w*( yy / (s*s) ) + xx/(s*s) ];
				//im.data[xx + im.w*(yy + im.h*cc)] = im.data[xx/s*s + im.w*(yy/s*s + im.h*cc)];
				//im.data[xx + j*im.w + cc*im.w*im.h] = 0;
			}
		}
	}
}

// �����̹����� �纻�̹����� �ҹ����Ѵ�
void embed_image( image source, image dest, int xs, int ys )
{
	int xx, yy, cc;

	for ( cc=0; cc < source.c; ++cc )
	{
		for ( yy=0; yy < source.h; ++yy )
		{
			for ( xx=0; xx < source.w; ++xx )
			{
				float val = get_pixel( source, xx, yy, cc );
				set_pixel( dest, xs+xx, ys+yy, cc, val );
			}
		}
	}
}
// �����̹����� �纻�̹����� �ҹ����Ѵ�
void SoBaGi_image( image WonBon, image SaBon, int Bo_GaRo, int Bo_SeRo, int nWiChi )
{
	int nPanSu		= SaBon.w / Bo_GaRo;	// �Ѻ��� �� ������ ����
	int xx, yy, cc;

	//int iMok		= (nWiChi * WonBon.c) / nPanSu;		// ��
	//int iNaMeoJi	= (nWiChi * WonBon.c) % nPanSu;		// ������

	//int SiJakGaRo	= iNaMeoJi * Bo_GaRo;
	//int SiJakSeRo	= iMok * Bo_SeRo;

	for ( cc=0; cc < WonBon.c; ++cc )			// �Ǽ��� �ݺ�
	{
		int iMok		= (nWiChi * WonBon.c + cc) / nPanSu;		// ��
		int iNaMeoJi	= (nWiChi * WonBon.c + cc) % nPanSu;		// ������

		int SiJakGaRo	= iNaMeoJi * Bo_GaRo;
		int SiJakSeRo	= iMok * Bo_SeRo;

		for ( yy=0; yy < WonBon.h; ++yy )		// ���θ� �ݺ�
		{
			for ( xx=0; xx < WonBon.w; ++xx )	// ���θ� �ݺ�
			{
				float val = get_pixel( WonBon, xx, yy, cc );

				set_pixel( SaBon, SiJakGaRo+xx, SiJakSeRo+yy, 0, val );
			}
		}
	}
}

//
image collapse_image_layers( image source, int border )
{
	int h = source.h;
	h = (h+border)*source.c - border;
	image dest = make_image( source.w, h, 1 );

	int i;
	for ( i = 0; i < source.c; ++i )
	{
		image layer = get_image_layer( source, i );
		int h_offset = i*(source.h+border);
		embed_image( layer, dest, 0, h_offset );
		free_image( layer );
	}
	return dest;
}
// �ڷᰪ�� 0~1 ���̰����� �����Ѵ�
void constrain_image( image im )
{
	int ii;
	for ( ii=0; ii < im.w*im.h*im.c; ++ii )
	{
		if ( im.data[ii] < 0 ) im.data[ii] = 0;
		if ( im.data[ii] > 1 ) im.data[ii] = 1;
	}
}
// �̹����ϳ����� �ִ밪�� �ּҰ��� �˾Ƴ��� ���⸦ �Ѵ�
void normalize_image( image p )
{
	int ii;
	float min = 9999999.0f;
	float max = -999999.0f;
	// �ִ밪�� �ּҰ��� �˾Ƴ���
	for ( ii=0; ii < p.h*p.w*p.c; ++ii )
	{
		float v = p.data[ii];
		if ( v < min ) min = v;
		if ( v > max ) max = v;
	}
	// �ִ밪�� �ּҰ� ���̰� �ʹ� ������ 1, 0 ���� �����Ѵ�
	if ( max - min < 0.000000001f )
	{
		min = 0.0f;
		max = 1.0f;
	}
	// �ִ밪�� �ּҰ��� ����
	for ( ii=0; ii < p.c*p.w*p.h; ++ii )
	{
		p.data[ii] = ( p.data[ii]-min ) / ( max-min );
	}
}
// �̹����������� �ִ밪�� �ּҰ��� �˾Ƴ��� ���⸦ �Ѵ�
void normalize_image_MuRi( image *pImage, int iPanSu )
{
	int ii, jj;
	float min = 9999999.0f;
	float max = -999999.0f;

	image img = pImage[0];
	int BanBok	= img.h * img.w * img.c;

	// �ִ밪�� �ּҰ��� �˾Ƴ���
	for ( jj=0; jj<iPanSu; ++jj )
	{
		img	= pImage[jj];

		//int BanBok	= img.h * img.w * img.c;

		for ( ii=0; ii < BanBok; ++ii )
		{
			float v = img.data[ii];
			if ( v < min ) min = v;
			if ( v > max ) max = v;
		}
	}
	// �ִ밪�� �ּҰ� ���̰� �ʹ� ������ 1, 0 ���� �����Ѵ�
	if ( max - min < 0.000000001f )
	{
		min = 0.0f;
		max = 1.0f;
	}
	// �ִ밪�� �ּҰ��� ����
	for ( jj=0; jj<iPanSu; ++jj )
	{
		img	= pImage[jj];

		//int BanBok	= img.h * img.w * img.c;

		for ( ii=0; ii < BanBok; ++ii )
		{
			img.data[ii] = (img.data[ii]-min) / (max-min);
		}
	}
}

void normalize_image2( image p )
{
	float *min = calloc( p.c, sizeof( float ) );
	float *max = calloc( p.c, sizeof( float ) );

	int i, j;
	for ( i = 0; i < p.c; ++i ) min[i] = max[i] = p.data[i*p.h*p.w];

	for ( j = 0; j < p.c; ++j )
	{
		for ( i = 0; i < p.h*p.w; ++i )
		{
			float v = p.data[i+j*p.h*p.w];
			if ( v < min[j] ) min[j] = v;
			if ( v > max[j] ) max[j] = v;
		}
	}

	for ( i = 0; i < p.c; ++i )
	{
		if ( max[i] - min[i] < 0.000000001f )
		{
			min[i] = 0;
			max[i] = 1;
		}
	}

	for ( j = 0; j < p.c; ++j )
	{
		for ( i = 0; i < p.w*p.h; ++i )
		{
			p.data[i+j*p.h*p.w] = (p.data[i+j*p.h*p.w] - min[j])/(max[j]-min[j]);
		}
	}

	free( min );
	free( max );
}

void copy_image_into( image src, image dest )
{
	memcpy( dest.data, src.data, src.h*src.w*src.c*sizeof( float ) );
}
// ��Ƶ� �޸𸮸� �Ҵ��ϰ�, �ڷᰪ�� �����ϰ�, ��Ƶ� �ּҸ� ��ȯ�Ѵ�
image copy_image( image p )
{
	image copy = p;
	//copy.data = calloc( p.h*p.w*p.c, sizeof( float ) );
	//memcpy( copy.data, p.data, p.h*p.w*p.c*sizeof( float ) );

	int Sul	= p.h*p.w*p.c;	// ������ �ڷ�ũ��
	copy.data = calloc( Sul, sizeof( float ) );
	memcpy( copy.data, p.data, Sul * sizeof( float ) );

	return copy;
}
// RGB �̹����� BGR �̹����� ��ȯ???
void rgbgr_image( image im )
{
	int ii;
	for ( ii=0; ii < im.w*im.h; ++ii )
	{
		float swap	= im.data[ii];					// R ���� ��Ƶд�
		im.data[ii]	= im.data[ im.w*im.h*2 + ii ];	// B ���� R ���� ���
		im.data[ im.w*im.h*2 + ii ] = swap;			// ��Ƶ� R ���� B ���� ���
	}
}

#ifdef OPENCV
//
void show_image_cv( image p, const char *name, IplImage *disp )
{
	int xx, yy, cc;
	if ( p.c == 3 ) rgbgr_image( p );
	//normalize_image(copy);

	char buff[256];
	//sprintf( buff, "%s (%d)", name, windows );
	//sprintf( buff, "%s", name );
	sprintf_s( buff, 256, "%s", name );

	int step = disp->widthStep;
	cvNamedWindow( buff, CV_WINDOW_NORMAL );
	//cvMoveWindow(buff, 100*(windows%10) + 200*(windows/10), 100*(windows%10));
	++windows;
	// �Ǽ����� ���������� ��ȯ
	for ( yy=0; yy < p.h; ++yy )
	{
		for ( xx=0; xx < p.w; ++xx )
		{
			for ( cc=0; cc < p.c; ++cc )
			{
				disp->imageData[yy*step + xx*p.c + cc] = (unsigned char)(get_pixel( p, xx, yy, cc )*255);
			}
		}
	}

	if ( 0 )
	{
		int w = 448;
		int h = w*p.h/p.w;

		if ( h > 1000 )
		{
			h = 1000;
			w = h*p.w/p.h;
		}

		IplImage *buffer = disp;
		disp = cvCreateImage( cvSize( w, h ), buffer->depth, buffer->nChannels );
		cvResize( buffer, disp, CV_INTER_LINEAR );
		cvReleaseImage( &buffer );
	}

	cvShowImage( buff, disp );
}
#endif

void show_image( image p, const char *name )
{
	#ifdef OPENCV
	IplImage *disp	= cvCreateImage( cvSize( p.w, p.h ), IPL_DEPTH_8U, p.c );
	image copy		= copy_image( p );
	constrain_image( copy );			//�ڷᰪ�� 0~1 ���̰����� �����Ѵ�
	show_image_cv( copy, name, disp );
	free_image( copy );
	cvReleaseImage( &disp );

	#else
	fprintf( stderr
	//	, "Not compiled with OpenCV, saving to %s.png instead\n"	//  [7/12/2018 jobs]
		, "OpenCV�� ������ �ȵ�, ��ſ� %s.png �� ����\n"	//  [7/12/2018 jobs]
		, name );
	save_image( p, name );

	#endif
}

#ifdef OPENCV
// OpenCV�� IplImage ���� �̹��� ���� �Ǽ��� ��ȯ�Ͽ� image ������ ���� ����
void ipl_into_image( IplImage* src, image im )
{
	unsigned char *data = (unsigned char *)src->imageData;
	int hh	 = src->height;
	int ww	 = src->width;
	int cc	 = src->nChannels;
	int step = src->widthStep;
	int yy, xx, kk;

	for ( yy=0; yy < hh; ++yy )
	{
		for ( kk=0; kk < cc; ++kk )
		{
			for ( xx=0; xx < ww; ++xx )
			{
				//image[ ��		  + ��	  + �� ] =	Ipl[ ��		 + ��	 + �� ]
				im.data[ kk*ww*hh + yy*ww + xx ] = data[ yy*step + xx*cc + kk ]/255.0f;
			}
		}
	}
}
// IplImage ���� �̹����� image ���� �̹����� �����Ͽ� ��ȯ�Ѵ�
image ipl_to_image( IplImage* src )
{
	int h = src->height;
	int w = src->width;
	int c = src->nChannels;
	image out = make_image( w, h, c );
	ipl_into_image( src, out );
	return out;
}
// �̹����� OpenCV�� ž���ϰ� image �ڷᱸ���� ��ȯ�Ͽ� ��ȯ�Ѵ�
image load_image_cv( char *filename, int channels )
{
	IplImage* src = 0;
	int flag = -1;

	if		( channels == 0 ) flag = -1;
	else if ( channels == 1 ) flag = 0;
	else if ( channels == 3 ) flag = 1;
	else
	{
		//fprintf( stderr, "OpenCV can't force load with %d channels\n", channels );	//  [7/6/2018 jobs]
		fprintf( stderr, "OpenCV�� %d ä�η� ����ž���� �� ����!\n", channels );	//  [7/6/2018 jobs]
	}

	if ( (src = cvLoadImage( filename, flag )) == 0 )
	{
		//fprintf( stderr, "Cannot load image \"%s\"\n", filename );	//  [7/6/2018 jobs]
		fprintf( stderr, "\"%s\" �̹����� ž������ ����!\n", filename );	//  [7/6/2018 jobs]

		char buff[256];
		//sprintf( buff, "echo %s >> bad.list", filename );
		//sprintf_s( buff, 256, "echo %s >> bad.list", filename );	//  [7/6/2018 jobs]
		sprintf_s( buff, 256, "echo %s >> �߸���.���", filename );	//  [7/6/2018 jobs]

		system( buff );
		return make_image( 10, 10, 3 );
		//exit(0);
	}

	image out = ipl_to_image( src );	// �ڷḦ ���� �����ϱ� ���� ��ȯ
	cvReleaseImage( &src );				// OpenCV �̹����� �Ҵ��� �޸� ����
	rgbgr_image( out );					// RGB �̹����� BGR �̹����� ��ȯ

	return out;
}

void flush_stream_buffer( CvCapture *cap, int n )
{
	int i;
	for ( i = 0; i < n; ++i )
	{
		cvQueryFrame( cap );
	}
}
// �����󿡼� ������� IplImage ������ ��ȹ�Ͽ� image ������ ��ȯ�Ͽ� ��ȯ�Ѵ�
image get_image_from_stream( CvCapture *cap )
{
	IplImage* src = cvQueryFrame( cap );

	if ( !src ) return make_empty_image( 0, 0, 0 );

	image im = ipl_to_image( src );
	rgbgr_image( im );

	return im;
}

int fill_image_from_stream( CvCapture *cap, image im )
{
	IplImage* src = cvQueryFrame( cap );
	if ( !src ) return 0;
	ipl_into_image( src, im );
	rgbgr_image( im );
	return 1;
}

void save_image_jpg( image p, const char *name )
{
	image copy = copy_image( p );
	if ( p.c == 3 ) rgbgr_image( copy );
	int xx, yy, cc;

	char buff[256];
	//sprintf( buff, "%s.jpg", name );
	sprintf_s( buff, 256, "%s.jpg", name );

	IplImage *disp = cvCreateImage( cvSize( p.w, p.h ), IPL_DEPTH_8U, p.c );
	int step = disp->widthStep;

	for ( yy=0; yy < p.h; ++yy )
	{
		for ( xx=0; xx < p.w; ++xx )
		{
			for ( cc=0; cc < p.c; ++cc )
			{
				disp->imageData[yy*step + xx*p.c + cc] = (unsigned char)(get_pixel( copy, xx, yy, cc )*255);
			}
		}
	}

	cvSaveImage( buff, disp, 0 );
	cvReleaseImage( &disp );
	free_image( copy );
}
#endif

void save_image_png( image im, const char *name )
{
	char buff[256];
	//sprintf( buff, "%s (%d)", name, windows );
	//sprintf( buff, "%s.png", name );
	sprintf_s( buff, 256, "%s.png", name );
	unsigned char *data = calloc( im.w*im.h*im.c, sizeof( char ) );

	int i, k;
	for ( k = 0; k < im.c; ++k )
	{
		for ( i = 0; i < im.w*im.h; ++i )
		{
			data[i*im.c+k] = (unsigned char)(255*im.data[i + k*im.w*im.h]);
		}
	}

	int success = stbi_write_png( buff, im.w, im.h, im.c, data, im.w*im.c );
	free( data );

	if ( !success )
	//	fprintf( stderr, "Failed to write image %s\n", buff );	//  [7/6/2018 jobs]
		fprintf( stderr, "%s �̹��� ��Ͻ���!\n", buff );	//  [7/6/2018 jobs]
}

void save_image( image im, const char *name )
{
	#ifdef OPENCV
	save_image_jpg( im, name );
	#else
	save_image_png( im, name );
	#endif
}


void show_image_layers( image p, char *name )
{
	int i;
	char buff[256];
	for ( i = 0; i < p.c; ++i )
	{
		//sprintf( buff, "%s - Layer %d", name, i );
		sprintf_s( buff, 256, "%s - Layer %d", name, i );
		image layer = get_image_layer( p, i );
		show_image( layer, buff );
		free_image( layer );
	}
}

void show_image_collapsed( image p, char *name )
{
	image c = collapse_image_layers( p, 1 );
	show_image( c, name );
	free_image( c );
}

// �� �̹���(�޸𸮹迭 �Ҵ����)�� �����
image make_empty_image( int w, int h, int c )
{
	image out;
	out.data = 0;
	out.h = h;
	out.w = w;
	out.c = c;
	return out;
}

// �̹����迭�� �޸� �Ҵ�
image make_image( int w, int h, int c )
{
	image out = make_empty_image( w, h, c );
	out.data = calloc( h*w*c, sizeof( float ) );
	return out;
}

// �̹����迭�� �޸� �Ҵ��ϰ� ���� �Ѹ���
image make_random_image( int w, int h, int c )
{
	image out = make_empty_image( w, h, c );
	out.data = calloc( h*w*c, sizeof( float ) );
	int i;
	for ( i = 0; i < w*h*c; ++i )
	{
		out.data[i] = (rand_normal() * 0.25f) + 0.5f;
	}
	return out;
}

// �ڷ�迭�� �ּҸ� ������(�̹����� ä��, ����, ���� ������ �߰���)
image float_to_image( int w, int h, int c, float *data )
{
	image out = make_empty_image( w, h, c );
	out.data = data;
	return out;
}

void place_image( image im, int ww, int hh, int dx, int dy, image canvas )
{
	int xx, yy, cc;

	for ( cc=0; cc < im.c; ++cc )
	{
		for ( yy=0; yy < hh; ++yy )
		{
			for ( xx=0; xx < ww; ++xx )
			{
				float rx = ((float)xx / ww) * im.w;
				float ry = ((float)yy / hh) * im.h;
				float val = bilinear_interpolate( im, rx, ry, cc );
				set_pixel( canvas, xx + dx, yy + dy, cc, val );
			}
		}
	}
}

image center_crop_image( image im, int w, int h )
{
	int m = (im.w < im.h) ? im.w : im.h;
	image c = crop_image( im, (im.w - m) / 2, (im.h - m)/2, m, m );
	image r = resize_image( c, w, h );
	free_image( c );
	return r;
}

image rotate_crop_image( image im, float rad, float s, int w, int h, float dx, float dy, float aspect )
{
	int x, y, c;
	float cx = (float)im.w / 2.0f;
	float cy = (float)im.h / 2.0f;
	image rot = make_image( w, h, im.c );

	for ( c = 0; c < im.c; ++c )
	{
		for ( y = 0; y < h; ++y )
		{
			for ( x = 0; x < w; ++x )
			{
				float rx = (float)( cos( rad )*( (x - w/2.0)/s*aspect + dx/s*aspect ) -
									sin( rad )*( (y - h/2.0)/s + dy/s ) + cx );
				float ry = (float)( sin( rad )*( (x - w/2.0)/s*aspect + dx/s*aspect ) +
									cos( rad )*( (y - h/2.0)/s + dy/s ) + cy );
				float val = bilinear_interpolate( im, rx, ry, c );
				set_pixel( rot, x, y, c, val );
			}
		}
	}
	return rot;
}

image rotate_image( image im, float rad )
{
	int x, y, c;
	float cx = im.w/2.;
	float cy = im.h/2.;
	image rot = make_image( im.w, im.h, im.c );

	for ( c = 0; c < im.c; ++c )
	{
		for ( y = 0; y < im.h; ++y )
		{
			for ( x = 0; x < im.w; ++x )
			{
				float rx = cos( rad )*(x-cx) - sin( rad )*(y-cy) + cx;
				float ry = sin( rad )*(x-cx) + cos( rad )*(y-cy) + cy;
				float val = bilinear_interpolate( im, rx, ry, c );
				set_pixel( rot, x, y, c, val );
			}
		}
	}
	return rot;
}
// �̹��� �ڷᰪ ������ ������ ä���
void fill_image( image m, float s )
{
	int i;
	for ( i = 0; i < m.h*m.w*m.c; ++i ) m.data[i] = s;
}

void translate_image( image m, float s )
{
	int i;
	for ( i = 0; i < m.h*m.w*m.c; ++i ) m.data[i] += s;
}

void scale_image( image m, float s )
{
	int i;
	for ( i = 0; i < m.h*m.w*m.c; ++i ) m.data[i] *= s;
}

image crop_image( image im, int dx, int dy, int w, int h )
{
	image cropped = make_image( w, h, im.c );
	int i, j, k;
	for ( k = 0; k < im.c; ++k )
	{
		for ( j = 0; j < h; ++j )
		{
			for ( i = 0; i < w; ++i )
			{
				int r = j + dy;
				int c = i + dx;
				float val = 0;
				r = constrain_int( r, 0, im.h-1 );
				c = constrain_int( c, 0, im.w-1 );
				val = get_pixel( im, c, r, k );
				set_pixel( cropped, i, j, k, val );
			}
		}
	}
	return cropped;
}

int best_3d_shift_r( image a, image b, int min, int max )
{
	if ( min == max ) return min;
	int mid = floor( (min + max) / 2. );
	image c1 = crop_image( b, 0, mid, b.w, b.h );
	image c2 = crop_image( b, 0, mid+1, b.w, b.h );
	float d1 = dist_array( c1.data, a.data, a.w*a.h*a.c, 10 );
	float d2 = dist_array( c2.data, a.data, a.w*a.h*a.c, 10 );
	free_image( c1 );
	free_image( c2 );
	if ( d1 < d2 ) return best_3d_shift_r( a, b, min, mid );
	else return best_3d_shift_r( a, b, mid+1, max );
}

int best_3d_shift( image a, image b, int min, int max )
{
	int i;
	int best = 0;
	float best_distance = FLT_MAX;
	for ( i = min; i <= max; i += 2 )
	{
		image c = crop_image( b, 0, i, b.w, b.h );
		float d = dist_array( c.data, a.data, a.w*a.h*a.c, 100 );
		if ( d < best_distance )
		{
			best_distance = d;
			best = i;
		}
		printf( "%d %f\n", i, d );
		free_image( c );
	}
	return best;
}

void composite_3d( char *f1, char *f2, char *out, int delta )
{
	if ( !out ) out = "out";
	image a = load_image( f1, 0, 0, 0 );
	image b = load_image( f2, 0, 0, 0 );
	int shift = best_3d_shift_r( a, b, -a.h/100, a.h/100 );

	image c1 = crop_image( b, 10, shift, b.w, b.h );
	float d1 = dist_array( c1.data, a.data, a.w*a.h*a.c, 100 );
	image c2 = crop_image( b, -10, shift, b.w, b.h );
	float d2 = dist_array( c2.data, a.data, a.w*a.h*a.c, 100 );

	if ( d2 < d1 && 0 )
	{
		image swap = a;
		a = b;
		b = swap;
		shift = -shift;
		printf( "swapped, %d\n", shift );
	}
	else
	{
		printf( "%d\n", shift );
	}

	image c = crop_image( b, delta, shift, a.w, a.h );
	int i;
	for ( i = 0; i < c.w*c.h; ++i )
	{
		c.data[i] = a.data[i];
	}

	#ifdef OPENCV
	save_image_jpg( c, out );
	#else
	save_image( c, out );
	#endif
}

void letterbox_image_into( image im, int ww, int hh, image boxed )
{
	int new_w = im.w;
	int new_h = im.h;

	if ( ((float)ww/im.w) < ((float)hh/im.h) )
	{
		new_w = ww;
		new_h = (im.h * ww)/im.w;
	}
	else
	{
		new_h = hh;
		new_w = (im.w * hh)/im.h;
	}

	image resized = resize_image( im, new_w, new_h );
	embed_image( resized, boxed, (ww-new_w)/2, (hh-new_h)/2 );
	free_image( resized );
}

image letterbox_image( image im, int ww, int hh )
{
	int new_w = im.w;	// �Է��ڷ� �ʺ�
	int new_h = im.h;	// �Է��ڷ� ����

	if ( ((float)ww/im.w) < ((float)hh/im.h) )
	{
		new_w = ww;
		new_h = (im.h * ww)/im.w;
	}
	else
	{
		new_h = hh;
		new_w = (im.w * hh)/im.h;
	}

	image resized	= resize_image( im, new_w, new_h );
	image boxed		= make_image( ww, hh, im.c );
	fill_image( boxed, 0.5f );
	//int i;
	//for(i = 0; i < boxed.w*boxed.h*boxed.c; ++i) boxed.data[i] = 0;
	embed_image( resized, boxed, (ww-new_w)/2, (hh-new_h)/2 );
	free_image( resized );

	return boxed;
}

image resize_max( image im, int max )
{
	int w = im.w;
	int h = im.h;
	if ( w > h )
	{
		h = (h * max) / w;
		w = max;
	}
	else
	{
		w = (w * max) / h;
		h = max;
	}
	if ( w == im.w && h == im.h ) return im;
	image resized = resize_image( im, w, h );
	return resized;
}

image resize_min( image im, int min )
{
	int w = im.w;
	int h = im.h;
	if ( w < h )
	{
		h = (h * min) / w;
		w = min;
	}
	else
	{
		w = (w * min) / h;
		h = min;
	}
	if ( w == im.w && h == im.h ) return im;
	image resized = resize_image( im, w, h );
	return resized;
}

image random_crop_image( image im, int w, int h )
{
	int dx = rand_int( 0, im.w - w );
	int dy = rand_int( 0, im.h - h );
	image crop = crop_image( im, dx, dy, w, h );
	return crop;
}

augment_args random_augment_args( image im
								, float angle
								, float aspect
								, int low
								, int high
								, int w
								, int h )
{
	augment_args a = { 0 };
	aspect = rand_scale( aspect );
	int r = rand_int( low, high );
	int min = (im.h < im.w*aspect) ? im.h : im.w*aspect;
	float scale = (float)r / min;

	float rad = rand_uniform( -angle, angle ) * TWO_PI / 360.;

	float dx = (im.w*scale/aspect - w) / 2.;
	float dy = (im.h*scale - w) / 2.;
	//if(dx < 0) dx = 0;
	//if(dy < 0) dy = 0;
	dx = rand_uniform( -dx, dx );
	dy = rand_uniform( -dy, dy );

	a.rad = rad;
	a.scale = scale;
	a.w = w;
	a.h = h;
	a.dx = dx;
	a.dy = dy;
	a.aspect = aspect;
	return a;
}

image random_augment_image( image im
						, float angle
						, float aspect
						, int low
						, int high
						, int w
						, int h )
{
	augment_args a = random_augment_args( im
								, angle
								, aspect
								, low
								, high
								, w
								, h );
	image crop = rotate_crop_image( im
								, a.rad
								, a.scale
								, a.w
								, a.h
								, a.dx
								, a.dy
								, a.aspect );
	return crop;
}

float three_way_max( float a, float b, float c )
{
	return (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c);
}

float three_way_min( float a, float b, float c )
{
	return (a < b) ? ((a < c) ? a : c) : ((b < c) ? b : c);
}

void yuv_to_rgb( image im )
{
	assert( im.c == 3 );
	int i, j;
	float r, g, b;
	float y, u, v;
	for ( j = 0; j < im.h; ++j )
	{
		for ( i = 0; i < im.w; ++i )
		{
			y = get_pixel( im, i, j, 0 );
			u = get_pixel( im, i, j, 1 );
			v = get_pixel( im, i, j, 2 );

			r = y +  1.13983f*v;
			g = y + -0.39465f*u + -0.58060f*v;
			b = y +  2.03211f*u;

			set_pixel( im, i, j, 0, r );
			set_pixel( im, i, j, 1, g );
			set_pixel( im, i, j, 2, b );
		}
	}
}

void rgb_to_yuv( image im )
{
	assert( im.c == 3 );
	int i, j;
	float r, g, b;
	float y, u, v;
	for ( j = 0; j < im.h; ++j )
	{
		for ( i = 0; i < im.w; ++i )
		{
			r = get_pixel( im, i, j, 0 );
			g = get_pixel( im, i, j, 1 );
			b = get_pixel( im, i, j, 2 );

			y =  0.299f*r   +  0.587f*g   +  0.114f*b;
			u = -0.14713f*r + -0.28886f*g +  0.436f*b;
			v =  0.615f*r   + -0.51499f*g + -0.10001f*b;

			set_pixel( im, i, j, 0, y );
			set_pixel( im, i, j, 1, u );
			set_pixel( im, i, j, 2, v );
		}
	}
}

// http://www.cs.rit.edu/~ncs/color/t_convert.html
void rgb_to_hsv( image im )
{
	assert( im.c == 3 );
	int i, j;
	float r, g, b;
	float h, s, v;
	for ( j = 0; j < im.h; ++j )
	{
		for ( i = 0; i < im.w; ++i )
		{
			r = get_pixel( im, i, j, 0 );
			g = get_pixel( im, i, j, 1 );
			b = get_pixel( im, i, j, 2 );
			float max = three_way_max( r, g, b );
			float min = three_way_min( r, g, b );
			float delta = max - min;
			v = max;
			if ( max == 0 )
			{
				s = 0;
				h = 0;
			}
			else
			{
				s = delta/max;

				if		( r == max )	{	h = (g - b) / delta;		}
				else if ( g == max )	{	h = 2 + (b - r) / delta;	}
				else					{	h = 4 + (r - g) / delta;	}

				if ( h < 0 ) h += 6;

				h = h/6.;
			}
			set_pixel( im, i, j, 0, h );
			set_pixel( im, i, j, 1, s );
			set_pixel( im, i, j, 2, v );
		}
	}
}

void hsv_to_rgb( image im )
{
	assert( im.c == 3 );
	int i, j;
	float r, g, b;
	float h, s, v;
	float f, p, q, t;

	for ( j = 0; j < im.h; ++j )
	{
		for ( i = 0; i < im.w; ++i )
		{
			h = 6 * get_pixel( im, i, j, 0 );
			s = get_pixel( im, i, j, 1 );
			v = get_pixel( im, i, j, 2 );

			if ( s == 0 )
			{
				r = g = b = v;
			}
			else
			{
				int index = floor( h );
				f = h - index;
				p = v*(1-s);
				q = v*(1-s*f);
				t = v*(1-s*(1-f));

				if		( index == 0 )	{	r = v; g = t; b = p;	}
				else if ( index == 1 )	{	r = q; g = v; b = p;	}
				else if ( index == 2 )	{	r = p; g = v; b = t;	}
				else if ( index == 3 )	{	r = p; g = q; b = v;	}
				else if ( index == 4 )	{	r = t; g = p; b = v;	}
				else					{	r = v; g = p; b = q;	}
			}

			set_pixel( im, i, j, 0, r );
			set_pixel( im, i, j, 1, g );
			set_pixel( im, i, j, 2, b );
		}
	}
}

void grayscale_image_3c( image im )
{
	assert( im.c == 3 );
	int i, j, k;
	float scale[] = { 0.299, 0.587, 0.114 };
	for ( j = 0; j < im.h; ++j )
	{
		for ( i = 0; i < im.w; ++i )
		{
			float val = 0;
			for ( k = 0; k < 3; ++k )
			{
				val += scale[k]*get_pixel( im, i, j, k );
			}
			im.data[0*im.h*im.w + im.w*j + i] = val;
			im.data[1*im.h*im.w + im.w*j + i] = val;
			im.data[2*im.h*im.w + im.w*j + i] = val;
		}
	}
}

image grayscale_image( image im )
{
	assert( im.c == 3 );
	int i, j, k;
	image gray = make_image( im.w, im.h, 1 );
	float scale[] = { 0.299, 0.587, 0.114 };
	for ( k = 0; k < im.c; ++k )
	{
		for ( j = 0; j < im.h; ++j )
		{
			for ( i = 0; i < im.w; ++i )
			{
				gray.data[i+im.w*j] += scale[k]*get_pixel( im, i, j, k );
			}
		}
	}
	return gray;
}

image threshold_image( image im, float thresh )
{
	int i;
	image t = make_image( im.w, im.h, im.c );
	for ( i = 0; i < im.w*im.h*im.c; ++i )
	{
		t.data[i] = im.data[i]>thresh ? 1 : 0;
	}
	return t;
}

image blend_image( image fore, image back, float alpha )
{
	assert( fore.w == back.w && fore.h == back.h && fore.c == back.c );
	image blend = make_image( fore.w, fore.h, fore.c );
	int i, j, k;
	for ( k = 0; k < fore.c; ++k )
	{
		for ( j = 0; j < fore.h; ++j )
		{
			for ( i = 0; i < fore.w; ++i )
			{
				float val = alpha * get_pixel( fore, i, j, k ) +
							(1 - alpha) * get_pixel( back, i, j, k );
				set_pixel( blend, i, j, k, val );
			}
		}
	}
	return blend;
}

void scale_image_channel( image im, int c, float v )
{
	int i, j;
	for ( j = 0; j < im.h; ++j )
	{
		for ( i = 0; i < im.w; ++i )
		{
			float pix = get_pixel( im, i, j, c );
			pix = pix*v;
			set_pixel( im, i, j, c, pix );
		}
	}
}

void translate_image_channel( image im, int c, float v )
{
	int i, j;
	for ( j = 0; j < im.h; ++j )
	{
		for ( i = 0; i < im.w; ++i )
		{
			float pix = get_pixel( im, i, j, c );
			pix = pix+v;
			set_pixel( im, i, j, c, pix );
		}
	}
}

image binarize_image( image im )
{
	image c = copy_image( im );
	int i;
	for ( i = 0; i < im.w * im.h * im.c; ++i )
	{
		if ( c.data[i] > .5 ) c.data[i] = 1;
		else c.data[i] = 0;
	}
	return c;
}

void saturate_image( image im, float sat )
{
	rgb_to_hsv( im );
	scale_image_channel( im, 1, sat );
	hsv_to_rgb( im );
	constrain_image( im );
}

void hue_image( image im, float hue )
{
	rgb_to_hsv( im );
	int i;
	for ( i = 0; i < im.w*im.h; ++i )
	{
		im.data[i] = im.data[i] + hue;
		if ( im.data[i] > 1 ) im.data[i] -= 1;
		if ( im.data[i] < 0 ) im.data[i] += 1;
	}
	hsv_to_rgb( im );
	constrain_image( im );
}

void exposure_image( image im, float sat )
{
	rgb_to_hsv( im );
	scale_image_channel( im, 2, sat );
	hsv_to_rgb( im );
	constrain_image( im );
}

void distort_image( image im, float hue, float sat, float val )
{
	rgb_to_hsv( im );
	scale_image_channel( im, 1, sat );
	scale_image_channel( im, 2, val );
	int i;
	for ( i = 0; i < im.w*im.h; ++i )
	{
		im.data[i] = im.data[i] + hue;
		if ( im.data[i] > 1 ) im.data[i] -= 1;
		if ( im.data[i] < 0 ) im.data[i] += 1;
	}
	hsv_to_rgb( im );
	constrain_image( im );
}

void random_distort_image( image im, float hue, float saturation, float exposure )
{
	float dhue = rand_uniform( -hue, hue );
	float dsat = rand_scale( saturation );
	float dexp = rand_scale( exposure );
	distort_image( im, dhue, dsat, dexp );
}

void saturate_exposure_image( image im, float sat, float exposure )
{
	rgb_to_hsv( im );
	scale_image_channel( im, 1, sat );
	scale_image_channel( im, 2, exposure );
	hsv_to_rgb( im );
	constrain_image( im );
}
// �̹��� ũ�⸦ ������ ũ��� �����Ѵ�
image resize_image( image im, int w, int h )
{
	image resized	= make_image( w, h, im.c );		// ���� �����̹���
	image part		= make_image( w, im.h, im.c );	// �κ� �����̹���
	int r, c, k;
	float w_scale = (float)(im.w - 1) / (w - 1);
	float h_scale = (float)(im.h - 1) / (h - 1);
	// ���̸� �����Ѵ�
	for ( k = 0; k < im.c; ++k )
	{
		for ( r = 0; r < im.h; ++r )
		{
			for ( c = 0; c < w; ++c )
			{
				float val = 0;
				if ( c == w-1 || im.w == 1 )
				{
					val = get_pixel( im, im.w-1, r, k );
				}
				else
				{
					float sx = c*w_scale;
					int ix = (int)sx;
					float dx = sx - ix;
					val = (1 - dx) * get_pixel( im, ix, r, k ) + dx * get_pixel( im, ix+1, r, k );
				}
				set_pixel( part, c, r, k, val );
			}
		}
	}
	// �ʺ� �����Ѵ�
	for ( k = 0; k < im.c; ++k )
	{
		for ( r = 0; r < h; ++r )
		{
			float sy = r*h_scale;
			int iy = (int)sy;
			float dy = sy - iy;
			for ( c = 0; c < w; ++c )
			{
				float val = (1-dy) * get_pixel( part, c, iy, k );
				set_pixel( resized, c, r, k, val );
			}
			if ( r == h-1 || im.h == 1 ) continue;
			for ( c = 0; c < w; ++c )
			{
				float val = dy * get_pixel( part, c, iy+1, k );
				add_pixel( resized, c, r, k, val );
			}
		}
	}

	free_image( part );
	return resized;
}


void test_resize( char *filename )
{
	image im = load_image( filename, 0, 0, 3 );
	float mag = mag_array( im.data, im.w*im.h*im.c );
	printf( "L2 Norm: %f\n", mag );
	image gray = grayscale_image( im );

	image c1 = copy_image( im );
	image c2 = copy_image( im );
	image c3 = copy_image( im );
	image c4 = copy_image( im );
	distort_image( c1, 0.1f, 1.5f, 1.5f );
	distort_image( c2, -0.1f, 0.66666f, 0.66666f );
	distort_image( c3, 0.1f, 1.5f, 0.66666f );
	distort_image( c4, 0.1f, 0.66666f, 1.5f );


	show_image( im, "Original" );
	show_image( gray, "Gray" );
	show_image( c1, "C1" );
	show_image( c2, "C2" );
	show_image( c3, "C3" );
	show_image( c4, "C4" );

	#ifdef OPENCV
	while ( 1 )
	{
		image aug = random_augment_image( im, 0.0f, 0.75f, 320, 448, 320, 320 );
		show_image( aug, "aug" );
		free_image( aug );

		float exposure		= 1.15f;
		float saturation	= 1.15f;
		float hue			= 0.05f;

		image c = copy_image( im );

		float dexp = rand_scale( exposure );
		float dsat = rand_scale( saturation );
		float dhue = rand_uniform( -hue, hue );

		distort_image( c, dhue, dsat, dexp );
		show_image( c, "rand" );
		printf( "%f %f %f\n", dhue, dsat, dexp );
		free_image( c );
		cvWaitKey( 0 );
	}
	#endif
}

// stbi �� ����Ͽ� ���Ͽ��� image ������ �ڷ� ����
image load_image_stb( char *filename, int channels )
{
	int ww, hh, cc;
	unsigned char *WonBon = stbi_load( filename, &ww, &hh, &cc, channels );

	if ( !WonBon )
	{
		fprintf( stderr
		//	, "Cannot load image \"%s\"\nSTB Reason: %s\n"	//  [7/6/2018 jobs]
			, "\"%s\" �̹��� ž�����!\nSTB ����: %s\n"
			, filename, stbi_failure_reason() );
		exit( 0 );
	}

	if ( channels ) cc = channels;

	int xx, yy, kk;
	image im = make_image( ww, hh, cc );
	// �� �ݺ�
	for ( kk=0; kk < cc; ++kk )
	{
		// ���� �ݺ�
		for ( yy=0; yy < hh; ++yy )
		{
			// ���� �ݺ�
			for ( xx=0; xx < ww; ++xx )
			{
				//				����ġ	 + ȭ��   + ��
				int src_index = ww*yy*cc + cc*xx + kk;	// ����ȭ�� ����
				//				����ġ	 + ����ġ + ȭ��
				int dst_index = ww*hh*kk + ww*yy + xx;	// �纻ȭ�� ����
				im.data[dst_index] = (float)WonBon[src_index]/255.;
			}
		}
	}

	free( WonBon );
	return im;
}
// �ϳ��� �׸����� �ڷῡ�� image �����ڷḦ ������ ä�η� ž���Ѵ�
image load_image( char *filename, int ww, int hh, int cc )
{
	#ifdef OPENCV
	image out = load_image_cv( filename, cc );
	#else
	image out = load_image_stb( filename, cc );
	#endif
	// �̹����� �ʺ�, ���̰� ���� �о� ���̹����� �ʺ�� ���̰� ���� ������ �̹���ũ�� ����
	if ( (hh && ww) && (hh != out.h || ww != out.w) )
	{
		image resized = resize_image( out, ww, hh );
		free_image( out );
		out = resized;
	}
	return out;
}
// �ϳ��� �׸����� �ڷῡ�� image �����ڷḦ 3ä�η� ž���Ѵ�
image load_image_color( char *filename, int ww, int hh )
{
	return load_image( filename, ww, hh, 3 );
}

image get_image_layer( image m, int l )
{
	image out = make_image( m.w, m.h, 1 );

	int ii;
	for ( ii=0; ii < m.h*m.w; ++ii )
	{
		out.data[ii] = m.data[ ii + l*m.h*m.w ];
	}
	return out;
}

void print_image( image m )
{
	int ii, jj, kk;
	for ( ii=0; ii < m.c; ++ii )
	{
		for ( jj=0; jj < m.h; ++jj )
		{
			for ( kk=0; kk < m.w; ++kk )
			{
				printf( "%.2lf, ", m.data[ ii*m.h*m.w + jj*m.w + kk ] );
				if ( kk > 30 ) break;
			}
			printf( "\n" );
			if ( jj > 30 ) break;
		}
		printf( "\n" );
	}
	printf( "\n" );
}
//�̹����� ���簢������ ������
image collapse_images_square( image *ims, int nn )
{
	int iSaek	= 1;
	int iTeDuRi	= 1;

	int nPanSu	= ims[0].c;

	// �Ѻ��� �� ������ ������ ����Ѵ�
	int nSu = (int)ceil( sqrt( nn * nPanSu ) );	// �Ѻ��� ������ ���Ѵ�(�Ҽ��� �ø�)

	// �׵θ��� ������ �Ѻ��� ���̸� ���
	int iNeoBi	= nSu * ( ims[0].w + iTeDuRi );
	int iNoPi	= iNeoBi;

	image filters = make_image( iNeoBi, iNoPi, iSaek );	// ������ �޸𸮸� �Ҵ��Ѵ�

	int iBo_GaRo = ims[0].w + iTeDuRi;
	int iBo_SeRo = ims[0].w + iTeDuRi;

	int ii, jj;
	// �Ǽ��� �ݺ��Ѵ�
	for ( ii=0; ii < nn; ++ii )
	{
		image copy = copy_image( ims[ii] );	// �������߰��� ���Ǿ� ����()

		//normalize_image(copy);
		if ( nPanSu == 3 && iSaek )
		{
			//embed_image( copy, filters, 0, h_offset );
		}
		else
		{
			SoBaGi_image( copy, filters, iBo_GaRo, iBo_SeRo, ii );
		}

		free_image( copy );
	}

	return filters;
}
// �̹����� �������� ������
image collapse_images_vert( image *ims, int nn )
{
	int color	= 1;
	int border	= 1;
	int hh, ww, cc;
	ww = ims[0].w;
	hh = (ims[0].h + border) * nn - border;
	cc = ims[0].c;

	if ( cc != 3 || !color )
	{
		ww = ( ww + border ) * cc - border;
		cc = 1;
	}

	image filters = make_image( ww, hh, cc );

	int ii, jj;
	// �Ǽ��� �ݺ��Ѵ�
	for ( ii=0; ii < nn; ++ii )
	{
		int h_offset = ii * ( ims[0].h + border );
		image copy = copy_image( ims[ii] );	// �������߰��� ���Ǿ� ����

		//normalize_image(copy);
		if ( cc == 3 && color )
		{
			embed_image( copy, filters, 0, h_offset );
		}
		else
		{
			for ( jj=0; jj < copy.c; ++jj )
			{
				int w_offset = jj * ( ims[0].w + border );
				image layer = get_image_layer( copy, jj );
				embed_image( layer, filters, w_offset, h_offset );
				free_image( layer );
			}
		}

		free_image( copy );
	}

	return filters;
}

image collapse_images_horz( image *ims, int nn )
{
	int color	= 1;
	int border	= 1;
	int hh, ww, cc;
	int size = ims[0].h;
	hh = size;
	ww = (ims[0].w + border) * nn - border;
	cc = ims[0].c;

	if ( cc != 3 || !color )
	{
		hh = ( hh + border ) * cc - border;
		cc = 1;
	}

	image filters = make_image( ww, hh, cc );

	int ii, jj;
	// �Ǽ��� �ݺ��Ѵ�
	for ( ii=0; ii < nn; ++ii )
	{
		int w_offset = ii * ( size + border );
		image copy = copy_image( ims[ii] );	// �������߰��� ���Ǿ� ����

		//normalize_image(copy);
		if ( cc == 3 && color )
		{
			embed_image( copy, filters, w_offset, 0 );
		}
		else
		{
			for ( jj=0; jj < copy.c; ++jj )
			{
				int h_offset = jj * ( size + border );
				image layer = get_image_layer( copy, jj );
				embed_image( layer, filters, w_offset, h_offset );
				free_image( layer );
			}
		}

		free_image( copy );
	}

	return filters;
}

void show_image_normalized( image im, const char *name )
{
	image c = copy_image( im );
	normalize_image( c );
	show_image( c, name );
	free_image( c );
}
// �̹����� ���簢������ �迭�� �����ϰ� ȭ�鿡 �����ش�
void show_images( image *ims, int n, char *window )
{
	//image m = collapse_images_vert( ims, n );
	image m = collapse_images_square( ims, n );
/*
	int w = 448;
	int h = ((float)m.h/m.w) * 448;

	if ( h > 896 )
	{
		h = 896;
		w = ((float)m.w/m.h) * 896;
	}

	image sized = resize_image(m, w, h);
*/
	//normalize_image( m );		// �ߺ����� ���� ó���� �Ѵ� [7/5/2018 jobs]
	save_image( m, window );
	show_image( m, window );
	free_image( m );
}

void free_image( image m )
{
	if ( m.data )
	{
		free( m.data );
	}
}
