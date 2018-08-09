#include "data.h"
#include "utils.h"
#include "image.h"
#include "cuda.h"

//#include "pthread.h"	//  [6/28/2018 jobs] timespec 재정의 문제

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void MNIST_ImageSwap( MNIST_ImageHeader *Hdr )
{
	Hdr->magic	= bswap( Hdr->magic );
	Hdr->GaeSu	= bswap( Hdr->GaeSu );
	Hdr->SeRo	= bswap( Hdr->SeRo );
	Hdr->GaRo	= bswap( Hdr->GaRo );
}

void MNIST_LabelSwap( MNIST_LabelHeader *Hdr )
{
	Hdr->magic	= bswap( Hdr->magic );
	Hdr->GaeSu	= bswap( Hdr->GaeSu );
}

// 파일에서 목록을 추출한다
list *get_paths( char *filename )
{
	char *path;
	//FILE *file = fopen( filename, "r" );
	FILE *file; fopen_s( &file, filename, "r" );

	if ( !file ) file_error( filename );

	list *lines = make_list();

	while ( (path=fgetl( file )) )
	{
		list_insert( lines, path );
	}
	fclose( file );

	return lines;
}


char **get_random_paths_indexes( char **paths, int nn, int mm, int *indexes )
{
	char **random_paths = calloc( nn, sizeof( char* ) );
	int ii;

	pthread_mutex_lock( &mutex );

	for ( ii=0; ii < nn; ++ii )
	{
		int index		= rand() % mm;
		indexes[ii]		= index;
		random_paths[ii] = paths[index];

		if ( ii == 0 ) printf( "%s\n", paths[index] );
	}

	pthread_mutex_unlock( &mutex );

	return random_paths;
}

// 
char **get_random_paths( char **paths, int nn, int mm )
{
	char **random_paths = calloc( nn, sizeof( char* ) );

	pthread_mutex_lock( &mutex );

	int ii;
	for ( ii=0; ii < nn; ++ii )
	{
		int index = rand() % mm;
		random_paths[ii] = paths[index];
		//if(i == 0) printf("%s\n", paths[index]);
	}
	pthread_mutex_unlock( &mutex );

	return random_paths;
}

char **find_replace_paths( char **paths, int nn, char *find, char *replace )
{
	char **replace_paths = calloc( nn, sizeof( char* ) );

	int ii;
	for ( ii=0; ii < nn; ++ii )
	{
		char replaced[4096];
		find_replace( paths[ii], find, replace, replaced );
		replace_paths[ii] = copy_string( replaced );
	}

	return replace_paths;
}

matrix load_image_paths_gray( char **paths, int nn, int ww, int hh )
{
	int ii;
	matrix XX;
	XX.rows = nn;
	XX.vals = calloc( XX.rows, sizeof( float* ) );
	XX.cols = 0;

	for ( ii=0; ii < nn; ++ii )
	{
		image im	= load_image( paths[ii], ww, hh, 3 );

		image gray	= grayscale_image( im );
		free_image( im );
		im = gray;

		XX.vals[ii]	= im.data;
		XX.cols		= im.h*im.w*im.c;
	}
	return XX;
}
// 파일이름으로 자료 적재
matrix load_image_paths( char **paths, int nn, int ww, int hh )
{
	matrix XX;
	XX.rows = nn;
	XX.vals = calloc( XX.rows, sizeof( float* ) );
	XX.cols = 0;

	int ii;
	for ( ii=0; ii < nn; ++ii )
	{
		image im	= load_image_color( paths[ii], ww, hh );
		XX.vals[ii]	= im.data;
		XX.cols		= im.h*im.w*im.c;
	}

	return XX;
}
// 자료개수에 해당하는 사비자료를 탑재한다
matrix load_image_augment_paths( char **paths	// 자료목록
							, int nn			// 자료개수
							, int min
							, int max
							, int size
							, float angle
							, float aspect
							, float hue
							, float saturation
							, float exposure
							, int center )
{
	int ii;
	matrix X;
	X.rows = nn;
	X.vals = calloc( X.rows, sizeof( float* ) );
	X.cols = 0;

	for ( ii=0; ii < nn; ++ii )
	{
		image im = load_image_color( paths[ii], 0, 0 );
		image crop;

		if ( center )
		{
			crop = center_crop_image( im, size, size );
		}
		else
		{
			crop = random_augment_image( im, angle, aspect, min, max, size, size );
		}

		int flip = rand()%2;
		if ( flip ) flip_image( crop );
		random_distort_image( crop, hue, saturation, exposure );

		/*
		show_image(im, "orig");
		show_image(crop, "crop");
		cvWaitKey(0);
		*/
		free_image( im );
		X.vals[ii]	= crop.data;
		X.cols		= crop.h*crop.w*crop.c;
	}
	return X;
}


box_label *read_boxes( char *filename, int *nn )
{
	//FILE *file = fopen( filename, "r" );
	FILE *file; fopen_s( &file, filename, "r" );

	if ( !file ) file_error( filename );
	float xx, yy, hh, ww;
	int id;
	int count = 0;
	int size = 64;
	box_label *boxes = calloc( size, sizeof( box_label ) );

	//while ( fscanf( file, "%d %f %f %f %f", &id, &xx, &yy, &ww, &hh ) == 5 )
	while ( fscanf_s( file, "%d %f %f %f %f", &id, &xx, &yy, &ww, &hh ) == 5 )
	{
		if ( count == size )
		{
			size = size * 2;
			boxes = realloc( boxes, size*sizeof( box_label ) );
		}
		boxes[count].id = id;
		boxes[count].x = xx;
		boxes[count].y = yy;
		boxes[count].h = hh;
		boxes[count].w = ww;
		boxes[count].left   = xx - ww/2;
		boxes[count].right  = xx + ww/2;
		boxes[count].top    = yy - hh/2;
		boxes[count].bottom = yy + hh/2;
		++count;
	}

	fclose( file );
	*nn = count;
	return boxes;
}

void randomize_boxes( box_label *b, int nn )
{
	int ii;
	for ( ii=0; ii < nn; ++ii )
	{
		box_label swap	= b[ii];
		int index	= rand() % nn;
		b[ii]		= b[index];
		b[index]	= swap;
	}
}

void correct_boxes( box_label *boxes
				, int n
				, float dx
				, float dy
				, float sx
				, float sy
				, int flip )
{
	int i;
	for ( i = 0; i < n; ++i )
	{
		if ( boxes[i].x == 0 && boxes[i].y == 0 )
		{
			boxes[i].x = 999999;
			boxes[i].y = 999999;
			boxes[i].w = 999999;
			boxes[i].h = 999999;
			continue;
		}
		boxes[i].left   = boxes[i].left  * sx - dx;
		boxes[i].right  = boxes[i].right * sx - dx;
		boxes[i].top    = boxes[i].top   * sy - dy;
		boxes[i].bottom = boxes[i].bottom* sy - dy;

		if ( flip )
		{
			float swap = boxes[i].left;
			boxes[i].left = 1.0f - boxes[i].right;
			boxes[i].right = 1.0f - swap;
		}

		boxes[i].left	= constrain( 0, 1, boxes[i].left );
		boxes[i].right	= constrain( 0, 1, boxes[i].right );
		boxes[i].top	= constrain( 0, 1, boxes[i].top );
		boxes[i].bottom	= constrain( 0, 1, boxes[i].bottom );

		boxes[i].x = (boxes[i].left+boxes[i].right)/2;
		boxes[i].y = (boxes[i].top+boxes[i].bottom)/2;
		boxes[i].w = (boxes[i].right - boxes[i].left);
		boxes[i].h = (boxes[i].bottom - boxes[i].top);

		boxes[i].w = constrain( 0, 1, boxes[i].w );
		boxes[i].h = constrain( 0, 1, boxes[i].h );
	}
}

void fill_truth_swag( char *path
					, float *truth
					, int classes
					, int flip
					, float dx
					, float dy
					, float sx
					, float sy )
{
	char labelpath[4096];
	find_replace( path, "images", "labels", labelpath );
	find_replace( labelpath, "JPEGImages", "labels", labelpath );
	find_replace( labelpath, ".jpg", ".txt", labelpath );
	find_replace( labelpath, ".JPG", ".txt", labelpath );
	find_replace( labelpath, ".JPEG", ".txt", labelpath );

	int count = 0;
	box_label *boxes = read_boxes( labelpath, &count );
	randomize_boxes( boxes, count );
	correct_boxes( boxes, count, dx, dy, sx, sy, flip );
	float x, y, w, h;
	int id;
	int i;

	for ( i = 0; i < count && i < 90; ++i )
	{
		x =  boxes[i].x;
		y =  boxes[i].y;
		w =  boxes[i].w;
		h =  boxes[i].h;
		id = boxes[i].id;

		if ( w < .0 || h < .0 ) continue;

		int index = (4+classes) * i;

		truth[index++] = x;
		truth[index++] = y;
		truth[index++] = w;
		truth[index++] = h;

		if ( id < classes ) truth[index+id] = 1;
	}
	free( boxes );
}

void fill_truth_region( char *path
					, float *truth
					, int classes
					, int num_boxes
					, int flip
					, float dx
					, float dy
					, float sx
					, float sy )
{
	char labelpath[4096];
	find_replace( path, "images", "labels", labelpath );
	find_replace( labelpath, "JPEGImages", "labels", labelpath );

	find_replace( labelpath, ".jpg", ".txt", labelpath );
	find_replace( labelpath, ".png", ".txt", labelpath );
	find_replace( labelpath, ".JPG", ".txt", labelpath );
	find_replace( labelpath, ".JPEG", ".txt", labelpath );
	int count = 0;
	box_label *boxes = read_boxes( labelpath, &count );
	randomize_boxes( boxes, count );
	correct_boxes( boxes, count, dx, dy, sx, sy, flip );
	float x, y, w, h;
	int id;
	int i;

	for ( i = 0; i < count; ++i )
	{
		x =  boxes[i].x;
		y =  boxes[i].y;
		w =  boxes[i].w;
		h =  boxes[i].h;
		id = boxes[i].id;

		if ( w < .005 || h < .005 ) continue;

		int col = (int)(x*num_boxes);
		int row = (int)(y*num_boxes);

		x = x*num_boxes - col;
		y = y*num_boxes - row;

		int index = (col+row*num_boxes)*(5+classes);
		if ( truth[index] ) continue;
		truth[index++] = 1;

		if ( id < classes ) truth[index+id] = 1;
		index += classes;

		truth[index++] = x;
		truth[index++] = y;
		truth[index++] = w;
		truth[index++] = h;
	}
	free( boxes );
}

void load_rle( image im, int *rle, int n )
{
	int count = 0;
	int curr = 0;
	int i, j;
	for ( i = 0; i < n; ++i )
	{
		for ( j = 0; j < rle[i]; ++j )
		{
			im.data[count++] = (float)curr;
		}
		curr = 1 - curr;
	}
	for ( ; count < im.h*im.w*im.c; ++count )
	{
		im.data[count] = (float)curr;
	}
}

void or_image( image src, image dest, int c )
{
	int i;
	for ( i = 0; i < src.w*src.h; ++i )
	{
		if ( src.data[i] ) dest.data[dest.w*dest.h*c + i] = 1;
	}
}

void exclusive_image( image src )
{
	int k, j, i;
	int s = src.w*src.h;
	for ( k = 0; k < src.c-1; ++k )
	{
		for ( i = 0; i < s; ++i )
		{
			if ( src.data[k*s + i] )
			{
				for ( j = k+1; j < src.c; ++j )
				{
					src.data[j*s + i] = 0;
				}
			}
		}
	}
}

box bound_image( image im )
{
	int x, y;
	int minx = im.w;
	int miny = im.h;
	int maxx = 0;
	int maxy = 0;
	for ( y = 0; y < im.h; ++y )
	{
		for ( x = 0; x < im.w; ++x )
		{
			if ( im.data[y*im.w + x] )
			{
				minx = (x < minx) ? x : minx;
				miny = (y < miny) ? y : miny;
				maxx = (x > maxx) ? x : maxx;
				maxy = (y > maxy) ? y : maxy;
			}
		}
	}
	box b = { (float)minx, (float)miny, (float)(maxx-minx + 1), (float)(maxy-miny + 1) };
	//printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);
	return b;
}

void fill_truth_iseg( char *path
					, int num_boxes
					, float *truth
					, int classes
					, int w
					, int h
					, augment_args aug
					, int flip
					, int mw
					, int mh )
{
	char labelpath[4096];
	find_replace( path, "images", "mask", labelpath );
	find_replace( labelpath, "JPEGImages", "mask", labelpath );
	find_replace( labelpath, ".jpg", ".txt", labelpath );
	find_replace( labelpath, ".JPG", ".txt", labelpath );
	find_replace( labelpath, ".JPEG", ".txt", labelpath );

	//FILE *file = fopen( labelpath, "r" );
	FILE *file; fopen_s( &file, labelpath, "r" );

	if ( !file ) file_error( labelpath );
	char buff[32788];
	int id;
	int i = 0;
	image part = make_image( w, h, 1 );

	//while ( (fscanf( file, "%d %s", &id, buff ) == 2) && i < num_boxes )
	while ( (fscanf_s( file, "%d %s", &id, buff ) == 2) && i < num_boxes )
	{
		int n = 0;
		int *rle = read_intlist( buff, &n, 0 );
		load_rle( part, rle, n );
		image sized = rotate_crop_image( part
									, aug.rad
									, aug.scale
									, aug.w
									, aug.h
									, aug.dx
									, aug.dy
									, aug.aspect );

		if ( flip ) flip_image( sized );

		box b = bound_image( sized );

		if ( b.w > 0 )
		{
			image crop = crop_image( sized, (int)b.x, (int)b.y, (int)b.w, (int)b.h );
			image mask = resize_image( crop, mw, mh );
			truth[i*(4 + mw*mh + 1) + 0] = (b.x + b.w/2.0f)/sized.w;
			truth[i*(4 + mw*mh + 1) + 1] = (b.y + b.h/2.0f)/sized.h;
			truth[i*(4 + mw*mh + 1) + 2] = b.w/sized.w;
			truth[i*(4 + mw*mh + 1) + 3] = b.h/sized.h;

			int j;
			for ( j = 0; j < mw*mh; ++j )
			{
				truth[i*(4 + mw*mh + 1) + 4 + j] = mask.data[j];
			}

			truth[i*(4 + mw*mh + 1) + 4 + mw*mh] = (float)id;
			free_image( crop );
			free_image( mask );
			++i;
		}
		free_image( sized );
		free( rle );
	}
	fclose( file );
	free_image( part );
}


void fill_truth_detection( char *path
						, int num_boxes
						, float *truth
						, int classes
						, int flip
						, float dx
						, float dy
						, float sx
						, float sy )
{
	char labelpath[4096];
	find_replace( path, "images", "labels", labelpath );
	find_replace( labelpath, "JPEGImages", "labels", labelpath );

	find_replace( labelpath, "raw", "labels", labelpath );
	find_replace( labelpath, ".jpg", ".txt", labelpath );
	find_replace( labelpath, ".png", ".txt", labelpath );
	find_replace( labelpath, ".JPG", ".txt", labelpath );
	find_replace( labelpath, ".JPEG", ".txt", labelpath );
	int count = 0;
	box_label *boxes = read_boxes( labelpath, &count );
	randomize_boxes( boxes, count );
	correct_boxes( boxes, count, dx, dy, sx, sy, flip );
	if ( count > num_boxes ) count = num_boxes;
	float x, y, w, h;
	int id;
	int i;
	int sub = 0;

	for ( i = 0; i < count; ++i )
	{
		x =  boxes[i].x;
		y =  boxes[i].y;
		w =  boxes[i].w;
		h =  boxes[i].h;
		id = boxes[i].id;

		if ( (w < .001 || h < .001) )
		{
			++sub;
			continue;
		}

		truth[(i-sub)*5+0] = x;
		truth[(i-sub)*5+1] = y;
		truth[(i-sub)*5+2] = w;
		truth[(i-sub)*5+3] = h;
		truth[(i-sub)*5+4] = (float)id;
	}
	free( boxes );
}

#define NUMCHARS 37

void print_letters( float *pred, int n )
{
	int i;
	for ( i = 0; i < n; ++i )
	{
		int index = max_index( pred+i*NUMCHARS, NUMCHARS );
		printf( "%c", int_to_alphanum( index ) );
	}
	printf( "\n" );
}

void fill_truth_captcha( char *path, int n, float *truth )
{
	char *begin = strrchr( path, '/' );
	++begin;
	int i;
	for ( i = 0; i < strlen( begin ) && i < n && begin[i] != '.'; ++i )
	{
		int index = alphanum_to_int( begin[i] );
		if ( index > 35 ) printf( "Bad %c\n", begin[i] );
		truth[i*NUMCHARS+index] = 1;
	}
	for ( ;i < n; ++i )
	{
		truth[i*NUMCHARS + NUMCHARS-1] = 1;
	}
}

data load_data_captcha( char **paths, int n, int m, int k, int w, int h )
{
	if ( m ) paths = get_random_paths( paths, n, m );
	data d = { 0 };
	d.shallow = 0;
	d.X = load_image_paths( paths, n, w, h );
	d.y = make_matrix( n, k*NUMCHARS );
	int i;
	for ( i = 0; i < n; ++i )
	{
		fill_truth_captcha( paths[i], k, d.y.vals[i] );
	}
	if ( m ) free( paths );
	return d;
}

data load_data_captcha_encode( char **paths, int n, int m, int w, int h )
{
	if ( m ) paths = get_random_paths( paths, n, m );
	data d = { 0 };
	d.shallow = 0;
	d.X = load_image_paths( paths, n, w, h );
	d.X.cols = 17100;
	d.y = d.X;
	if ( m ) free( paths );
	return d;
}
// 목표값 해당위치에 목표값을 "1.0" 으로 설정
void fill_truth( char *path, char **labels, int kk, float *truth )
{
	memset( truth, 0, kk*sizeof( float ) );
	// 출력단의 출력개수를 반복
	int ii;
	int count = 0;
	for ( ii=0; ii < kk; ++ii )
	{
		if ( strstr( path, labels[ii] ) )
		{
			truth[ii] = 1;
			++count;
			//printf("%s %s %d\n", path, labels[i], i);
		}
	}

	if ( count != 1 && (kk != 1 || count != 0) )
		printf( "Too many or too few labels: %d, %s\n", count, path );
}
// 계층
void fill_hierarchy( float *truth, int kk, tree *hierarchy )
{
	int jj;
	for ( jj=0; jj < kk; ++jj )
	{
		if ( truth[jj] )
		{
			int parent = hierarchy->parent[jj];
			while ( parent >= 0 )
			{
				truth[parent] = 1;
				parent = hierarchy->parent[parent];
			}
		}
	}

	int ii;
	int count = 0;
	for ( jj=0; jj < hierarchy->groups; ++jj )
	{
		//printf("%d\n", count);
		int mask = 1;
		for ( ii=0; ii < hierarchy->group_size[jj]; ++ii )
		{
			if ( truth[count + ii] )
			{
				mask = 0;
				break;
			}
		}

		if ( mask )
		{
			for ( ii=0; ii < hierarchy->group_size[jj]; ++ii )
			{
				truth[count + ii] = SECRET_NUM;
			}
		}
		count += hierarchy->group_size[jj];
	}
}
// 회귀???
matrix load_regression_labels_paths( char **paths, int n, int k )
{
	matrix y = make_matrix( n, k );
	int i, j;
	for ( i = 0; i < n; ++i )
	{
		char labelpath[4096];
		find_replace( paths[i], "images", "labels", labelpath );
		find_replace( labelpath, "JPEGImages", "labels", labelpath );
		find_replace( labelpath, ".BMP", ".txt", labelpath );
		find_replace( labelpath, ".JPEG", ".txt", labelpath );
		find_replace( labelpath, ".JPG", ".txt", labelpath );
		find_replace( labelpath, ".JPeG", ".txt", labelpath );
		find_replace( labelpath, ".Jpeg", ".txt", labelpath );
		find_replace( labelpath, ".PNG", ".txt", labelpath );
		find_replace( labelpath, ".TIF", ".txt", labelpath );
		find_replace( labelpath, ".bmp", ".txt", labelpath );
		find_replace( labelpath, ".jpeg", ".txt", labelpath );
		find_replace( labelpath, ".jpg", ".txt", labelpath );
		find_replace( labelpath, ".png", ".txt", labelpath );
		find_replace( labelpath, ".tif", ".txt", labelpath );

		//FILE *file = fopen( labelpath, "r" );
		FILE *file; fopen_s( &file, labelpath, "r" );

		for ( j = 0; j < k; ++j )
		{
			//fscanf( file, "%f", &(y.vals[i][j]) );
			fscanf_s( file, "%f", &(y.vals[i][j]) );
		}
		fclose( file );
	}
	return y;
}

matrix load_labels_paths( char **paths, int nn, char **labels, int kk, tree *hierarchy )
{
	matrix yy = make_matrix( nn, kk );	// 목표값 메모리 할당(nn행, kk열)

	int ii;
	for ( ii=0; ii < nn && labels; ++ii )
	{
		fill_truth( paths[ii], labels, kk, yy.vals[ii] );	// 목표값 위치에 목표값을 "1.0" 으로 설정

		if ( hierarchy )
		{
			fill_hierarchy( yy.vals[ii], kk, hierarchy );
		}
	}

	return yy;
}
// m 개의 열로 목표값을 만들어 n 개의 목표값자료를 추가한다
matrix load_tags_paths( char **paths, int nn, int kk )
{
	matrix yy = make_matrix( nn, kk );	// n 행, m 열의 배열을 생성한다

	int ii;
	//int count = 0;
	for ( ii=0; ii < nn; ++ii )	// 자료개수를 반복한다
	{
		char label[4096];
		find_replace( paths[ii], "images", "labels", label );
		find_replace( label, ".jpg", ".txt", label );
		//FILE *file = fopen( label, "r" );
		FILE *file; fopen_s( &file, label, "r" );

		if ( !file ) continue;	// 파일이 없으면 다음것을 찾는다

		// paths[ii] + images + labels + .jpg + .txt  파일을 열었으면
		//++count;
		int tag;

		//while ( fscanf( file, "%d", &tag ) == 1 )
		while ( fscanf_s( file, "%d", &tag ) == 1 )
		{
			if ( tag < kk )
			{
				yy.vals[ii][tag] = 1;
			}
		}
		fclose( file );
	}
	//printf("%d/%d\n", count, n);
	return yy;
}
// 파일에서 분류딱지를 추출한다
char **get_labels( char *filename )
{
	list *plist		= get_paths( filename );
	char **labels	= (char **)list_to_array( plist );
	free_list( plist );
	return labels;
}

void free_data( data d )
{
	if ( !d.shallow )
	{
		free_matrix( d.X );
		free_matrix( d.y );
	}
	else
	{
		free( d.X.vals );
		free( d.y.vals );
	}

	if ( d.labels )
	{
		free( d.labels );
	}
}

image get_segmentation_image( char *path, int w, int h, int classes )
{
	char labelpath[4096];
	find_replace( path, "images", "mask", labelpath );
	find_replace( labelpath, "JPEGImages", "mask", labelpath );
	find_replace( labelpath, ".jpg", ".txt", labelpath );
	find_replace( labelpath, ".JPG", ".txt", labelpath );
	find_replace( labelpath, ".JPEG", ".txt", labelpath );
	image mask = make_image( w, h, classes );

	//FILE *file = fopen( labelpath, "r" );
	FILE *file; fopen_s( &file, labelpath, "r" );

	if ( !file ) file_error( labelpath );
	char buff[32788];
	int id;
	image part = make_image( w, h, 1 );

	//while ( fscanf( file, "%d %s", &id, buff ) == 2 )
	while ( fscanf_s( file, "%d %s", &id, buff ) == 2 )
	{
		int n = 0;
		int *rle = read_intlist( buff, &n, 0 );
		load_rle( part, rle, n );
		or_image( part, mask, id );
		free( rle );
	}

	//exclusive_image(mask);
	fclose( file );
	free_image( part );
	return mask;
}

image get_segmentation_image2( char *path, int w, int h, int classes )
{
	char labelpath[4096];
	find_replace( path, "images", "mask", labelpath );
	find_replace( labelpath, "JPEGImages", "mask", labelpath );
	find_replace( labelpath, ".jpg", ".txt", labelpath );
	find_replace( labelpath, ".JPG", ".txt", labelpath );
	find_replace( labelpath, ".JPEG", ".txt", labelpath );
	image mask = make_image( w, h, classes+1 );

	int i;
	for ( i = 0; i < w*h; ++i )
	{
		mask.data[w*h*classes + i] = 1;
	}

	//FILE *file = fopen( labelpath, "r" );
	FILE *file; fopen_s( &file, labelpath, "r" );

	if ( !file ) file_error( labelpath );
	char buff[32788];
	int id;
	image part = make_image( w, h, 1 );

	//while ( fscanf( file, "%d %s", &id, buff ) == 2 )
	while ( fscanf_s( file, "%d %s", &id, buff ) == 2 )
	{
		int n = 0;
		int *rle = read_intlist( buff, &n, 0 );
		load_rle( part, rle, n );
		or_image( part, mask, id );

		for ( i = 0; i < w*h; ++i )
		{
			if ( part.data[i] ) mask.data[w*h*classes + i] = 0;
		}

		free( rle );
	}

	//exclusive_image(mask);
	fclose( file );
	free_image( part );
	return mask;
}

data load_data_seg( int n
				, char **paths
				, int m
				, int w
				, int h
				, int classes
				, int min
				, int max
				, float angle
				, float aspect
				, float hue
				, float saturation
				, float exposure
				, int div )
{
	char **random_paths = get_random_paths( paths, n, m );
	int i;
	data d = { 0 };
	d.shallow = 0;

	d.X.rows = n;
	d.X.vals = calloc( d.X.rows, sizeof( float* ) );
	d.X.cols = h*w*3;


	d.y.rows = n;
	d.y.cols = h*w*classes/div/div;
	d.y.vals = calloc( d.X.rows, sizeof( float* ) );

	for ( i = 0; i < n; ++i )
	{
		image orig = load_image_color( random_paths[i], 0, 0 );
		augment_args arg = random_augment_args( orig, angle, aspect, min, max, w, h );
		image sized = rotate_crop_image( orig
										, arg.rad
										, arg.scale
										, arg.w
										, arg.h
										, arg.dx
										, arg.dy
										, arg.aspect );

		int flip = rand()%2;
		if ( flip ) flip_image( sized );
		random_distort_image( sized, hue, saturation, exposure );
		d.X.vals[i] = sized.data;

		image mask = get_segmentation_image( random_paths[i], orig.w, orig.h, classes );
		//image mask = make_image(orig.w, orig.h, classes+1);
		image sized_m = rotate_crop_image( mask
										, arg.rad
										, arg.scale/div
										, arg.w/div
										, arg.h/div
										, arg.dx/div
										, arg.dy/div
										, arg.aspect );

		if ( flip ) flip_image( sized_m );
		d.y.vals[i] = sized_m.data;

		free_image( orig );
		free_image( mask );

		/*
		image rgb = mask_to_rgb( sized_m, classes );
		show_image( rgb, "part" );
		show_image( sized, "orig" );
		cvWaitKey( 0 );
		free_image( rgb );
		*/
	}

	free( random_paths );

	return d;
}

data load_data_iseg( int n
				, char **paths
				, int m
				, int w
				, int h
				, int classes
				, int boxes
				, int coords
				, int min
				, int max
				, float angle
				, float aspect
				, float hue
				, float saturation
				, float exposure )
{
	char **random_paths = get_random_paths( paths, n, m );

	data newd = { 0 };
	newd.shallow = 0;
	newd.X.rows = n;
	newd.X.vals = calloc( newd.X.rows, sizeof( float* ) );
	newd.X.cols = h*w*3;

	newd.y = make_matrix( n, (coords+1)*boxes );

	int ii;
	for ( ii=0; ii < n; ++ii )
	{
		image orig			= load_image_color( random_paths[ii], 0, 0 );
		augment_args arg	= random_augment_args( orig, angle, aspect, min, max, w, h );
		image sized			= rotate_crop_image( orig
											, arg.rad
											, arg.scale
											, arg.w
											, arg.h
											, arg.dx
											, arg.dy
											, arg.aspect );

		int flip = rand()%2;
		if ( flip ) flip_image( sized );
		random_distort_image( sized, hue, saturation, exposure );
		newd.X.vals[ii] = sized.data;
		//show_image(sized, "image");

		fill_truth_iseg( random_paths[ii]
					, boxes
					, newd.y.vals[ii]
					, classes
					, orig.w
					, orig.h
					, arg
					, flip
					, 14
					, 14 );

		free_image( orig );

		/*
		image rgb = mask_to_rgb( sized_m, classes );
		show_image( rgb, "part" );
		show_image( sized, "orig" );
		cvWaitKey( 0 );
		free_image( rgb );
		*/
	}

	free( random_paths );

	return newd;
}

data load_data_region( int n
					, char **paths
					, int m
					, int w
					, int h
					, int size
					, int classes
					, float jitter
					, float hue
					, float saturation
					, float exposure )
{
	char **random_paths = get_random_paths( paths, n, m );
	int i;
	data d = { 0 };
	d.shallow = 0;

	d.X.rows = n;
	d.X.vals = calloc( d.X.rows, sizeof( float* ) );
	d.X.cols = h*w*3;


	int k = size*size*(5+classes);
	d.y = make_matrix( n, k );
	for ( i = 0; i < n; ++i )
	{
		image orig = load_image_color( random_paths[i], 0, 0 );

		int oh = orig.h;
		int ow = orig.w;

		int dw = (int)( ow*jitter );
		int dh = (int)( oh*jitter );

		int pleft  = (int)rand_uniform( -dw, dw );
		int pright = (int)rand_uniform( -dw, dw );
		int ptop   = (int)rand_uniform( -dh, dh );
		int pbot   = (int)rand_uniform( -dh, dh );

		int swidth =  ow - pleft - pright;
		int sheight = oh - ptop - pbot;

		float sx = (float)swidth  / ow;
		float sy = (float)sheight / oh;

		int flip = rand()%2;
		image cropped = crop_image( orig, pleft, ptop, swidth, sheight );

		float dx = ( (float)pleft/ow ) / sx;
		float dy = ( (float)ptop /oh ) / sy;

		image sized = resize_image( cropped, w, h );
		if ( flip ) flip_image( sized );
		random_distort_image( sized, hue, saturation, exposure );
		d.X.vals[i] = sized.data;

		fill_truth_region( random_paths[i], d.y.vals[i], classes, size, flip, dx, dy, 1.0f/sx, 1.0f/sy );

		free_image( orig );
		free_image( cropped );
	}
	free( random_paths );
	return d;
}

data load_data_compare( int n, char **paths, int m, int classes, int w, int h )
{
	if ( m ) paths = get_random_paths( paths, 2*n, m );
	int i, j;
	data d = { 0 };
	d.shallow = 0;

	d.X.rows = n;
	d.X.vals = calloc( d.X.rows, sizeof( float* ) );
	d.X.cols = h*w*6;

	int k = 2*(classes);
	d.y = make_matrix( n, k );
	for ( i = 0; i < n; ++i )
	{
		image im1 = load_image_color( paths[i*2], w, h );
		image im2 = load_image_color( paths[i*2+1], w, h );

		d.X.vals[i] = calloc( d.X.cols, sizeof( float ) );
		memcpy( d.X.vals[i], im1.data, h*w*3*sizeof( float ) );
		memcpy( d.X.vals[i] + h*w*3, im2.data, h*w*3*sizeof( float ) );

		int id;
		float iou;

		char imlabel1[4096];
		char imlabel2[4096];
		find_replace( paths[i*2], "imgs", "labels", imlabel1 );
		find_replace( imlabel1, "jpg", "txt", imlabel1 );
		//FILE *fp1 = fopen( imlabel1, "r" );
		FILE *fp1; fopen_s( &fp1, imlabel1, "r" );

		//while ( fscanf( fp1, "%d %f", &id, &iou ) == 2 )
		while ( fscanf_s( fp1, "%d %f", &id, &iou ) == 2 )
		{
			if ( d.y.vals[i][2*id] < iou ) d.y.vals[i][2*id] = iou;
		}

		find_replace( paths[i*2+1], "imgs", "labels", imlabel2 );
		find_replace( imlabel2, "jpg", "txt", imlabel2 );
		//FILE *fp2 = fopen( imlabel2, "r" );
		FILE *fp2; fopen_s( &fp2, imlabel2, "r" );

		//while ( fscanf( fp2, "%d %f", &id, &iou ) == 2 )
		while ( fscanf_s( fp2, "%d %f", &id, &iou ) == 2 )
		{
			if ( d.y.vals[i][2*id + 1] < iou ) d.y.vals[i][2*id + 1] = iou;
		}

		for ( j = 0; j < classes; ++j )
		{
			if ( d.y.vals[i][2*j] > .5 &&  d.y.vals[i][2*j+1] < .5 )
			{
				d.y.vals[i][2*j] = 1;
				d.y.vals[i][2*j+1] = 0;
			}
			else if ( d.y.vals[i][2*j] < .5 &&  d.y.vals[i][2*j+1] > .5 )
			{
				d.y.vals[i][2*j] = 0;
				d.y.vals[i][2*j+1] = 1;
			}
			else
			{
				d.y.vals[i][2*j]   = SECRET_NUM;
				d.y.vals[i][2*j+1] = SECRET_NUM;
			}
		}
		fclose( fp1 );
		fclose( fp2 );

		free_image( im1 );
		free_image( im2 );
	}
	if ( m ) free( paths );
	return d;
}

data load_data_swag( char **paths, int n, int classes, float jitter )
{
	int index = rand()%n;
	char *random_path = paths[index];

	image orig = load_image_color( random_path, 0, 0 );
	int h = orig.h;
	int w = orig.w;

	data d = { 0 };
	d.shallow = 0;
	d.w = w;
	d.h = h;

	d.X.rows = 1;
	d.X.vals = calloc( d.X.rows, sizeof( float* ) );
	d.X.cols = h*w*3;

	int k = (4+classes)*90;
	d.y = make_matrix( 1, k );

	int dw = (int)( w*jitter );
	int dh = (int)( h*jitter );

	int pleft  = (int)rand_uniform( -dw, dw );
	int pright = (int)rand_uniform( -dw, dw );
	int ptop   = (int)rand_uniform( -dh, dh );
	int pbot   = (int)rand_uniform( -dh, dh );

	int swidth =  w - pleft - pright;
	int sheight = h - ptop - pbot;

	float sx = (float)( swidth  / w );
	float sy = (float)( sheight / h );

	int flip = rand()%2;
	image cropped = crop_image( orig, pleft, ptop, swidth, sheight );

	float dx = ( (float)pleft/w ) / sx;
	float dy = ( (float)ptop /h ) / sy;

	image sized = resize_image( cropped, w, h );
	if ( flip ) flip_image( sized );
	d.X.vals[0] = sized.data;

	fill_truth_swag( random_path, d.y.vals[0], classes, flip, dx, dy, 1.0f/sx, 1.0f/sy );

	free_image( orig );
	free_image( cropped );

	return d;
}

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
						, float exposure )
{
	char **random_paths = get_random_paths( paths, n, m );
	int i;
	data d = { 0 };
	d.shallow = 0;

	d.X.rows = n;
	d.X.vals = calloc( d.X.rows, sizeof( float* ) );
	d.X.cols = h*w*3;

	d.y = make_matrix( n, 5*boxes );
	for ( i = 0; i < n; ++i )
	{
		image orig = load_image_color( random_paths[i], 0, 0 );
		image sized = make_image( w, h, orig.c );
		fill_image( sized, .5 );

		float dw = jitter * orig.w;
		float dh = jitter * orig.h;

		float new_ar = (orig.w + rand_uniform( -dw, dw )) / (orig.h + rand_uniform( -dh, dh ));
		float scale = rand_uniform( .25, 2 );

		float nw, nh;

		if ( new_ar < 1 )
		{
			nh = scale * h;
			nw = nh * new_ar;
		}
		else
		{
			nw = scale * w;
			nh = nw / new_ar;
		}

		float dx = rand_uniform( 0, w - nw );
		float dy = rand_uniform( 0, h - nh );

		place_image( orig, (int)nw, (int)nh, (int)dx, (int)dy, sized );

		random_distort_image( sized, hue, saturation, exposure );

		int flip = rand()%2;
		if ( flip ) flip_image( sized );
		d.X.vals[i] = sized.data;


		fill_truth_detection( random_paths[i], boxes, d.y.vals[i], classes, flip, -dx/w, -dy/h, nw/w, nh/h );

		free_image( orig );
	}
	free( random_paths );
	return d;
}
/// MNIST 자료 탑재
data make_data_mnist( char **paths )
{
	data newd = { 0 };
	newd.shallow = 0;

	MNIST_ImageHeader imgHeader;
	MNIST_LabelHeader lblHeader;

	FILE *rd_image; fopen_s( &rd_image, paths[0], "r" );	// 이미지자료
	fread( &imgHeader, sizeof( MNIST_ImageHeader ), 1, rd_image );
	MNIST_ImageSwap( &imgHeader );

	FILE *rd_label; fopen_s( &rd_label, paths[1], "r" );	// 꼬리표
	fread( &lblHeader, sizeof( MNIST_LabelHeader ), 1, rd_label );
	MNIST_LabelSwap( &lblHeader );

	if ( imgHeader.magic != MNIST_IMAGE_MAGIC )
	{
		printf( "MNIST 입력자료 탑재 및 생성 실패!!!\n" );
		printf( "이미지 고유번호: %d, 파일자료 고유번호: %d 로 다름", MNIST_IMAGE_MAGIC, imgHeader.magic );

		return newd;
	}
	else if ( lblHeader.magic != MNIST_LABEL_MAGIC )
	{
		printf( "MNIST 꼬리표자료 탑재 및 생성 실패!!!\n" );
		printf( "꼬리표 고유번호: %d, 파일자료 고유번호: %d 로 다름", MNIST_LABEL_MAGIC, lblHeader.magic );

		return newd;
	}

	matrix XX;	// 사비값
	XX.rows = imgHeader.GaeSu;
	XX.vals = calloc( XX.rows, sizeof( float* ) );
	XX.cols = 0;

	matrix yy;	// 꼬리표
	yy.rows = lblHeader.GaeSu;
	yy.vals = calloc( yy.rows, sizeof( float* ) );
	yy.cols = 0;

	int GaRo	= imgHeader.GaRo;
	int SeRo	= imgHeader.SeRo;
	int Bo		= GaRo*SeRo;

	newd.labels	= calloc( yy.rows, sizeof( int ) );
	//void **DdakJi	= calloc( yy.rows, sizeof( void* ) );
	//int *DdakJi	= (int *)malloc( yy.rows );
	unsigned char *chImg	= (unsigned char*)malloc( Bo );
	unsigned char Lbl;

	int ii, jj;
	for ( ii=0; ii<imgHeader.GaeSu; ++ii )
	{
		fseek( rd_image, 16+Bo*ii, SEEK_SET );
		fseek( rd_label, 8+ii, SEEK_SET );

		fread( chImg, Bo, 1, rd_image );
		fread( &Lbl, 1, 1, rd_label );

		image SaBi		= make_image( GaRo, SeRo, 1 );	// 이미지메모리 할당
		image MokPyo	= make_image( 10, 1, 1 );		// 목표값메모리 할당

		for ( jj=0; jj < Bo; ++jj )
		{
			SaBi.data[jj]	= (float)chImg[jj] / 255.0f;
		}

		XX.vals[ii]	= SaBi.data;
		XX.cols		= SaBi.w*SaBi.h*SaBi.c;

		int mGabWiChi	= (int)Lbl;
		MokPyo.data[mGabWiChi]	= 1.0;
		yy.vals[ii]	= MokPyo.data;
		yy.cols		= MokPyo.w*MokPyo.h*MokPyo.c;

		//DdakJi[ii]		= mGabWiChi;
		newd.labels[ii]	= mGabWiChi;
	}

	newd.X = XX;
	newd.y = yy;
	//newd.labels	= DdakJi;

	free( chImg );

	fclose( rd_image );
	fclose( rd_label );

	return newd;
}

data load_data_mnist( data *WonBon
					, int nn	// 사비자료 개수
					, int mm	// 자료 총 개수
					, int kk	// 분류 개수
					, int ww
					, int hh )
{
//	pthread_mutex_lock( &mutex );

	data newd = { 0 };
	newd.shallow = 0;

	matrix XX;	// 사비값
	XX.rows = nn;
	XX.vals = calloc( XX.rows, sizeof( float* ) );
	XX.cols = 0;

	matrix yy;
	yy.vals	= calloc( nn, sizeof( float* ) );
	XX.cols = 0;

	int ii, jj;

	//random_device	BbuRiGae;
	//srand( (unsigned int)time( NULL ) );

	for ( ii=0; ii < nn; ++ii )
	{
		srand( (unsigned int)time( NULL ) );

		//int WiChi = rand() % mm;
		int WiChi = (unsigned int)( UNIFORM_ZERO_THRU_ONE * mm );;

		XX.vals[ii]	= WonBon->X.vals[WiChi];
		XX.cols		= WonBon->X.cols;

		yy.vals[ii]	= WonBon->y.vals[WiChi];
		yy.cols		= WonBon->y.cols;
	}

	newd.X = XX;
	newd.y = yy;

//	pthread_mutex_unlock( &mutex );

	return newd;
}

// 망 종류별로 자료를 탑재한다
void *load_thread( void *ptr )
{
	//printf("Loading data: %d\n", rand());
	load_args arg = *(struct load_args*)ptr;

	if ( arg.exposure == 0 )	arg.exposure	= 1;
	if ( arg.saturation == 0 )	arg.saturation	= 1;
	if ( arg.aspect == 0 )		arg.aspect		= 1;

	if ( arg.type == OLD_CLASSIFICATION_DATA )
	{
		*arg.d = load_data_old( arg.paths
			, arg.n
			, arg.m
			, arg.labels
			, arg.classes
			, arg.w
			, arg.h );
	}
	else if ( arg.type == REGRESSION_DATA )
	{
		*arg.d = load_data_regression( arg.paths
			, arg.n
			, arg.m
			, arg.classes
			, arg.min
			, arg.max
			, arg.size
			, arg.angle
			, arg.aspect
			, arg.hue
			, arg.saturation
			, arg.exposure );
	}
	else if ( arg.type == CLASSIFICATION_DATA )
	{
		*arg.d = load_data_augment( arg.paths
			, arg.n
			, arg.m
			, arg.labels
			, arg.classes
			, arg.hierarchy
			, arg.min
			, arg.max
			, arg.size
			, arg.angle
			, arg.aspect
			, arg.hue
			, arg.saturation
			, arg.exposure
			, arg.center );
	}
	else if ( arg.type == SUPER_DATA )
	{
		*arg.d = load_data_super( arg.paths, arg.n, arg.m, arg.w, arg.h, arg.scale );
	}
	else if ( arg.type == WRITING_DATA )
	{
		*arg.d = load_data_writing( arg.paths, arg.n, arg.m, arg.w, arg.h, arg.out_w, arg.out_h );
	}
	else if ( arg.type == INSTANCE_DATA )
	{
		*arg.d = load_data_iseg( arg.n
			, arg.paths
			, arg.m
			, arg.w
			, arg.h
			, arg.classes
			, arg.num_boxes
			, arg.coords
			, arg.min
			, arg.max
			, arg.angle
			, arg.aspect
			, arg.hue
			, arg.saturation
			, arg.exposure );
	}
	else if ( arg.type == SEGMENTATION_DATA )
	{
		*arg.d = load_data_seg( arg.n
			, arg.paths
			, arg.m
			, arg.w
			, arg.h
			, arg.classes
			, arg.min
			, arg.max
			, arg.angle
			, arg.aspect
			, arg.hue
			, arg.saturation
			, arg.exposure
			, arg.scale );
	}
	else if ( arg.type == REGION_DATA )
	{
		*arg.d = load_data_region( arg.n
			, arg.paths
			, arg.m
			, arg.w
			, arg.h
			, arg.num_boxes
			, arg.classes
			, arg.jitter
			, arg.hue
			, arg.saturation
			, arg.exposure );
	}
	else if ( arg.type == DETECTION_DATA )
	{
		*arg.d = load_data_detection( arg.n
			, arg.paths
			, arg.m
			, arg.w
			, arg.h
			, arg.num_boxes
			, arg.classes
			, arg.jitter
			, arg.hue
			, arg.saturation
			, arg.exposure );
	}
	else if ( arg.type == SWAG_DATA )
	{
		*arg.d = load_data_swag( arg.paths, arg.n, arg.classes, arg.jitter );
	}
	else if ( arg.type == COMPARE_DATA )
	{
		*arg.d = load_data_compare( arg.n, arg.paths, arg.m, arg.classes, arg.w, arg.h );
	}
	else if ( arg.type == IMAGE_DATA )
	{
		*(arg.im) = load_image_color( arg.path, 0, 0 );
		*(arg.resized) = resize_image( *(arg.im), arg.w, arg.h );
	}
	else if ( arg.type == LETTERBOX_DATA )
	{
		*(arg.im) = load_image_color( arg.path, 0, 0 );
		*(arg.resized) = letterbox_image( *(arg.im), arg.w, arg.h );
	}
	else if ( arg.type == TAG_DATA )
	{
		*arg.d = load_data_tag( arg.paths
			, arg.n
			, arg.m
			, arg.classes
			, arg.min
			, arg.max
			, arg.size
			, arg.angle
			, arg.aspect
			, arg.hue
			, arg.saturation
			, arg.exposure );
	}
	else if ( arg.type == MNIST_DATA )
	{
		*arg.d	= load_data_mnist( arg.JaRyo_MNIST
				, arg.n
				, arg.m
				, arg.classes
				, arg.w
				, arg.h );
	}
	else if ( arg.type == BYEORIM_DATA )
	{

	}

	free( ptr );

	return 0;
}

pthread_t load_data_in_thread( load_args args )
{
	pthread_t thread;
	struct load_args *ptr = calloc( 1, sizeof( struct load_args ) );
	*ptr = args;

	if ( pthread_create( &thread, 0, load_thread, ptr ) )
		//error( "Thread creation failed" );	//  [7/16/2018 jobs]
		error( "쓰레드 생성실패!..." );		//  [7/16/2018 jobs]

	return thread;
}

void *load_threads( void *ptr )
{
	load_args args	= *(load_args *)ptr;

	if ( args.threads == 0 ) args.threads = 1;

	data *out = args.d;
	int total = args.n;
	free( ptr );

	data		*buffers	= calloc( args.threads, sizeof( data ) );
	pthread_t	*threads	= calloc( args.threads, sizeof( pthread_t ) );

	int ii;
	for ( ii=0; ii < args.threads; ++ii )
	{
		args.d		= buffers + ii;
		args.n		= ((ii+1) * total)/args.threads - (ii*total)/args.threads;
		threads[ii]	= load_data_in_thread( args );
	}

	for ( ii=0; ii < args.threads; ++ii )
	{
		pthread_join( threads[ii], 0 );
	}

	*out = concat_datas( buffers, args.threads );	// 버퍼에 자료를 엮어서 담아온다
	out->shallow = 0;

	for ( ii=0; ii < args.threads; ++ii )
	{
		buffers[ii].shallow = 1;
		free_data( buffers[ii] );
	}

	free( buffers );
	free( threads );

	return 0;
}

void load_data_blocking( load_args args )
{
	struct load_args *ptr = calloc( 1, sizeof( struct load_args ) );
	*ptr = args;
	load_thread( ptr );
}

pthread_t load_data( load_args args )
{
	pthread_t thread;
	struct load_args *ptr = calloc( 1, sizeof( struct load_args ) );
	*ptr = args;

	if ( pthread_create( &thread, 0, load_threads, ptr ) )
		error( "Thread creation failed" );

	return thread;
}

data load_data_writing( char **paths, int nn, int mm, int ww, int hh, int out_w, int out_h )
{
	if ( mm ) paths = get_random_paths( paths, nn, mm );

	char **replace_paths = find_replace_paths( paths, nn, ".png", "-label.png" );

	data newd	 = { 0 };
	newd.shallow = 0;
	newd.X		 = load_image_paths( paths, nn, ww, hh );
	newd.y		 = load_image_paths_gray( replace_paths, nn, out_w, out_h );

	if ( mm ) free( paths );
	int ii;
	for ( ii=0; ii < nn; ++ii ) free( replace_paths[ii] );
	free( replace_paths );

	return newd;
}

data load_data_old( char **paths, int nn, int mm, char **labels, int kk, int ww, int hh )
{
	// nn 개수만큼 뿌려서 사비자료 파일이름을 가져온다
	if ( mm ) paths = get_random_paths( paths, nn, mm );

	data newd = { 0 };
	newd.shallow = 0;
	// nn 개수만큼 사비자료를 가져온다
	newd.X = load_image_paths( paths, nn, ww, hh );
	// nn 개수만큼 목표자료를 가져온다
	newd.y = load_labels_paths( paths, nn, labels, kk, 0 );

	if ( mm ) free( paths );

	return newd;
}

/*
data load_data_study( char **paths
					, int n
					, int m
					, char **labels
					, int k
					, int min
					, int max
					, int size
					, float angle
					, float aspect
					, float hue
					, float saturation
					, float exposure)
{
   data d = {0};
   d.indexes = calloc(n, sizeof(int));

   if(m) paths = get_random_paths_indexes(paths, n, m, d.indexes);

   d.shallow = 0;
   d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure);
   d.y = load_labels_paths(paths, n, labels, k);

   if(m) free(paths);
   return d;
}
 */

data load_data_super( char **paths, int n, int m, int w, int h, int scale )
{
	if ( m ) paths = get_random_paths( paths, n, m );
	data d = { 0 };
	d.shallow = 0;

	int i;
	d.X.rows = n;
	d.X.vals = calloc( n, sizeof( float* ) );
	d.X.cols = w*h*3;

	d.y.rows = n;
	d.y.vals = calloc( n, sizeof( float* ) );
	d.y.cols = w*scale * h*scale * 3;

	for ( i = 0; i < n; ++i )
	{
		image im = load_image_color( paths[i], 0, 0 );
		image crop = random_crop_image( im, w*scale, h*scale );
		int flip = rand()%2;
		if ( flip ) flip_image( crop );
		image resize = resize_image( crop, w, h );
		d.X.vals[i] = resize.data;
		d.y.vals[i] = crop.data;
		free_image( im );
	}

	if ( m ) free( paths );
	return d;
}

data load_data_regression( char **paths
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
						, float exposure )
{
	if ( m ) paths = get_random_paths( paths, n, m );

	data d = { 0 };
	d.shallow = 0;
	d.X = load_image_augment_paths( paths
								, n
								, min
								, max
								, size
								, angle
								, aspect
								, hue
								, saturation
								, exposure, 0 );
	d.y = load_regression_labels_paths( paths, n, k );

	if ( m ) free( paths );
	return d;
}

data select_data( data *orig, int *inds )
{
	data d = { 0 };
	d.shallow = 1;
	d.w = orig[0].w;
	d.h = orig[0].h;

	d.X.rows = orig[0].X.rows;
	d.y.rows = orig[0].X.rows;

	d.X.cols = orig[0].X.cols;
	d.y.cols = orig[0].y.cols;

	d.X.vals = calloc( orig[0].X.rows, sizeof( float * ) );
	d.y.vals = calloc( orig[0].y.rows, sizeof( float * ) );
	int i;
	for ( i = 0; i < d.X.rows; ++i )
	{
		d.X.vals[i] = orig[inds[i]].X.vals[i];
		d.y.vals[i] = orig[inds[i]].y.vals[i];
	}
	return d;
}

data *tile_data( data orig, int divs, int size )
{
	data *outd = calloc( divs*divs, sizeof( data ) );
	int ii, jj;

	#pragma omp parallel for
	for ( ii=0; ii < divs*divs; ++ii )
	{
		data newd;
		newd.shallow = 0;
		newd.w		= orig.w/divs * size;
		newd.h		= orig.h/divs * size;
		newd.X.rows	= orig.X.rows;
		newd.X.cols	= newd.w*newd.h*3;
		newd.X.vals	= calloc( newd.X.rows, sizeof( float* ) );

		newd.y = copy_matrix( orig.y );

		#pragma omp parallel for
		for ( jj=0; jj < orig.X.rows; ++jj )
		{
			int xx = (ii%divs) * orig.w / divs - (newd.w - orig.w/divs)/2;
			int yy = (ii/divs) * orig.h / divs - (newd.h - orig.h/divs)/2;
			image im = float_to_image( orig.w, orig.h, 3, orig.X.vals[jj] );
			newd.X.vals[jj] = crop_image( im, xx, yy, newd.w, newd.h ).data;
		}

		outd[ii] = newd;
	}

	return outd;
}

data resize_data( data orig, int ww, int hh )
{
	int ii;

	data newd	= { 0 };
	newd.shallow = 0;
	newd.w		= ww;
	newd.h		= hh;
	newd.X.rows = orig.X.rows;
	newd.X.cols = ww*hh*3;
	newd.X.vals = calloc( newd.X.rows, sizeof( float* ) );

	newd.y = copy_matrix( orig.y );

	#pragma omp parallel for
	for ( ii=0; ii < orig.X.rows; ++ii )
	{
		image im = float_to_image( orig.w, orig.h, 3, orig.X.vals[ii] );
		newd.X.vals[ii] = resize_image( im, ww, hh ).data;
	}

	return newd;
}
// 분류망에 필요한 자료를 탑재한다
data load_data_augment( char **paths	// 모든(사비)자료 파일목록
					, int nn			// 자료개수
					, int mm			// 자료 총개수
					, char **labels
					, int kk
					, tree *hierarchy
					, int min
					, int max
					, int size
					, float angle
					, float aspect
					, float hue
					, float saturation
					, float exposure
					, int center )
{
	// n개의 자료(사비 파일명)목록을 뿌려서 가져온다(총 자료개수(m 개)를 넘지않게)
	if ( mm ) paths = get_random_paths( paths, nn, mm );

	data newd = { 0 };
	newd.shallow = 0;
	newd.w = size;
	newd.h = size;
	// 사비값(이미지 형태)을 탑재한다
	newd.X = load_image_augment_paths( paths	// 자료목록
								, nn			// 자료개수
								, min
								, max
								, size
								, angle
								, aspect
								, hue
								, saturation
								, exposure
								, center );
	// 목표값을 탑재한다
	newd.y = load_labels_paths( paths, nn, labels, kk, hierarchy );

	if ( mm ) free( paths );

	return newd;
}

data load_data_tag( char **paths
				, int nn
				, int mm
				, int kk
				, int min
				, int max
				, int size
				, float angle
				, float aspect
				, float hue
				, float saturation
				, float exposure )
{
	if ( mm ) paths = get_random_paths( paths, nn, mm );

	data newd = { 0 };
	newd.w = size;
	newd.h = size;
	newd.shallow = 0;
	newd.X = load_image_augment_paths( paths
								, nn
								, min
								, max
								, size
								, angle
								, aspect
								, hue
								, saturation
								, exposure
								, 0 );
	newd.y = load_tags_paths( paths, nn, kk );

	if ( mm ) free( paths );

	return newd;
}
// 두 행렬을 엮는다
matrix concat_matrix( matrix m1, matrix m2 )
{
	int ii, count = 0;
	matrix newm;
	newm.cols = m1.cols;
	newm.rows = m1.rows+m2.rows;
	newm.vals = calloc( m1.rows + m2.rows, sizeof( float* ) );	// 엮을자료 크기만큼 새 행렬 메모리 할당
	// 새 행렬에 m1 값을 담는다
	for ( ii=0; ii < m1.rows; ++ii )
	{
		newm.vals[count++] = m1.vals[ii];
	}
	// 새 행렬에 m2 값을 추가로 담는다
	for ( ii=0; ii < m2.rows; ++ii )
	{
		newm.vals[count++] = m2.vals[ii];
	}

	return newm;
}
// 두 자료를 엮는다
data concat_data( data d1, data d2 )
{
	data newd = { 0 };
	newd.shallow = 1;
	newd.X = concat_matrix( d1.X, d2.X );	// 입력값을 엮는다
	newd.y = concat_matrix( d1.y, d2.y );	// 목표값을 엮는다
	newd.w = d1.w;
	newd.h = d1.h;
	return newd;
}
// 자료를 지정한 개수만큼 엮는다
data concat_datas( data *dd, int nn )
{
	int ii;
	data outd = { 0 };

	for ( ii=0; ii < nn; ++ii )
	{
		data newd = concat_data( dd[ii], outd );	// 쓰레드개수만큼 자료를 엮는다
		free_data( outd );
		outd = newd;
	}

	return outd;
}

data load_categorical_data_csv( char *filename, int target, int kk )
{
	data newd		= { 0 };
	newd.shallow	= 0;
	matrix XX		= csv_to_matrix( filename );
	float *truth_1d	= pop_column( &XX, target );
	float **truth	= one_hot_encode( truth_1d, XX.rows, kk );
	matrix YY;
	YY.rows	= XX.rows;
	YY.cols	= kk;
	YY.vals	= truth;
	newd.X	= XX;
	newd.y	= YY;

	free( truth_1d );

	return newd;
}

data load_cifar10_data( char *filename )
{
	long ii, jj;

	data newd		= { 0 };
	newd.shallow	= 0;
	matrix XX		= make_matrix( 10000, 3072 );
	matrix YY		= make_matrix( 10000, 10 );
	newd.X			= XX;
	newd.y			= YY;

	//FILE *fp = fopen( filename, "rb" );
	FILE *fp; fopen_s( &fp, filename, "rb" );

	if ( !fp ) file_error( filename );

	for ( ii=0; ii < 10000; ++ii )
	{
		unsigned char bytes[3073];
		fread( bytes, 1, 3073, fp );
		int class = bytes[0];
		YY.vals[ii][class] = 1;

		for ( jj=0; jj < XX.cols; ++jj )
		{
			XX.vals[ii][jj] = (float)bytes[jj+1];
		}
	}

	scale_data_rows( newd, 1.0f/255 );
	//normalize_data_rows(d);
	fclose( fp );
	return newd;
}

void get_random_batch( data d, int n, float *X, float *y )
{
	int j;
	for ( j = 0; j < n; ++j )
	{
		int index = rand()%d.X.rows;
		memcpy( X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof( float ) );
		memcpy( y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof( float ) );
	}
}
// 신경망에 사비값을 전달하기 위해 한동이에 해당하는 사리수만큼 자료를 복사사한다
void get_next_batch( data d, int nn, int offset, float *X, float *y )
{
	int jj;
	for ( jj=0; jj < nn; ++jj )
	{
		int index = offset + jj;
		// 입력값을 사리수만큼 복사한다
		memcpy( X+jj*d.X.cols, d.X.vals[index], d.X.cols*sizeof( float ) );
		// 목표값을 사리수만큼 복사한다
		if ( y )
			memcpy( y+jj*d.y.cols, d.y.vals[index], d.y.cols*sizeof( float ) );
	}
}

void smooth_data( data d )
{
	int i, j;
	float scale = 1.0f / d.y.cols;
	float eps = 0.1f;
	for ( i = 0; i < d.y.rows; ++i )
	{
		for ( j = 0; j < d.y.cols; ++j )
		{
			d.y.vals[i][j] = eps * scale + (1-eps) * d.y.vals[i][j];
		}
	}
}

data load_all_cifar10()
{
	int ii, jj, bb;
	data newd	= { 0 };
	newd.shallow = 0;
	matrix XX	= make_matrix( 50000, 3072 );	// 3072 = 32x32x3 (가로x세로x색)
	matrix yy	= make_matrix( 50000, 10 );
	newd.X		= XX;
	newd.y		= yy;
	// 5 종의 자료뭉치를 반복한다
	for ( bb=0; bb < 5; ++bb )
	{
		char buff[256];
		//sprintf( buff, "data/cifar/cifar-10-batches-bin/data_batch_%d.bin", b+1 );
		sprintf_s( buff, 256, "data/cifar/cifar-10-batches-bin/data_batch_%d.bin", bb+1 );

		//FILE *fp = fopen( buff, "rb" );
		FILE *fp; fopen_s( &fp, buff, "rb" );

		if ( !fp ) file_error( buff );
		// 1 종당 10000 개의 자료가 있다
		for ( ii=0; ii < 10000; ++ii )
		{
			unsigned char bytes[3073];
			fread( bytes, 1, 3073, fp );		// 목표값을 포함한 자료를 읽는다

			int class = bytes[0];				// 목표값 위지
			yy.vals[ii+bb*10000][class] = 1.0f;	// 목표값 해당위지를 1.0 으로 설정한다
			// 입력값 자료를 적재한다
			for ( jj=0; jj < XX.cols; ++jj )
			{
				XX.vals[ii+bb*10000][jj] = (float)bytes[jj+1];
			}
		}

		fclose( fp );
	}

	//normalize_data_rows(d);
	scale_data_rows( newd, 1.0f/255 );
	smooth_data( newd );
	return newd;
}

data load_go( char *filename )
{
	//FILE *fp = fopen( filename, "rb" );
	FILE *fp; fopen_s( &fp, filename, "rb" );

	matrix X = make_matrix( 3363059, 361 );
	matrix y = make_matrix( 3363059, 361 );
	int row, col;

	if ( !fp ) file_error( filename );
	char *label;
	int count = 0;

	while ( (label = fgetl( fp )) )
	{
		int i;
		if ( count == X.rows )
		{
			X = resize_matrix( X, count*2 );
			y = resize_matrix( y, count*2 );
		}

		//sscanf( label, "%d %d", &row, &col );
		sscanf_s( label, "%d %d", &row, &col );
		char *board = fgetl( fp );

		int index = row*19 + col;
		y.vals[count][index] = 1;

		for ( i = 0; i < 19*19; ++i )
		{
			float val = 0;
			if ( board[i] == '1' ) val = 1;
			else if ( board[i] == '2' ) val = -1;
			X.vals[count][i] = val;
		}

		++count;
		free( label );
		free( board );
	}

	X = resize_matrix( X, count );
	y = resize_matrix( y, count );

	data d = { 0 };
	d.shallow = 0;
	d.X = X;
	d.y = y;

	fclose( fp );

	return d;
}


void randomize_data( data d )
{
	int i;
	for ( i = d.X.rows-1; i > 0; --i )
	{
		int index = rand()%i;
		float *swap = d.X.vals[index];
		d.X.vals[index] = d.X.vals[i];
		d.X.vals[i] = swap;

		swap = d.y.vals[index];
		d.y.vals[index] = d.y.vals[i];
		d.y.vals[i] = swap;
	}
}

void scale_data_rows( data d, float s )
{
	int i;
	for ( i = 0; i < d.X.rows; ++i )
	{
		scale_array( d.X.vals[i], d.X.cols, s );
	}
}

void translate_data_rows( data d, float s )
{
	int i;
	for ( i = 0; i < d.X.rows; ++i )
	{
		translate_array( d.X.vals[i], d.X.cols, s );
	}
}

data copy_data( data d )
{
	data c = { 0 };
	c.w = d.w;
	c.h = d.h;
	c.shallow = 0;
	c.num_boxes = d.num_boxes;
	c.boxes = d.boxes;
	c.X = copy_matrix( d.X );
	c.y = copy_matrix( d.y );
	return c;
}

void normalize_data_rows( data d )
{
	int i;
	for ( i = 0; i < d.X.rows; ++i )
	{
		normalize_array( d.X.vals[i], d.X.cols );
	}
}

data get_data_part( data d, int part, int total )
{
	data p = { 0 };
	p.shallow = 1;
	p.X.rows = d.X.rows * (part + 1) / total - d.X.rows * part / total;
	p.y.rows = d.y.rows * (part + 1) / total - d.y.rows * part / total;
	p.X.cols = d.X.cols;
	p.y.cols = d.y.cols;
	p.X.vals = d.X.vals + d.X.rows * part / total;
	p.y.vals = d.y.vals + d.y.rows * part / total;
	return p;
}

data get_random_data( data d, int num )
{
	data r = { 0 };
	r.shallow = 1;

	r.X.rows = num;
	r.y.rows = num;

	r.X.cols = d.X.cols;
	r.y.cols = d.y.cols;

	r.X.vals = calloc( num, sizeof( float * ) );
	r.y.vals = calloc( num, sizeof( float * ) );

	int i;
	for ( i = 0; i < num; ++i )
	{
		int index = rand()%d.X.rows;
		r.X.vals[i] = d.X.vals[index];
		r.y.vals[i] = d.y.vals[index];
	}
	return r;
}

data *split_data( data d, int part, int total )
{
	data *split = calloc( 2, sizeof( data ) );
	int i;
	int start = part*d.X.rows/total;
	int end = (part+1)*d.X.rows/total;
	data train;
	data test;
	train.shallow = test.shallow = 1;

	test.X.rows = test.y.rows = end-start;
	train.X.rows = train.y.rows = d.X.rows - (end-start);
	train.X.cols = test.X.cols = d.X.cols;
	train.y.cols = test.y.cols = d.y.cols;

	train.X.vals	= calloc( train.X.rows, sizeof( float* ) );
	test.X.vals		= calloc( test.X.rows, sizeof( float* ) );
	train.y.vals	= calloc( train.y.rows, sizeof( float* ) );
	test.y.vals		= calloc( test.y.rows, sizeof( float* ) );

	for ( i = 0; i < start; ++i )
	{
		train.X.vals[i] = d.X.vals[i];
		train.y.vals[i] = d.y.vals[i];
	}

	for ( i = start; i < end; ++i )
	{
		test.X.vals[i-start] = d.X.vals[i];
		test.y.vals[i-start] = d.y.vals[i];
	}

	for ( i = end; i < d.X.rows; ++i )
	{
		train.X.vals[i-(end-start)] = d.X.vals[i];
		train.y.vals[i-(end-start)] = d.y.vals[i];
	}

	split[0] = train;
	split[1] = test;

	return split;
}

