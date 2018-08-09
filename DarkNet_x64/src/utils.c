
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
//#include "unistd.h"		// rand �����ǹ��� [6/28/2018 jobs]
#include <float.h>
#include <limits.h>
#include <time.h>
//#include <sys/time.h>		// ���н� [7/4/2018 jobs]
#include "gettimeofday.h"	// ������ [7/4/2018 jobs]

#include "utils.h"

/*
// old timing. is it better? who knows!!
double get_wall_time()
{
	struct timeval time;
	if (gettimeofday(&time,NULL))
	{
		return 0;
	}
	return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
*/

double what_time_is_it_now()
{
	struct timeval time;

	if ( gettimeofday( &time, NULL ) )
	{
		return 0;
	}

	return (double)time.tv_sec + (double)time.tv_usec * 0.000001;
}

int *read_intlist( char *gpu_list, int *ngpus, int dd )
{
	int *gpus = 0;

	if ( gpu_list )
	{
		int len = (int)strlen( gpu_list );
		*ngpus = 1;

		int ii;
		for ( ii=0; ii < len; ++ii )
		{
			if ( gpu_list[ii] == ',' )	++*ngpus;
		}

		gpus = calloc( *ngpus, sizeof( int ) );

		for ( ii=0; ii < *ngpus; ++ii )
		{
			gpus[ii] = atoi( gpu_list );
			gpu_list = strchr( gpu_list, ',' )+1;
		}
	}
	else
	{
		gpus = calloc( 1, sizeof( float ) );
		*gpus = dd;
		*ngpus = 1;
	}

	return gpus;
}
// ��� ��(�з��� ����???) �б�
int *read_map( char *filename )
{
	int nn		= 0;
	int *map	= 0;
	char *str;
	//FILE *file = fopen( filename, "r" );
	FILE *file; fopen_s( &file, filename, "r" );

	if ( !file ) file_error( filename );

	while ( (str=fgetl( file )) )
	{
		++nn;
		map = realloc( map, nn*sizeof( int ) );
		map[nn-1] = atoi( str );
	}

	return map;
}

void sorta_shuffle( void *arr, size_t nn, size_t size, size_t sections )
{
	size_t ii;
	for ( ii=0; ii < sections; ++ii )
	{
		size_t start = nn*ii/sections;
		size_t end = nn*(ii+1)/sections;
		size_t num = end-start;
		//shuffle( arr+(start*size), num, size );
		shuffle( (char*)arr+(start*size), num, size );
	}
}

void shuffle( void *arr, size_t nn, size_t size )
{
	size_t ii;
	void *swp = calloc( 1, size );

	for ( ii=0; ii < nn-1; ++ii )
	{
		size_t jj = ii + rand()/(RAND_MAX / (nn-ii)+1);
		//memcpy(swp,          arr+(j*size), size);
		//memcpy(arr+(j*size), arr+(i*size), size);
		//memcpy(arr+(i*size), swp,          size);
		memcpy( swp,					(char*)arr+(jj*size),	size );
		memcpy( (char*)arr+(jj*size),	(char*)arr+(ii*size),	size );
		memcpy( (char*)arr+(ii*size),	swp,					size );
	}
}

int *random_index_order( int min, int max )
{
	int *inds = calloc( max-min, sizeof( int ) );

	int ii;
	for ( ii=min; ii < max; ++ii )
	{
		inds[ii] = ii;
	}

	for ( ii=min; ii < max-1; ++ii )
	{
		int swap = inds[ii];
		int index = ii + rand()%(max-ii);
		inds[ii] = inds[index];
		inds[index] = swap;
	}

	return inds;
}
// �������������� ������ ��ġ�� ���� 0 ���� �����Ѵ�(�ᱹ�� �����ϴ� ��)
void del_arg( int argc, char **argv, int index )
{
	int ii;
	for ( ii=index; ii < argc-1; ++ii ) argv[ii] = argv[ii+1];

	argv[ii] = 0;
}
// �������������� ������ ���� ������ 1�� ������ 0�� ��ȯ�Ѵ�
int find_arg( int argc, char* argv[], char *arg )
{
	int ii;
	for ( ii=0; ii < argc; ++ii )
	{
		if ( !argv[ii] ) continue;

		if ( 0==strcmp( argv[ii], arg ) )
		{
			del_arg( argc, argv, ii );
			return 1;
		}
	}

	return 0;
}
// �������������� ������ ���� ������ �� �������� �ش��ϴ� �������� ��ȯ�Ѵ�
int find_int_arg( int argc, char **argv, char *arg, int def )
{
	int ii;
	for ( ii=0; ii < argc-1; ++ii )
	{
		if ( !argv[ii] ) continue;

		if ( 0==strcmp( argv[ii], arg ) )
		{
			def = atoi( argv[ii+1] );
			//�������� �������� �ش��ϴ� ���� �����Ѵ�
			del_arg( argc, argv, ii );
			del_arg( argc, argv, ii );
			break;
		}
	}

	return def;
}
// �������������� ������ ���� ������ �� �������� �ش��ϴ� �Ǽ����� ��ȯ�Ѵ�
float find_float_arg( int argc, char **argv, char *arg, float def )
{
	int ii;
	for ( ii=0; ii < argc-1; ++ii )
	{
		if ( !argv[ii] ) continue;

		if ( 0==strcmp( argv[ii], arg ) )
		{
			def = atof( argv[ii+1] );
			//�������� �������� �ش��ϴ� ���� �����Ѵ�
			del_arg( argc, argv, ii );
			del_arg( argc, argv, ii );
			break;
		}
	}
	return def;
}
// �������������� ������ ���� ������ �� �������� �ش��ϴ� ���ڿ����� ��ȯ�Ѵ�
char *find_char_arg( int argc, char **argv, char *arg, char *def )
{
	int ii;
	for ( ii=0; ii < argc-1; ++ii )
	{
		if ( !argv[ii] ) continue;

		if ( 0==strcmp( argv[ii], arg ) )
		{
			def = argv[ii+1];
			//�������� �������� �ش��ϴ� ���� �����Ѵ�
			del_arg( argc, argv, ii );
			del_arg( argc, argv, ii );
			break;
		}
	}

	return def;
}


char *basecfg( char *cfgfile )
{
	char *cc = cfgfile;
	char *next;

	while ( (next = strchr( cc, '/' )) )
	{
		cc = next+1;
	}

	cc = copy_string( cc );
	next = strchr( cc, '.' );

	if ( next ) *next = 0;

	return cc;
}

int alphanum_to_int( char cc )
{
	return (cc < 58) ? cc - 48 : cc-87;
}

char int_to_alphanum( int i )
{
	if ( i == 36 ) return '.';

	return (i < 10) ? i + 48 : i + 87;
}

void pm( int M, int N, float *A )
{
	int ii, jj;
	for ( ii=0; ii < M; ++ii )
	{
		printf( "%d ", ii+1 );

		for ( jj=0; jj < N; ++jj )
		{
			printf( "%2.4f, ", A[ii*N+jj] );
		}
		printf( "\n" );
	}

	printf( "\n" );
}
// str ���� orig �� ã�Ƽ�
void find_replace( char *str, char *orig, char *rep, char *output )
{
	char buffer[4096] = { 0 };
	char *p;

	//sprintf( buffer, "%s", str );					//  [6/27/2018 jobs]
	sprintf_s( buffer, strlen(str), "%s", str );	//  [6/27/2018 jobs]
	// buffer ���� orig �� ���ڿ���  ã�´�
	if ( !(p = strstr( buffer, orig )) )
	{	// Is 'orig' even in 'str'?
		// orig �� ���ڿ��� ��ã�Ҵ�, output �� ���ڿ��� str �� ��ü�Ѵ�
		//sprintf( output, "%s", str );					//  [6/27/2018 jobs]
		sprintf_s( output, strlen(str), "%s", str );	//  [6/27/2018 jobs]
		return;
	}

	*p = '\0';
	// output �� ���ڿ��� ã�Ҵ�, output �� str + rep + orig �� ��ü�Ѵ�
	//sprintf( output, "%s%s%s", buffer, rep, p+strlen( orig ) );	//  [6/27/2018 jobs]
	sprintf_s( output												//  [6/27/2018 jobs]
			, strlen(buffer)+strlen(rep)+strlen(p)+strlen(orig)
			, "%s%s%s"
			, buffer
			, rep
			, p+strlen( orig ) );									
}

float sec( clock_t clocks )
{
	return (float)clocks/CLOCKS_PER_SEC;
}

void top_k( float *aa, int nn, int kk, int *index )
{
	int ii, jj;
	for ( jj=0; jj < kk; ++jj ) index[jj] = -1;

	for ( ii=0; ii < nn; ++ii )
	{
		int curr = ii;
		for ( jj=0; jj < kk; ++jj )
		{
			if ( (index[jj] < 0) || aa[curr] > aa[index[jj]] )
			{
				int swap	= curr;
				curr		= index[jj];
				index[jj]	= swap;
			}
		}
	}
}

void error( const char *s )
{
	perror( s );
	assert( 0 );
	exit( -1 );
}

unsigned char *read_file( char *filename )
{
	//FILE *fp = fopen( filename, "rb" );		//  [6/27/2018 jobs]
	FILE *fp; fopen_s( &fp, filename, "rb" );	//  [6/27/2018 jobs]

	size_t size;

	fseek( fp, 0, SEEK_END );
	size = ftell( fp );
	fseek( fp, 0, SEEK_SET );

	unsigned char *text = calloc( size+1, sizeof( char ) );
	fread( text, 1, size, fp );
	fclose( fp );
	return text;
}

void malloc_error()
{
	//fprintf( stderr, "Malloc error\n" );	//  [7/7/2018 jobs]
	fprintf( stderr, "�޸��Ҵ� ����!\n" );	//  [7/7/2018 jobs]
	exit( -1 );
}

void file_error( char *ss )
{
	//fprintf( stderr, "Couldn't open file: %s\n", ss );	//  [7/7/2018 jobs]
	fprintf( stderr, "���Ͽ��� ������!: %s\n", ss );
	exit( 0 );
}

list *split_str( char *ss, char delim )
{
	size_t ii;
	size_t len	= strlen( ss );
	list *ll	= make_list();
	list_insert( ll, ss );

	for ( ii=0; ii < len; ++ii )
	{
		if ( ss[ii] == delim )
		{
			ss[ii] = '\0';
			list_insert( ll, &(ss[ii+1]) );
		}
	}

	return ll;
}

void strip( char *ss )
{
	size_t ii;
	size_t len = strlen( ss );
	size_t offset = 0;

	for ( ii=0; ii < len; ++ii )
	{
		char cc = ss[ii];
		if ( cc==' '||cc=='\t'||cc=='\n' ) ++offset;
		else ss[ii-offset] = cc;
	}

	ss[len-offset] = '\0';
}

void strip_char( char *ss, char bad )
{
	size_t ii;
	size_t len = strlen( ss );
	size_t offset = 0;

	for ( ii=0; ii < len; ++ii )
	{
		char cc = ss[ii];
		if ( cc==bad ) ++offset;
		else ss[ii-offset] = cc;
	}

	ss[len-offset] = '\0';
}

void free_ptrs( void **ptrs, int nn )
{
	int ii;
	for ( ii=0; ii < nn; ++ii ) free( ptrs[ii] );

	free( ptrs );
}
// ���Ͽ��� ���ٿ� �ش��ϴ� ���ڿ��� �о ��ȯ�Ѵ�(���ٴ�����)
char *fgetl( FILE *fp )
{
	if ( feof( fp ) ) return 0;	// ������ ���̸� 0�� ��ȯ

	size_t size = 512;
	char *line = malloc( size*sizeof( char ) );

	// �ٴ����� line ���ۿ� ���ڿ��� �����´�
	if ( !fgets( line, (int)size, fp ) )
	{	// ���ϳ��̶� ���۸� ���� 0�� ��ȯ�Ѵ�
		free( line );
		return 0;
	}

	size_t curr = strlen( line );

	// ���ۿ� ������ ���� ���� �о����� Ȯ���ϰ�
	// �ٹٲ��� �ƴϰ� ������ ���� �ƴϸ� �ݺ��Ѵ�
	while ( (line[curr-1] != '\n') && !feof( fp ) )
	{
		if ( curr == size-1 )
		{
			// ������ ũ�⸦ �ø���, ���Ҵ� �Ѵ�
			size *= 2;
			// ������ �����ϴ� ������ ���� �״�� �����ϰ� ������ ũ�⸸ �ø���
			line = realloc( line, size*sizeof( char ) );

			if ( !line )
			{
				printf( "%ld\n", size );
				malloc_error();
			}
		}

		size_t readsize = size-curr;	// �߰��� �о�� ũ��

		if ( readsize > INT_MAX ) readsize = INT_MAX-1;

		// ���ٿ� �о���� ���� ������ ���� �о�´�
		fgets( &line[curr], (int)readsize, fp );
		curr = strlen( line );
	}

	if ( line[curr-1] == '\n' ) line[curr-1] = '\0';

	return line;
}

int read_int( int fd )
{
	int n = 0;
	//int next = read( fd, &n, sizeof( int ) );	//  [6/27/2018 jobs]
	int next = _read( fd, &n, sizeof( int ) );	//  [6/27/2018 jobs]

	if ( next <= 0 ) return -1;
	return n;
}

void write_int( int fd, int nn )
{
	//int next = write( fd, &n, sizeof( int ) );	//  [6/27/2018 jobs]
	int next = _write( fd, &nn, sizeof( int ) );		//  [6/27/2018 jobs]

	//if ( next <= 0 ) error( "read failed" );	//  [7/7/2018 jobs]
	if ( next <= 0 ) error( "�б� ����!" );	//  [7/7/2018 jobs]
}

int read_all_fail( int fd, char *buffer, size_t bytes )
{
	size_t nn = 0;
	while ( nn < bytes )
	{
		//int next = read( fd, buffer + n, bytes-n );	//  [6/27/2018 jobs]
		int next = _read( fd, buffer + nn, (unsigned int)( bytes-nn ) );	//  [6/27/2018 jobs]

		if ( next <= 0 ) return 1;
		nn += next;
	}
	return 0;
}

int write_all_fail( int fd, char *buffer, size_t bytes )
{
	size_t nn = 0;
	while ( nn < bytes )
	{
		//size_t next = write( fd, buffer + n, bytes-n );	//  [6/27/2018 jobs]
		size_t next = _write( fd, buffer + nn, (unsigned int)( bytes-nn ) );	//  [6/27/2018 jobs]

		if ( next <= 0 ) return 1;
		nn += next;
	}
	return 0;
}

void read_all( int fd, char *buffer, size_t bytes )
{
	size_t nn = 0;
	while ( nn < bytes )
	{
		//int next = read( fd, buffer + n, bytes-n );	//  [6/27/2018 jobs]
		int next = _read( fd, buffer + nn, (unsigned int)( bytes-nn ) );	//  [6/27/2018 jobs]

		//if ( next <= 0 ) error( "read failed" );	//  [7/7/2018 jobs]
		if ( next <= 0 ) error( "�б� ����!" );	//  [7/7/2018 jobs]
		nn += next;
	}
}

void write_all( int fd, char *buffer, size_t bytes )
{
	size_t nn = 0;
	while ( nn < bytes )
	{
		//size_t next = write( fd, buffer + n, bytes-n );	//  [6/27/2018 jobs]
		size_t next = _write( fd, buffer + nn, (unsigned int)( bytes-nn ) );	//  [6/27/2018 jobs]

		//if ( next <= 0 ) error( "write failed" );	//  [7/7/2018 jobs]
		if ( next <= 0 ) error( "���� ����!" );	//  [7/7/2018 jobs]
		nn += next;
	}
}


char *copy_string( char *ss )
{
	char *copy = malloc( strlen( ss )+1 );
	//strncpy( copy, s, strlen( s )+1 );				//  [6/27/2018 jobs]
	strncpy_s( copy, strlen(copy), ss, strlen( ss )+1 );	//  [6/27/2018 jobs]
	return copy;
}

list *parse_csv_line( char *line )
{
	list *ll = make_list();
	char *cc, *pp;
	int in = 0;

	for ( cc=line, pp = line; *cc != '\0'; ++cc )
	{
		if ( *cc == '"' ) in = !in;
		else if ( *cc == ',' && !in )
		{
			*cc = '\0';
			list_insert( ll, copy_string( pp ) );
			pp = cc+1;
		}
	}

	list_insert( ll, copy_string( pp ) );
	return ll;
}

int count_fields( char *line )
{
	int count = 0;
	int done = 0;
	char *cc;

	for ( cc = line; !done; ++cc )
	{
		done = (*cc == '\0');
		if ( *cc == ',' || done ) ++count;
	}

	return count;
}

float *parse_fields( char *line, int nn )
{
	float *field = calloc( nn, sizeof( float ) );
	char *cc, *pp, *end;
	int count = 0;
	int done = 0;

	for ( cc = line, pp = line; !done; ++cc )
	{
		done = (*cc == '\0');

		if ( *cc == ',' || done )
		{
			*cc = '\0';
			field[count] = (float)strtod( pp, &end );

			if ( pp == cc )
				field[count] = (float)nan( "" );

			if ( end != cc && (end != cc-1 || *end != '\r') )
				field[count] = (float)nan( "" ); //DOS file formats!

			pp = cc+1;
			++count;
		}
	}

	return field;
}

float sum_array( float *an, int nn )
{
	int ii;
	float sum = 0;

	for ( ii=0; ii < nn; ++ii ) sum += an[ii];

	return sum;
}

float mean_array( float *aa, int nn )
{
	return sum_array( aa, nn )/nn;
}

void mean_arrays( float **aa, int nn, int els, float *avg )
{
	int ii;
	int jj;
	memset( avg, 0, els*sizeof( float ) );

	for ( jj=0; jj < nn; ++jj )
	{
		for ( ii=0; ii < els; ++ii )
		{
			avg[ii] += aa[jj][ii];
		}
	}

	for ( ii=0; ii < els; ++ii )
	{
		avg[ii] /= nn;
	}
}

void print_statistics( float *aa, int nn )
{
	float mm = mean_array( aa, nn );
	float vv = variance_array( aa, nn );

	printf( "MSE: %.6f, Mean: %.6f, Variance: %.6f\n", mse_array( aa, nn ), mm, vv );
}

float variance_array( float *aa, int nn )
{
	int ii;
	float sum = 0;
	float mean = mean_array( aa, nn );

	for ( ii=0; ii < nn; ++ii )
		sum += (aa[ii] - mean) * (aa[ii] - mean);

	float variance = sum/nn;

	return variance;
}

int constrain_int( int a, int min, int max )
{
	if ( a < min ) return min;
	if ( a > max ) return max;
	return a;
}

float constrain( float min, float max, float aa )
{
	if ( aa < min ) return min;
	if ( aa > max ) return max;

	return aa;
}

float dist_array( float *aa, float *bb, int nn, int sub )
{
	int ii;
	float sum = 0;

	for ( ii = 0; ii < nn; ii += sub )
		sum += (float)pow( aa[ii]-bb[ii], 2 );

	return (float)sqrt( sum );
}

float mse_array( float *aa, int nn )
{
	int ii;
	float sum = 0;

	for ( ii=0; ii < nn; ++ii ) sum += aa[ii]*aa[ii];

	return (float)sqrt( sum/nn );
}

void normalize_array( float *aa, int nn )
{
	int ii;
	float mu = mean_array( aa, nn );
	float sigma = (float)sqrt( variance_array( aa, nn ) );

	for ( ii=0; ii < nn; ++ii )
	{
		aa[ii] = (aa[ii] - mu)/sigma;
	}

	mu = mean_array( aa, nn );
	sigma = (float)sqrt( variance_array( aa, nn ) );
}

void translate_array( float *aa, int nn, float ss )
{
	int ii;
	for ( ii=0; ii < nn; ++ii )
	{
		aa[ii] += ss;
	}
}

float mag_array( float *aa, int nn )
{
	int ii;
	float sum = 0;

	for ( ii=0; ii < nn; ++ii )
	{
		sum += aa[ii]*aa[ii];
	}

	return (float)sqrt( sum );
}

void scale_array( float *aa, int nn, float ss )
{
	int ii;
	for ( ii=0; ii < nn; ++ii )
	{
		aa[ii] *= ss;
	}
}

int sample_array( float *aa, int nn )
{
	float sum = sum_array( aa, nn );
	scale_array( aa, nn, 1.0f/sum );
	float rr = rand_uniform( 0.0f, 1.0f );

	int ii;
	for ( ii = 0; ii < nn; ++ii )
	{
		rr = rr - aa[ii];

		if ( rr <= 0 ) return ii;
	}

	return nn-1;
}

int max_int_index( int *aa, int nn )
{
	if ( nn <= 0 ) return -1;

	int ii, max_i = 0;
	int max = aa[0];

	for ( ii=1; ii < nn; ++ii )
	{
		if ( aa[ii] > max )
		{
			max = aa[ii];
			max_i = ii;
		}
	}

	return max_i;
}
// ���� ū ���� ��ġ�� ��ȯ�Ѵ�
int max_index( float *aa, int nn )
{
	if ( nn <= 0 ) return -1;

	int ii, max_i = 0;
	float max = aa[0];

	for ( ii = 1; ii < nn; ++ii )
	{
		if ( aa[ii] > max )
		{
			max = aa[ii];
			max_i = ii;
		}
	}

	return max_i;
}

int int_index( int *aa, int val, int nn )
{
	int ii;
	for ( ii=0; ii < nn; ++ii )
	{
		if ( aa[ii] == val ) return ii;
	}

	return -1;
}

int rand_int( int min, int max )
{
	if ( max < min )
	{
		int ss = min;
		min = max;
		max = ss;
	}

	int rr = (rand()%(max - min + 1)) + min;

	return rr;
}

// From http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
float rand_normal()
{
	static int haveSpare = 0;
	static double rand1, rand2;

	if ( haveSpare )
	{
		haveSpare = 0;
		return (float)( sqrt( rand1 ) * sin( rand2 ) );
	}

	haveSpare = 1;

	rand1 = rand() / ((double)RAND_MAX);

	if ( rand1 < 1e-100 ) rand1 = 1e-100;

	rand1 = -2 * log( rand1 );
	rand2 = (rand() / ((double)RAND_MAX)) * TWO_PI;

	return (float)( sqrt( rand1 ) * cos( rand2 ) );
}

/*
float rand_normal()
{
	int n = 12;
	int i;
	float sum= 0;

	for ( i = 0; i < n; ++i )
		sum += (float)rand()/RAND_MAX;

	return sum-n/2.;
}
*/

size_t rand_size_t()
{
	return
		((size_t)(rand()&0xff) << 56) |
		((size_t)(rand()&0xff) << 48) |
		((size_t)(rand()&0xff) << 40) |
		((size_t)(rand()&0xff) << 32) |
		((size_t)(rand()&0xff) << 24) |
		((size_t)(rand()&0xff) << 16) |
		((size_t)(rand()&0xff) <<  8) |
		((size_t)(rand()&0xff) <<  0);
}
// �������� ū�� ���̷� �Ѹ����� ��ȯ�Ѵ�
float rand_uniform( float min, float max )
{
	if ( max < min )
	{
		float swap = min;
		min = max;
		max = swap;
	}

	return ( (float)rand()/RAND_MAX * (max - min) ) + min;
}

float rand_scale( float ss )
{
	float scale = rand_uniform( 1, ss );

	if ( rand()%2 ) return scale;

	return 1.0f/scale;
}
// �ѻ��� �迭�� nn ���� ���·� ��ȣȭ�� kk ���� �迭�� �����
float **one_hot_encode( float *aa, int nn, int kk )
{
	int ii;
	float **tt = calloc( nn, sizeof( float* ) );

	for ( ii=0; ii < nn; ++ii )
	{
		tt[ii] = calloc( kk, sizeof( float ) );

		int index = (int)aa[ii];

		tt[ii][index] = 1;
	}

	return tt;
}

