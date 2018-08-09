// DarkNet_CPP.cpp : 콘솔 응용 프로그램에 대한 진입점을 정의합니다.
//

#include "stdafx.h"
#include "darknet.h"

//#include <time.h>
//#include <stdlib.h>
//#include <stdio.h>

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

/*		//  [7/15/2018 jobs]
extern void predict_classifier( char *datacfg
						, char *cfgfile
						, char *weightfile
						, char *filename
						, int top );
extern void test_detector( char *datacfg
						, char *cfgfile
						, char *weightfile
						, char *filename
						, float thresh
						, float hier_thresh
						, char *outfile
						, int fullscreen );
extern void run_yolo( int argc, char **argv );
extern void run_detector( int argc, char **argv );
extern void run_coco( int argc, char **argv );
extern void run_captcha( int argc, char **argv );
extern void run_nightmare( int argc, char **argv );
extern void run_classifier( int argc, char **argv );
extern void run_regressor( int argc, char **argv );
extern void run_segmenter( int argc, char **argv );
extern void run_char_rnn( int argc, char **argv );
extern void run_tag( int argc, char **argv );
extern void run_cifar( int argc, char **argv );
extern void run_go( int argc, char **argv );
extern void run_art( int argc, char **argv );
extern void run_super( int argc, char **argv );
extern void run_lsd( int argc, char **argv );
*/

// 선정한 모든 파일의 신경망 가중값을 더하고 평균하여 새로운 신경망 가중값을 생성한다
void average( int argc, char *argv[] )
{
	char *cfgfile = argv[2];	// 신경망 설정값 파일명
	char *outfile = argv[3];	// 신경망 가중값을 저장할 파일명

	gpu_index = -1;

	network *net = parse_network_cfg( cfgfile );	// 
	network *sum = parse_network_cfg( cfgfile );	// 

	char *weightfile = argv[4];	// 탑재할 신경망 가중값 파일명

	load_weights( sum, weightfile );

	// 여러개의 신경망 가중값을 각각 더한다
	int i, j;
	int MangGaeSu = argc - 5;	// 평균할 신경망 파일 개수
	for ( i=0; i<MangGaeSu; ++i )
	{
		weightfile = argv[i+5];
		load_weights( net, weightfile );

		for ( j=0; j<net->n; ++j )
		{
			layer l = net->layers[j];
			layer out = sum->layers[j];

			if ( l.type == CONVOLUTIONAL )
			{
				int num = l.n*l.c*l.size*l.size;
				axpy_cpu( l.n, 1.0f, l.biases, 1, out.biases, 1 );		// 편향값을 더한다
				axpy_cpu( num, 1.0f, l.weights, 1, out.weights, 1 );	// 가중값을 더한다

				if ( l.batch_normalize )
				{
					axpy_cpu( l.n, 1.0f, l.scales, 1, out.scales, 1 );
					axpy_cpu( l.n, 1.0f, l.rolling_mean, 1, out.rolling_mean, 1 );
					axpy_cpu( l.n, 1.0f, l.rolling_variance, 1, out.rolling_variance, 1 );
				}
			}

			if ( l.type == CONNECTED )
			{
				axpy_cpu( l.outputs, 1.0f, l.biases, 1, out.biases, 1 );
				axpy_cpu( l.outputs*l.inputs, 1.0f, l.weights, 1, out.weights, 1 );
			}
		}
	}

	// 더해진 여러개의 신경망 가중값을 각각 평균을 계산하여 새로운 신경망 가중값을 만든다
	// 기본 신경망 + 추가한 신경망 개수
	MangGaeSu = MangGaeSu+1;

	float BiYul	= 1.0f / MangGaeSu;	// 적용할 가중값비율 계산

	for ( j=0; j<net->n; ++j )		// 신경망층을 반복한다
	{
		layer l = sum->layers[j];

		if ( l.type == CONVOLUTIONAL )
		{
			int num = l.n*l.c*l.size*l.size;
			scal_cpu( l.n, BiYul, l.biases, 1 );
			scal_cpu( num, BiYul, l.weights, 1 );

			if ( l.batch_normalize )
			{
				scal_cpu( l.n, BiYul, l.scales, 1 );
				scal_cpu( l.n, BiYul, l.rolling_mean, 1 );
				scal_cpu( l.n, BiYul, l.rolling_variance, 1 );
			}
		}

		if ( l.type == CONNECTED )
		{
			scal_cpu( l.outputs, BiYul, l.biases, 1 );
			scal_cpu( l.outputs*l.inputs, BiYul, l.weights, 1 );
		}
	}

	// 새로운 신경망 가중값을 파일로 저장한다
	save_weights( sum, outfile );
}

long numops( network *net )
{
	int i;
	long ops = 0;
	for ( i = 0; i < net->n; ++i )
	{
		layer l = net->layers[i];
		if ( l.type == CONVOLUTIONAL )
		{
			ops += 2l * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w;
		}
		else if ( l.type == CONNECTED )
		{
			ops += 2l * l.inputs * l.outputs;
		}
		else if ( l.type == RNN )
		{
			ops += 2l * l.input_layer->inputs * l.input_layer->outputs;
			ops += 2l * l.self_layer->inputs * l.self_layer->outputs;
			ops += 2l * l.output_layer->inputs * l.output_layer->outputs;
		}
		else if ( l.type == GRU )
		{
			ops += 2l * l.uz->inputs * l.uz->outputs;
			ops += 2l * l.uh->inputs * l.uh->outputs;
			ops += 2l * l.ur->inputs * l.ur->outputs;
			ops += 2l * l.wz->inputs * l.wz->outputs;
			ops += 2l * l.wh->inputs * l.wh->outputs;
			ops += 2l * l.wr->inputs * l.wr->outputs;
		}
		else if ( l.type == LSTM )
		{
			ops += 2l * l.uf->inputs * l.uf->outputs;
			ops += 2l * l.ui->inputs * l.ui->outputs;
			ops += 2l * l.ug->inputs * l.ug->outputs;
			ops += 2l * l.uo->inputs * l.uo->outputs;
			ops += 2l * l.wf->inputs * l.wf->outputs;
			ops += 2l * l.wi->inputs * l.wi->outputs;
			ops += 2l * l.wg->inputs * l.wg->outputs;
			ops += 2l * l.wo->inputs * l.wo->outputs;
		}
	}
	return ops;
}

// 신경망 계산속도 평가
void speed( char *cfgfile, int HoiSu )
{
	if ( HoiSu==0 )
		HoiSu = 1000;
	// 신경망 설정파일로 신경망을 생성한다
	network *net = parse_network_cfg( cfgfile );

	set_batch_network( net, 1 );

	int i;
	//time_t start = time( 0 );
	double start = what_time_is_it_now();
	image im = make_image( net->w, net->h, net->c*net->batch );

	// 회수를 반복하여 신경망 출력을 계산한다
	for ( i=0; i<HoiSu; ++i )
	{
		network_predict( net, im.data );
	}

	//double JiNanSiGan = difftime( time( 0 ), start );
	double JiNanSiGan = what_time_is_it_now() - start;

	long ops = numops( net );

	printf( "\n%d 회 평가에, %f 초 걸림\n", HoiSu, JiNanSiGan );
	printf( "부동소수연산: %.2f Bn\n", (float)ops/1000000000.0 );
	printf( "FLOPS: %.2f Bn\n", (float)ops/1000000000.0 * HoiSu/JiNanSiGan );
	printf( "1회 평가속도: %f 초(sec/평가회수)\n", JiNanSiGan/HoiSu );
	printf( "속도: %f Hz\n", HoiSu/JiNanSiGan );
}

void operations( char *cfgfile )
{
	gpu_index = -1;
	network *net = parse_network_cfg( cfgfile );

	long ops = numops(net);

	printf( "부동소수연산: %ld\n", ops );
	printf( "부동소수연산: %.2f Bn\n", (float)ops/1000000000.0 );
}

void oneoff( char *cfgfile, char *weightfile, char *outfile )
{
	gpu_index = -1;
	network *net = parse_network_cfg( cfgfile );

	int oldn = net->layers[net->n - 2].n;
	int c = net->layers[net->n - 2].c;

	scal_cpu( oldn*c, .1, net->layers[net->n - 2].weights, 1 );
	scal_cpu( oldn, 0, net->layers[net->n - 2].biases, 1 );

	net->layers[net->n - 2].n = 11921;
	net->layers[net->n - 2].biases += 5;
	net->layers[net->n - 2].weights += 5*c;

	if ( weightfile )
	{
		load_weights( net, weightfile );
	}

	net->layers[net->n - 2].biases -= 5;
	net->layers[net->n - 2].weights -= 5*c;
	net->layers[net->n - 2].n = oldn;
	printf( "%d\n", oldn );

	layer l = net->layers[net->n - 2];
	copy_cpu( l.n/3, l.biases, 1, l.biases +   l.n/3, 1 );
	copy_cpu( l.n/3, l.biases, 1, l.biases + 2*l.n/3, 1 );
	copy_cpu( l.n/3*l.c, l.weights, 1, l.weights +   l.n/3*l.c, 1 );
	copy_cpu( l.n/3*l.c, l.weights, 1, l.weights + 2*l.n/3*l.c, 1 );

	*net->seen = 0;

	save_weights( net, outfile );
}

void oneoff2( char *cfgfile, char *weightfile, char *outfile, int l )
{
	gpu_index = -1;
	network *net = parse_network_cfg( cfgfile );
	if ( weightfile )
	{
		load_weights_upto( net, weightfile, 0, net->n );
		load_weights_upto( net, weightfile, l, net->n );
	}
	*net->seen = 0;
	save_weights_upto( net, outfile, net->n );
}

void partial( char *cfgfile, char *weightfile, char *outfile, int max )
{
	gpu_index = -1;

	network *net = load_network( cfgfile, weightfile, 1 );

	save_weights_upto( net, outfile, max );
}

void print_weights( char *cfgfile, char *weightfile, int n )
{
	gpu_index = -1;
	network *net = load_network( cfgfile, weightfile, 1 );
	layer l = net->layers[n];

	int i, j;
	//printf("[");
	for ( i = 0; i < l.n; ++i )
	{
		//printf("[");
		for ( j = 0; j < l.size*l.size*l.c; ++j )
		{
			//if(j > 0) printf(",");
			printf( "%g ", l.weights[i*l.size*l.size*l.c + j] );
		}
		printf( "\n" );
		//printf("]%s\n", (i == l.n-1)?"":",");
	}
	//printf("]");
}

void rescale_net( char *cfgfile, char *weightfile, char *outfile )
{
	gpu_index = -1;
	network *net = load_network( cfgfile, weightfile, 0 );

	int i;
	for ( i=0; i<net->n; ++i )
	{
		layer l = net->layers[i];

		if ( l.type==CONVOLUTIONAL )
		{
			rescale_weights( l, 2.0f, -0.5f );
			break;
		}
	}

	save_weights( net, outfile );
}

void rgbgr_net( char *cfgfile, char *weightfile, char *outfile )
{
	gpu_index = -1;
	network *net = load_network(cfgfile, weightfile, 0);

	int i;
	for ( i=0; i<net->n; ++i )
	{
		layer l = net->layers[i];

		if ( l.type==CONVOLUTIONAL )
		{
			rgbgr_weights( l );
			break;
		}
	}

	save_weights( net, outfile );
}

void reset_normalize_net( char *cfgfile, char *weightfile, char *outfile )
{
	gpu_index = -1;
	network *net = load_network(cfgfile, weightfile, 0);

	int i;
	for ( i=0; i<net->n; ++i )
	{
		layer l = net->layers[i];

		if ( l.type == CONVOLUTIONAL && l.batch_normalize )
		{
			denormalize_convolutional_layer( l );
		}

		if ( l.type == CONNECTED && l.batch_normalize )
		{
			denormalize_connected_layer( l );
		}

		if ( l.type == GRU && l.batch_normalize )
		{
			denormalize_connected_layer( *l.input_z_layer );
			denormalize_connected_layer( *l.input_r_layer );
			denormalize_connected_layer( *l.input_h_layer );
			denormalize_connected_layer( *l.state_z_layer );
			denormalize_connected_layer( *l.state_r_layer );
			denormalize_connected_layer( *l.state_h_layer );
		}
	}

	save_weights( net, outfile );
}

layer normalize_layer( layer l, int n )
{
	int j;
	l.batch_normalize=1;
	//l.scales = calloc( n, sizeof( float ) );
	l.scales = (float *)calloc( n, sizeof( float ) );

	for ( j = 0; j < n; ++j )
	{
		l.scales[j] = 1;
	}

	//l.rolling_mean		= calloc( n, sizeof( float ) );
	//l.rolling_variance	= calloc( n, sizeof( float ) );
	l.rolling_mean		= (float *)calloc( n, sizeof( float ) );
	l.rolling_variance	= (float *)calloc( n, sizeof( float ) );

	return l;
}

void normalize_net( char *cfgfile, char *weightfile, char *outfile )
{
	gpu_index = -1;
	network *net = load_network(cfgfile, weightfile, 0);

	int i;
	for ( i = 0; i < net->n; ++i )
	{
		layer l = net->layers[i];
		if ( l.type == CONVOLUTIONAL && !l.batch_normalize )
		{
			net->layers[i] = normalize_layer( l, l.n );
		}
		if ( l.type == CONNECTED && !l.batch_normalize )
		{
			net->layers[i] = normalize_layer( l, l.outputs );
		}
		if ( l.type == GRU && l.batch_normalize )
		{
			*l.input_z_layer = normalize_layer( *l.input_z_layer, l.input_z_layer->outputs );
			*l.input_r_layer = normalize_layer( *l.input_r_layer, l.input_r_layer->outputs );
			*l.input_h_layer = normalize_layer( *l.input_h_layer, l.input_h_layer->outputs );
			*l.state_z_layer = normalize_layer( *l.state_z_layer, l.state_z_layer->outputs );
			*l.state_r_layer = normalize_layer( *l.state_r_layer, l.state_r_layer->outputs );
			*l.state_h_layer = normalize_layer( *l.state_h_layer, l.state_h_layer->outputs );
			net->layers[i].batch_normalize=1;
		}
	}
	save_weights( net, outfile );
}

void statistics_net( char *cfgfile, char *weightfile )
{
	gpu_index = -1;
	network *net = load_network( cfgfile, weightfile, 0 );

	int i;

	for ( i = 0; i < net->n; ++i )
	{
		layer l = net->layers[i];
		if ( l.type == CONNECTED && l.batch_normalize )
		{
			printf( "Connected Layer %d\n", i );
			statistics_connected_layer( l );
		}
		if ( l.type == GRU && l.batch_normalize )
		{
			printf( "GRU Layer %d\n", i );
			printf( "Input Z\n" );
			statistics_connected_layer( *l.input_z_layer );
			printf( "Input R\n" );
			statistics_connected_layer( *l.input_r_layer );
			printf( "Input H\n" );
			statistics_connected_layer( *l.input_h_layer );
			printf( "State Z\n" );
			statistics_connected_layer( *l.state_z_layer );
			printf( "State R\n" );
			statistics_connected_layer( *l.state_r_layer );
			printf( "State H\n" );
			statistics_connected_layer( *l.state_h_layer );
		}
		printf( "\n" );
	}
}

void denormalize_net( char *cfgfile, char *weightfile, char *outfile )
{
	gpu_index = -1;
	network *net = load_network( cfgfile, weightfile, 0 );

	int ii;
	for ( ii=0; ii < net->n; ++ii )
	{
		layer l = net->layers[ii];
		if ( (l.type == DECONVOLUTIONAL || l.type == CONVOLUTIONAL) && l.batch_normalize )
		{
			denormalize_convolutional_layer( l );
			net->layers[ii].batch_normalize = 0;
		}

		if ( l.type == CONNECTED && l.batch_normalize )
		{
			denormalize_connected_layer( l );
			net->layers[ii].batch_normalize = 0;
		}

		if ( l.type == GRU && l.batch_normalize )
		{
			denormalize_connected_layer( *l.input_z_layer );
			denormalize_connected_layer( *l.input_r_layer );
			denormalize_connected_layer( *l.input_h_layer );
			denormalize_connected_layer( *l.state_z_layer );
			denormalize_connected_layer( *l.state_r_layer );
			denormalize_connected_layer( *l.state_h_layer );
			l.input_z_layer->batch_normalize = 0;
			l.input_r_layer->batch_normalize = 0;
			l.input_h_layer->batch_normalize = 0;
			l.state_z_layer->batch_normalize = 0;
			l.state_r_layer->batch_normalize = 0;
			l.state_h_layer->batch_normalize = 0;
			net->layers[ii].batch_normalize=0;
		}
	}
	save_weights( net, outfile );
}
// 이미지 만들기
void mkimg( char *cfgfile, char *weightfile, int h, int w, int num, char *prefix )
{
	network *net	= load_network( cfgfile, weightfile, 0 );
	image *ims		= pull_convolutional_weights( net->layers[0] );

	int nn = net->layers[0].n;
	int z;

	for ( z = 0; z < num; ++z )
	{
		image img = make_image( h, w, 3 );
		fill_image( img, 0.5f );

		int i;
		for ( i = 0; i < 100; ++i )
		{
			image r = copy_image( ims[rand()%nn] );
			rotate_image_cw( r, rand()%4 );
			random_distort_image( r, 1, 1.5f, 1.5f );
			int dx = rand()%(w-r.w);
			int dy = rand()%(h-r.h);
			ghost_image( r, img, dx, dy );
			free_image( r );
		}

		char buff[256];
		//sprintf( buff, "%s/gen_%d", prefix, z );
		sprintf_s( buff, "%s/gen_%d", prefix, z );
		save_image( img, buff );
		free_image( img );
	}
}

/*		//  [7/15/2018 jobs]
// 신경망 가중값을 시각화하여 보여줌
void visualize( char *cfgfile, char *weightfile )
{
	network *net = load_network( cfgfile, weightfile, 0 );

	visualize_network( net );

#ifdef OPENCV
	cvWaitKey( 0 );
#endif
}
*/

// argc : Argument Count	=> 결정고유값 개수
// argv : Argument Vector	=> 결정고유값 배열
//int main()
int main( int argc, char **argv )
{
	// 응용프로그램 실행 폴더위치 출력
	//char *App_Home_Folder;
	//App_Home_Folder = getcwd( NULL, 255 );
	char App_Home_Folder[255];
	int Su = 0;

	char *BanHwan;
	BanHwan = _getcwd( App_Home_Folder, 255 );
	//_getcwd( App_Home_Folder, 255 );
	printf_s( "수: %d, 응용프로그램의 현재 폴더위치: %s\n", Su, App_Home_Folder );

	//chdir( "x64" );
	//_chdir( "x64/Debug" );
	//BanHwan = _getcwd( App_Home_Folder, 255 );
	//_getcwd( App_Home_Folder, 255 );
	//printf_s( "수: %d, 응용프로그램의 현재 폴더위치: %s\n", Su, App_Home_Folder );

	// 응용프로그램 실행 방법
	if ( argc < 2 )
	{
		//fprintf( stderr, "usage: %s <function>\n", argv[0] );
		fprintf( stderr, "사용방법: %s <기능>\n", argv[0] );
		return 0;
	}

	gpu_index = find_int_arg( argc, argv, "-i", 0 );

	// GPU를 사용하지 않는다는 결정이 있는지 확인한다
	if ( find_arg( argc, argv, "-nogpu" ) )
	{
		gpu_index = -1;
	}

#ifndef GPU
	gpu_index = -1;
#else
	if ( gpu_index >= 0 )
	{
		cuda_set_device( gpu_index );
	}
#endif

	if		( 0 == strcmp( argv[1], "average" ) )	{	average( argc, argv );		}
	else if ( 0 == strcmp( argv[1], "yolo" ) )		{	run_yolo( argc, argv );		}
	else if ( 0 == strcmp( argv[1], "voxel" ) )		{	run_voxel( argc, argv );	}
	else if ( 0 == strcmp( argv[1], "super" ) )		{	run_super( argc, argv );	}
	else if ( 0 == strcmp( argv[1], "lsd" ) )		{	run_lsd( argc, argv );		}
	else if ( 0 == strcmp( argv[1], "detector" ) )	{	run_detector( argc, argv );	}
	else if ( 0 == strcmp( argv[1], "detect" ) )	{	run_detector( argc, argv );	}
/*	{
		float	thresh		= find_float_arg( argc, argv, "-thresh", 0.5f );
		char	*filename	= (argc > 4) ? argv[4] : 0;
		char	*outfile	= find_char_arg( argc, argv, "-out", 0 );
		int		fullscreen	= find_arg( argc, argv, "-fullscreen" );
		test_detector( "cfg/coco.data", argv[2], argv[3], filename, thresh, 0.5f, outfile, fullscreen );
	}
*/	else if ( 0 == strcmp( argv[1], "cifar" ) )		{	run_cifar( argc, argv );	}
	else if ( 0 == strcmp( argv[1], "go" ) )		{	run_go( argc, argv );		}
	else if ( 0 == strcmp( argv[1], "rnn" ) )		{	run_char_rnn( argc, argv );	}
	else if ( 0 == strcmp( argv[1], "coco" ) )		{	run_coco( argc, argv );		}
	else if ( 0 == strcmp( argv[1], "classify" ) )	{	predict_classifier( "cfg/imagenet1k.data", argv[2], argv[3], argv[4], 5 );	}
	else if ( 0 == strcmp( argv[1], "classifier" ) ){	run_classifier( argc, argv );	}
	else if ( 0 == strcmp( argv[1], "regressor" ) )	{	run_regressor( argc, argv );	}
	else if ( 0 == strcmp( argv[1], "segmenter" ) )	{	run_segmenter( argc, argv );	}
	else if ( 0 == strcmp( argv[1], "art" ) )		{	run_art( argc, argv );	}
	else if ( 0 == strcmp( argv[1], "tag" ) )		{	run_tag( argc, argv );	}
	else if ( 0 == strcmp( argv[1], "3d" ) )		{	composite_3d( argv[2], argv[3], argv[4], (argc > 5) ? atof( argv[5] ) : 0 );	}
	else if ( 0 == strcmp( argv[1], "test" ) )		{	test_resize( argv[2] );	}
	else if ( 0 == strcmp( argv[1], "captcha" ) )	{	run_captcha( argc, argv );	}
	else if ( 0 == strcmp( argv[1], "nightmare" ) )	{	run_nightmare( argc, argv );	}
	else if ( 0 == strcmp( argv[1], "rgbgr" ) )		{	rgbgr_net( argv[2], argv[3], argv[4] );	}
	else if ( 0 == strcmp( argv[1], "reset" ) )		{	reset_normalize_net( argv[2], argv[3], argv[4] );	}
	else if ( 0 == strcmp( argv[1], "denormalize" ) ){	denormalize_net( argv[2], argv[3], argv[4] );	}
	else if ( 0 == strcmp( argv[1], "statistics" ) ){	statistics_net( argv[2], argv[3] );				}
	else if ( 0 == strcmp( argv[1], "normalize" ) )	{	normalize_net( argv[2], argv[3], argv[4] );		}
	else if ( 0 == strcmp( argv[1], "rescale" ) )	{	rescale_net( argv[2], argv[3], argv[4] );		}
	else if ( 0 == strcmp( argv[1], "ops" ) )		{	operations( argv[2] );	}
	else if ( 0 == strcmp( argv[1], "speed" ) )		{	speed( argv[2], (argc > 3 && argv[3]) ? atoi( argv[3] ) : 0 );	}
	else if ( 0 == strcmp( argv[1], "oneoff" ) )	{	oneoff( argv[2], argv[3], argv[4] );	}
	else if ( 0 == strcmp( argv[1], "oneoff2" ) )	{	oneoff2( argv[2], argv[3], argv[4], atoi( argv[5] ) );	}
	else if ( 0 == strcmp( argv[1], "print" ) )		{	print_weights( argv[2], argv[3], atoi( argv[4] ) );	}
	else if ( 0 == strcmp( argv[1], "partial" ) )	{	partial( argv[2], argv[3], argv[4], atoi( argv[5] ) );	}
	else if ( 0 == strcmp( argv[1], "average" ) )	{	average( argc, argv );	}
	// 신경망 가중값을 시각화하여 보여줌
	//else if ( 0 == strcmp( argv[1], "visualize" ) )	{	visualize( argv[2], (argc > 3) ? argv[3] : 0 );	}
	else if ( 0 == strcmp( argv[1], "visualize" ) )	{	run_visualize( argc, argv );	}
	else if ( 0 == strcmp( argv[1], "mkimg" ) )		{	mkimg( argv[2], argv[3], atoi( argv[4] ), atoi( argv[5] ), atoi( argv[6] ), argv[7] );	}
	else if ( 0 == strcmp( argv[1], "imtest" ) )	{	test_resize( argv[2] );	}
	else if ( 0 == strcmp( argv[1], "compare" ) )	{	run_compare( argc, argv );	}
	else if ( 0 == strcmp( argv[1], "dice" ) )		{	run_dice( argc, argv );	}
	else if ( 0 == strcmp( argv[1], "writing" ) )	{	run_writing( argc, argv );	}
	else if ( 0 == strcmp( argv[1], "vid" ) )		{	run_vid_rnn( argc, argv );	}
	else
	{
		fprintf( stderr, "결정항목 구현안됨: %s\n", argv[1] );
	}

	return 0;
}

