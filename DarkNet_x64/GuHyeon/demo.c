
#include "darknet.h"

#define DEMO 1

#ifdef OPENCV

static char **demo_names;		//
static image **demo_alphabet;	// 문자그림
static int demo_classes;		// 

static network *net;
static image buff[3];				// 쓰레드 처리용 입력자료 3장에 해당하는 메모리
static image buff_letter[3];
static int	 buff_index		= 0;	//
static CvCapture * cap;
static IplImage  * ipl;
static float fps			= 0;	//
static float demo_thresh	= 0;	// 쓰레드 처리용 문턱값
static float demo_hier		= 0.5f;	// 쓰레드 처리용 계층 문턱값
static int	 running		= 0;

static int	 demo_frame		= 3;	// 쓰레드 처리용 프레임수
static int	 demo_index		= 0;
static float **predictions;			// 쓰레드 처리용 예측한 신경망 출력값 복사본(출력개수 * 프레임수)
static float *avg;					// 쓰레드 처리용 신경망 출력값을 demo_frame 수로 평균한 값
static int	 demo_done		= 0;
static int	 demo_total		= 0;	// 쓰레드 처리용 신경망 검출층(YOLO, REGION, DETECTION)의 출력개수
double demo_time;


// 신경망 검출층(YOLO, REGION, DETECTION)의 출력개수를 알아낸다
int size_network( network *net )
{
	int ii;
	int count = 0;

	for ( ii=0; ii < net->n; ++ii )
	{
		layer l = net->layers[ii];

		if ( l.type == YOLO || l.type == REGION || l.type == DETECTION )
		{
			count += l.outputs;
		}
	}

	return count;
}
// 신경망 검출층(YOLO, REGION, DETECTION)의 출력값을 쓰레드 처리용 메모리로 복사
void remember_network( network *net )
{
	int ii;
	int count = 0;
	for ( ii=0; ii < net->n; ++ii )
	{
		layer l = net->layers[ii];
		if ( l.type == YOLO || l.type == REGION || l.type == DETECTION )
		{
			memcpy( predictions[demo_index] + count, net->layers[ii].output, sizeof( float ) * l.outputs );
			count += l.outputs;
		}
	}
}
// demo_frame 수로 평균값을 구하고, 신경망 출력메모리로 복사,
detection *avg_predictions( network *net, int *nboxes )
{
	int ii, jj;
	int count = 0;

	fill_cpu( demo_total, 0, avg, 1 );	// avg 메모리를 초기화

	// 프레임수를 반복
	for ( jj=0; jj < demo_frame; ++jj )
	{
		// predictions 의 값을 demo_frame 수로 나눈 평균값을 avg 에 더한다
		axpy_cpu( demo_total, 1.0f/demo_frame, predictions[jj], 1, avg, 1 );
	}

	for ( ii=0; ii < net->n; ++ii )
	{
		layer l = net->layers[ii];

		if ( l.type == YOLO || l.type == REGION || l.type == DETECTION )
		{
			// 평균값을 신경망 출력메모리로 복사
			memcpy( l.output, avg + count, sizeof( float ) * l.outputs );
			count += l.outputs;
		}
	}

	detection *dets = get_network_boxes( net
									, buff[0].w		// 너비
									, buff[0].h		// 높이
									, demo_thresh	// 문턱값
									, demo_hier		// 계층 문턱값
									, 0
									, 1
									, nboxes );		//	검출개수???

	return dets;
}

void *detect_in_thread( void *ptr )
{
	running = 1;
	float nms = 0.4f;

	layer l = net->layers[net->n-1];
	float *X = buff_letter[(buff_index+2)%3].data;
	network_predict( net, X );

/*	if ( l.type == DETECTION )
	{
		get_detection_boxes( l, 1, 1, demo_thresh, probs, boxes, 0 );
	}
	else
*/
	remember_network( net );
	detection *dets = 0;
	int nboxes = 0;
	dets = avg_predictions( net, &nboxes );

/*	int i, j;
	box zero = { 0 };
	int classes = l.classes;

	for ( i = 0; i < demo_detections; ++i )
	{
		avg[i].objectness = 0;
		avg[i].bbox = zero;
		memset( avg[i].prob, 0, classes*sizeof( float ) );

		for ( j = 0; j < demo_frame; ++j )
		{
			axpy_cpu( classes, 1./demo_frame, dets[j][i].prob, 1, avg[i].prob, 1 );
			avg[i].objectness += dets[j][i].objectness * 1./demo_frame;
			avg[i].bbox.x += dets[j][i].bbox.x * 1./demo_frame;
			avg[i].bbox.y += dets[j][i].bbox.y * 1./demo_frame;
			avg[i].bbox.w += dets[j][i].bbox.w * 1./demo_frame;
			avg[i].bbox.h += dets[j][i].bbox.h * 1./demo_frame;
		}

	//copy_cpu(classes, dets[0][i].prob, 1, avg[i].prob, 1);
	//avg[i].objectness = dets[0][i].objectness;
	}
*/

	if ( nms > 0 )
		do_nms_obj( dets, nboxes, l.classes, nms );

	printf( "\033[2J" );
	printf( "\033[1;1H" );
	printf( "\nFPS:%.1f\n", fps );
	printf( "Objects:\n\n" );

	image display = buff[(buff_index+2) % 3];

	draw_detections( display
				, dets				// 검출한 목록
				, nboxes			// 검출한 상자개수
				, demo_thresh
				, demo_names		// 검출 분류이름 목록
				, demo_alphabet		// 문자그림
				, demo_classes );	// 분류개수
	free_detections( dets, nboxes );

	demo_index = (demo_index + 1) % demo_frame;
	running = 0;
	return 0;
}

void *fetch_in_thread( void *ptr )
{
	int status = fill_image_from_stream( cap, buff[buff_index] );
	letterbox_image_into( buff[buff_index], net->w, net->h, buff_letter[buff_index] );

	if ( status == 0 ) demo_done = 1;
	return 0;
}

void *display_in_thread( void *ptr )
{
	show_image_cv( buff[(buff_index + 1)%3], "Demo", ipl );
	int ch = cvWaitKey( 1 );

	if ( ch != -1 ) ch = ch%256;

	if ( ch == 27 )			// ctrl+[ Esc
	{
		demo_done = 1;
		return 0;
	}
	else if ( ch == 82 )	// "R"
	{
		demo_thresh += 0.02f;
	}
	else if ( ch == 84 )	// "T"
	{
		demo_thresh -= 0.02f;
		if ( demo_thresh <= 0.02f ) demo_thresh = 0.02f;
	}
	else if ( ch == 83 )	// "S"
	{
		demo_hier += 0.02f;
	}
	else if ( ch == 81 )	// "Q"
	{
		demo_hier -= 0.02f;
		if ( demo_hier <= 0.0f ) demo_hier = 0.0f;
	}

	return 0;
}

void *display_loop( void *ptr )
{
	while ( 1 )
	{
		display_in_thread( 0 );
	}
}

void *detect_loop( void *ptr )
{
	while ( 1 )
	{
		detect_in_thread( 0 );
	}
}

void demo( char *cfgfile
		, char *weightfile
		, float thresh
		, int cam_index
		, const char *filename
		, char **names	// 분류이름 목록
		, int classes	// 분류개수
		, int delay
		, char *prefix
		, int avg_frames
		, float hier	// 계층 문턱값
		, int ww		//
		, int hh		//
		, int frames
		, int fullscreen )
{
	//demo_frame = avg_frames;
	image **alphabet = load_alphabet();	// 그림파일로 만들어진 문자이미지를 탑재한다
	demo_names		= names;
	demo_alphabet	= alphabet;
	demo_classes	= classes;
	demo_thresh		= thresh;
	demo_hier		= hier;				// 계층 문턱값

	printf( "Demo\n" );

	net = load_network( cfgfile, weightfile, 0 );
	set_batch_network( net, 1 );

	pthread_t detect_thread;
	pthread_t fetch_thread;
	srand( 2222222 );

	int ii;
	demo_total	= size_network( net );
	predictions	= calloc( demo_frame, sizeof( float* ) );

	for ( ii=0; ii < demo_frame; ++ii )
	{
		predictions[ii] = calloc( demo_total, sizeof( float ) );
	}

	avg = calloc( demo_total, sizeof( float ) );

	if ( filename )
	{
		//printf( "video file: %s\n", filename );	//  [7/12/2018 jobs]
		printf( "비디오 파일: %s\n", filename );		//  [7/12/2018 jobs]
		cap = cvCaptureFromFile( filename );
	}
	else
	{
		cap = cvCaptureFromCAM( cam_index );

		if ( ww )		{	cvSetCaptureProperty( cap, CV_CAP_PROP_FRAME_WIDTH, ww );	}
		if ( hh )		{	cvSetCaptureProperty( cap, CV_CAP_PROP_FRAME_HEIGHT, hh );	}
		if ( frames )	{	cvSetCaptureProperty( cap, CV_CAP_PROP_FPS, frames );		}
	}

	if ( !cap )
	//	error( "Couldn't connect to webcam.\n" );	//  [7/12/2018 jobs]
		error( "웹캠을 연결할수 없음...\n" );			//  [7/12/2018 jobs]

	buff[0] = get_image_from_stream( cap );
	buff[1] = copy_image( buff[0] );
	buff[2] = copy_image( buff[0] );
	buff_letter[0] = letterbox_image( buff[0], net->w, net->h );
	buff_letter[1] = letterbox_image( buff[0], net->w, net->h );
	buff_letter[2] = letterbox_image( buff[0], net->w, net->h );
	ipl = cvCreateImage( cvSize( buff[0].w, buff[0].h ), IPL_DEPTH_8U, buff[0].c );

	int count = 0;
	if ( !prefix )
	{
		cvNamedWindow( "Demo", CV_WINDOW_NORMAL );

		if ( fullscreen )
		{
			cvSetWindowProperty( "Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN );
		}
		else
		{
			cvMoveWindow( "Demo", 0, 0 );
			cvResizeWindow( "Demo", 1352, 1013 );
		}
	}

	demo_time = what_time_is_it_now();

	while ( !demo_done )
	{
		buff_index = (buff_index + 1) %3;
		if ( pthread_create( &fetch_thread, 0, fetch_in_thread, 0 ) )
			//error( "Thread creation failed" );	//  [7/12/2018 jobs]
			error( "쓰레드생성 실패!" );				//  [7/12/2018 jobs]
		if ( pthread_create( &detect_thread, 0, detect_in_thread, 0 ) )
			//error( "Thread creation failed" );	//  [7/12/2018 jobs]
			error( "쓰레드생성 실패!" );				//  [7/12/2018 jobs]

		if ( !prefix )
		{
			fps			= (float)( 1.0 / ( what_time_is_it_now() - demo_time ) );
			demo_time	= what_time_is_it_now();
			display_in_thread( 0 );
		}
		else
		{
			char name[256];
			//sprintf( name, "%s_%08d", prefix, count );
			sprintf_s( name, 256, "%s_%08d", prefix, count );
			save_image( buff[(buff_index + 1)%3], name );
		}

		pthread_join( fetch_thread, 0 );
		pthread_join( detect_thread, 0 );
		++count;
	}
}

/*
void demo_compare( char *cfg1
				, char *weight1
				, char *cfg2
				, char *weight2
				, float thresh
				, int cam_index
				, const char *filename
				, char **names
				, int classes
				, int delay
				, char *prefix
				, int avg_frames
				, float hier
				, int w
				, int h
				, int frames
				, int fullscreen)
{
	demo_frame		= avg_frames;
	predictions		= calloc( demo_frame, sizeof( float* ) );
	image **alphabet = load_alphabet();
	demo_names		= names;
	demo_alphabet	= alphabet;
	demo_classes	= classes;
	demo_thresh		= thresh;
	demo_hier		= hier;

	printf( "Demo\n" );

	net = load_network( cfg1, weight1, 0 );
	set_batch_network( net, 1 );
	pthread_t detect_thread;
	pthread_t fetch_thread;

	srand( 2222222 );

	if ( filename )
	{
		printf( "video file: %s\n", filename );
		cap = cvCaptureFromFile( filename );
	}
	else
	{
		cap = cvCaptureFromCAM( cam_index );

		if ( w )		{	cvSetCaptureProperty( cap, CV_CAP_PROP_FRAME_WIDTH, w );	}
		if ( h )		{	cvSetCaptureProperty( cap, CV_CAP_PROP_FRAME_HEIGHT, h );	}
		if ( frames )	{	cvSetCaptureProperty( cap, CV_CAP_PROP_FPS, frames );	}
	}

	if ( !cap ) error( "Couldn't connect to webcam.\n" );

	layer l = net->layers[net->n-1];
	demo_detections = l.n*l.w*l.h;

	avg = (float *)calloc( l.outputs, sizeof( float ) );

	int j;
	for ( j = 0; j < demo_frame; ++j )
	{
		predictions[j] = (float *)calloc( l.outputs, sizeof( float ) );
	}

	boxes = (box *)calloc( l.w*l.h*l.n, sizeof( box ) );
	probs = (float **)calloc( l.w*l.h*l.n, sizeof( float * ) );

	for ( j = 0; j < l.w*l.h*l.n; ++j )
	{
		probs[j] = (float *)calloc( l.classes+1, sizeof( float ) );
	}

	buff[0] = get_image_from_stream( cap );
	buff[1] = copy_image( buff[0] );
	buff[2] = copy_image( buff[0] );
	buff_letter[0] = letterbox_image( buff[0], net->w, net->h );
	buff_letter[1] = letterbox_image( buff[0], net->w, net->h );
	buff_letter[2] = letterbox_image( buff[0], net->w, net->h );
	ipl = cvCreateImage( cvSize( buff[0].w, buff[0].h ), IPL_DEPTH_8U, buff[0].c );

	int count = 0;
	if ( !prefix )
	{
		cvNamedWindow( "Demo", CV_WINDOW_NORMAL );
		if ( fullscreen )
		{
			cvSetWindowProperty( "Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN );
		}
		else
		{
			cvMoveWindow( "Demo", 0, 0 );
			cvResizeWindow( "Demo", 1352, 1013 );
		}
	}

	demo_time = what_time_is_it_now();

	while ( !demo_done )
	{
		buff_index = (buff_index + 1) %3;
		if ( pthread_create( &fetch_thread, 0, fetch_in_thread, 0 ) ) error( "Thread creation failed" );
		if ( pthread_create( &detect_thread, 0, detect_in_thread, 0 ) ) error( "Thread creation failed" );

		if ( !prefix )
		{
			fps = 1./(what_time_is_it_now() - demo_time);
			demo_time = what_time_is_it_now();
			display_in_thread( 0 );
		}
		else
		{
			char name[256];
			sprintf( name, "%s_%08d", prefix, count );
			save_image( buff[(buff_index + 1)%3], name );
		}

		pthread_join( fetch_thread, 0 );
		pthread_join( detect_thread, 0 );
		++count;
	}
}
*/

#else
void demo( char *cfgfile
		, char *weightfile
		, float thresh
		, int cam_index
		, const char *filename
		, char **names
		, int classes
		, int delay
		, char *prefix
		, int avg
		, float hier
		, int w
		, int h
		, int frames
		, int fullscreen )
{
	fprintf( stderr
	//	, "Demo needs OpenCV for webcam images.\n" );	//  [7/12/2018 jobs]
		, "데모는 웹캠 이미지를 위한 \"OpenCV\" 가 필요함!.\n" );	//  [7/12/2018 jobs]
}
#endif

