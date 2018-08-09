
// 파일명: detector.h

#include "../src/network.h"
#include "../src/layer_region.h"
#include "../src/layer_cost.h"
#include "../src/utils.h"
#include "../src/parser.h"
#include "../src/box.h"
#include "../src/option_list.h"

#include "./demo.h"

//#include "./coco.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/core/core_c.h"
#include "opencv2/core/version.hpp"
#ifndef CV_VERSION_EPOCH
//#include "opencv2/videoio/videoio_c.h"
//#pragma comment(lib, "opencv_world320.lib")
#else
#pragma comment(lib, "opencv_core2413.lib")  
#pragma comment(lib, "opencv_imgproc2413.lib")  
#pragma comment(lib, "opencv_highgui2413.lib") 
#endif
#endif

#ifdef __cplusplus	// 선언하는 헤더파일은 포함하지 않는다
extern "C" {
#endif
/*
static int coco_ids[] =
{
	 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,13,14,15,16,17,18,19,20,
	21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,
	41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,
	61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90
};
*/
void train_detector( char *datacfg
				, char *cfgfile
				, char *weightfile
				, int *gpus
				, int ngpus
				, int clear );

static int get_coco_image_id( char *filename );
/*
static void print_cocos( FILE *fp
				, char *image_path
				, box *boxes
				, float **probs
				, int num_boxes
				, int classes
				, int w
				, int h );
*/
void print_detector_detections( FILE **fps
				, char *id
				, box *boxes
				, float **probs
				, int total
				, int classes
				, int w
				, int h );

void print_imagenet_detections( FILE *fp
				, int id
				, box *boxes
				, float **probs
				, int total
				, int classes
				, int w
				, int h );

void validate_detector( char *datacfg, char *cfgfile, char *weightfile );

void validate_detector_recall( char *datacfg, char *cfgfile, char *weightfile );

void test_detector( char *datacfg
				, char *cfgfile
				, char *weightfile
				, char *filename
				, float thresh );

void run_detector( int argc, char **argv );

#ifdef __cplusplus
}
#endif
