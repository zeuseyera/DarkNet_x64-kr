
// 파일명: yolo.h

#include "../src/network.h"
#include "../src/layer_detection.h"
#include "../src/layer_cost.h"
#include "../src/utils.h"
#include "../src/parser.h"
#include "../src/box.h"
#include "./demo.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/version.hpp"
#ifndef CV_VERSION_EPOCH
//#include "opencv2/videoio/videoio_c.h"
#endif
#endif

#ifdef __cplusplus	// 선언하는 헤더파일은 포함하지 않는다
extern "C" {
#endif

void train_yolo( char *cfgfile, char *weightfile );

void print_yolo_detections( FILE **fps
						, char *id
						, box *boxes
						, float **probs
						, int total
						, int classes
						, int w
						, int h );

void validate_yolo( char *cfgfile, char *weightfile );

void validate_yolo_recall( char *cfgfile, char *weightfile );

void test_yolo( char *cfgfile, char *weightfile, char *filename, float thresh );

void run_yolo( int argc, char **argv );

#ifdef __cplusplus
}
#endif
