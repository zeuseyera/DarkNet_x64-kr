
// 파일명: voxel.h

#include "../src/network.h"
#include "../src/layer_cost.h"
#include "../src/utils.h"
#include "../src/parser.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/core/version.hpp"
#ifndef CV_VERSION_EPOCH
//#include "opencv2/videoio/videoio_c.h"
#endif
image get_image_from_stream( CvCapture *cap );
#endif

#ifdef __cplusplus	// 선언하는 헤더파일은 포함하지 않는다
extern "C" {
#endif

void extract_voxel( char *lfile, char *rfile, char *prefix );

void train_voxel( char *cfgfile, char *weightfile );

void test_voxel( char *cfgfile, char *weightfile, char *filename );

void run_voxel( int argc, char **argv );

#ifdef __cplusplus
}
#endif
