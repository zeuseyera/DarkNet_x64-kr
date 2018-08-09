
// 파일명: art.h

#include "../src/network.h"
#include "../src/utils.h"
#include "../src/parser.h"
#include "../src/option_list.h"
#include "../src/blas.h"
#include "./classifier.h"

#ifdef WIN32
#include <time.h>
#include <winsock.h>
#include "../src/gettimeofday.h"

#else
#include <sys/time.h>

#endif

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/core/version.hpp"
#ifndef CV_VERSION_EPOCH
//#include "opencv2/videoio/videoio_c.h"
#endif

#endif

#ifdef __cplusplus	// 선언하는 헤더파일은 포함하지 않는다
extern "C" {
#endif

void demo_art(char *cfgfile, char *weightfile, int cam_index);

void run_art(int argc, char **argv);

#ifdef __cplusplus
}
#endif
