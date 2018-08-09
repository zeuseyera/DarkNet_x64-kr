

// 파일명: tag.h

#include "../src/network.h"
#include "../src/utils.h"
#include "../src/parser.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

#ifdef __cplusplus	// 선언하는 헤더파일은 포함하지 않는다
extern "C" {
#endif

void train_tag( char *cfgfile, char *weightfile, int clear );

void test_tag( char *cfgfile, char *weightfile, char *filename );

void run_tag( int argc, char **argv );

#ifdef __cplusplus
}
#endif
