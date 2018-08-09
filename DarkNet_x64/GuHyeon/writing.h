
// ���ϸ�: writing.h

#include "../src/network.h"
#include "../src/utils.h"
#include "../src/parser.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

#ifdef __cplusplus	// �����ϴ� ��������� �������� �ʴ´�
extern "C" {
#endif

void train_writing( char *cfgfile, char *weightfile );

void test_writing( char *cfgfile, char *weightfile, char *filename );

void run_writing( int argc, char **argv );

#ifdef __cplusplus
}
#endif
