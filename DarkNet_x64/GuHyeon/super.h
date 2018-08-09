
// ���ϸ�: super.h

#include "../src/network.h"
#include "../src/layer_cost.h"
#include "../src/utils.h"
#include "../src/parser.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

#ifdef __cplusplus	// �����ϴ� ��������� �������� �ʴ´�
extern "C" {
#endif

void train_super( char *cfgfile, char *weightfile );

void test_super( char *cfgfile, char *weightfile, char *filename );

void run_super( int argc, char **argv );

#ifdef __cplusplus
}
#endif
