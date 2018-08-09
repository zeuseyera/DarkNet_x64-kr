
// ���ϸ�: captcha.h

#include "../src/network.h"
#include "../src/utils.h"
#include "../src/parser.h"

#ifdef __cplusplus	// �����ϴ� ��������� �������� �ʴ´�
extern "C" {
#endif

void fix_data_captcha( data d, int mask );

void train_captcha( char *cfgfile, char *weightfile );

void test_captcha( char *cfgfile, char *weightfile, char *filename );

void valid_captcha( char *cfgfile, char *weightfile, char *filename );

void run_captcha( int argc, char **argv );

#ifdef __cplusplus
}
#endif
