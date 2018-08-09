
// 파일명: captcha.h

#include "../src/network.h"
#include "../src/utils.h"
#include "../src/parser.h"

#ifdef __cplusplus	// 선언하는 헤더파일은 포함하지 않는다
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
