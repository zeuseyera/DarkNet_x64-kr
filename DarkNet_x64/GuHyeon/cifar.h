
// 파일명: cifar.h

#include "../src/network.h"
#include "../src/utils.h"
#include "../src/parser.h"
#include "../src/option_list.h"
#include "../src/blas.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

#ifdef __cplusplus	// 선언하는 헤더파일은 포함하지 않는다
extern "C" {
#endif

void train_cifar(char *cfgfile, char *weightfile);

void train_cifar_distill(char *cfgfile, char *weightfile);

void test_cifar_multi(char *filename, char *weightfile);

void test_cifar(char *filename, char *weightfile);

void extract_cifar();

void test_cifar_csv(char *filename, char *weightfile);

void test_cifar_csvtrain(char *filename, char *weightfile);

void eval_cifar_csv();

void run_cifar(int argc, char **argv);

#ifdef __cplusplus
}
#endif

