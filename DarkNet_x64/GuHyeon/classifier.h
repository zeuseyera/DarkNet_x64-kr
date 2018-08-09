
// ���ϸ�: classifier.h

#include <assert.h>

#include "../src/network.h"
#include "../src/utils.h"
#include "../src/parser.h"
#include "../src/option_list.h"
#include "../src/blas.h"
#include "../src/cuda.h"

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

#ifdef __cplusplus	// �����ϴ� ��������� �������� �ʴ´�
extern "C" {
#endif

float *get_regression_values( char **labels, int n );

void train_classifier( char *datacfg
					, char *cfgfile
					, char *weightfile
					, int *gpus
					, int ngpus
					, int clear );

void validate_classifier_crop( char *datacfg, char *filename, char *weightfile );

void validate_classifier_10( char *datacfg, char *filename, char *weightfile );

void validate_classifier_full( char *datacfg, char *filename, char *weightfile );

void validate_classifier_single( char *datacfg, char *filename, char *weightfile );

void validate_classifier_multi( char *datacfg, char *filename, char *weightfile );

void try_classifier( char *datacfg
				, char *cfgfile
				, char *weightfile
				, char *filename
				, int layer_num );
// ��°��� �����Ѵ�
void predict_classifier( char *datacfg
				, char *cfgfile
				, char *weightfile
				, char *filename
				, int top );
// ������ ����� �з� ������ ����Ѵ�
void label_classifier( char *datacfg, char *filename, char *weightfile );

void test_classifier( char *datacfg, char *cfgfile, char *weightfile, int target_layer );

void threat_classifier( char *datacfg
				, char *cfgfile
				, char *weightfile
				, int cam_index
				, const char *filename );

void gun_classifier( char *datacfg
				, char *cfgfile
				, char *weightfile
				, int cam_index
				, const char *filename );

void demo_classifier( char *datacfg
				, char *cfgfile
				, char *weightfile
				, int cam_index
				, const char *filename );

void run_classifier( int argc, char **argv );

#ifdef __cplusplus
}
#endif
