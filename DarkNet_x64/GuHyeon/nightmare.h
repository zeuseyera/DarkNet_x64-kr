
#include "../src/network.h"
#include "../src/parser.h"
#include "../src/blas.h"
#include "../src/utils.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

// ./darknet nightmare cfg/extractor.recon.cfg ~/trained/yolo-coco.conv frame6.png -reconstruct -iters 500 -i 3 -lambda .1 -rate .01 -smooth 2

#ifdef __cplusplus	// 선언하는 헤더파일은 포함하지 않는다
extern "C" {
#endif

float abs_mean( float *x, int n );

void calculate_loss( float *output, float *delta, int n, float thresh );

void optimize_picture( network *net
					, image orig
					, int max_layer
					, float scale
					, float rate
					, float thresh
					, int norm );

void smooth( image recon, image update, float lambda, int num );

void reconstruct_picture( network net
						, float *features
						, image recon
						, image update
						, float rate
						, float momentum
						, float lambda
						, int smooth_size
						, int iters );

void run_nightmare( int argc, char **argv );

#ifdef __cplusplus
}
#endif
