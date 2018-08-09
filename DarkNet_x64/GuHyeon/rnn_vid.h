
// 파일명: rnn_vid.h

#include "../src/network.h"
#include "../src/layer_cost.h"
#include "../src/utils.h"
#include "../src/parser.h"
#include "../src/blas.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/core/version.hpp"

#ifndef CV_VERSION_EPOCH
//#include "opencv2/videoio/videoio_c.h"
#endif
/*
image get_image_from_stream( CvCapture *cap );
image ipl_to_image( IplImage* src );

void reconstruct_picture( network net
						, float *features
						, image recon
						, image update
						, float rate
						, float momentum
						, float lambda
						, int smooth_size
						, int iters );

typedef struct
{
	float *x;
	float *y;
} float_pair;

float_pair get_rnn_vid_data( network net, char **files, int n, int batch, int steps );
*/

#ifdef __cplusplus	// 선언하는 헤더파일은 포함하지 않는다
extern "C" {
#endif

void train_vid_rnn( char *cfgfile, char *weightfile );

image save_reconstruction( network net, image *init, float *feat, char *name, int i );

void generate_vid_rnn( char *cfgfile, char *weightfile );

void run_vid_rnn( int argc, char **argv );

#ifdef __cplusplus
}
#endif

#else
void run_vid_rnn( int argc, char **argv )
{
}

#endif

