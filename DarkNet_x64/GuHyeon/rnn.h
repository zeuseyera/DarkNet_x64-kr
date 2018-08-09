
// 파일명: rnn.h

#include "../src/network.h"
#include "../src/layer_cost.h"
#include "../src/utils.h"
#include "../src/blas.h"
#include "../src/parser.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

#ifdef __cplusplus	// 선언하는 헤더파일은 포함하지 않는다
extern "C" {
#endif

typedef struct
{
	float *x;
	float *y;
} float_pair;

int *read_tokenized_data( char *filename, size_t *read );

char **read_tokens( char *filename, size_t *read );

float_pair get_rnn_token_data( int *tokens
						, size_t *offsets
						, int characters
						, size_t len
						, int batch
						, int steps );

float_pair get_rnn_data( unsigned char *text
						, size_t *offsets
						, int characters
						, size_t len
						, int batch
						, int steps );

void reset_rnn_state( network net, int b );

void train_char_rnn( char *cfgfile, char *weightfile, char *filename, int clear, int tokenized );

void print_symbol( int n, char **tokens );

void test_char_rnn( char *cfgfile
				, char *weightfile
				, int num
				, char *seed
				, float temp
				, int rseed
				, char *token_file );

void test_tactic_rnn( char *cfgfile
				, char *weightfile
				, int num
				, float temp
				, int rseed
				, char *token_file );

void valid_tactic_rnn( char *cfgfile, char *weightfile, char *seed );

void valid_char_rnn( char *cfgfile, char *weightfile, char *seed );

void vec_char_rnn( char *cfgfile, char *weightfile, char *seed );

void run_char_rnn( int argc, char **argv );

#ifdef __cplusplus
}
#endif
