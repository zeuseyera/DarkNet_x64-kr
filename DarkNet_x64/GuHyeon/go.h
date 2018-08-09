
// 파일명: go.h

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

typedef struct
{
	char **data;
	int n;
} moves;

char *fgetgo( FILE *fp );

moves load_go_moves( char *filename );

void string_to_board( char *s, float *board );

void board_to_string( char *s, float *board );

void random_go_moves( moves m, float *boards, float *labels, int n );

void train_go( char *cfgfile, char *weightfile );

void propagate_liberty( float *board, int *lib, int *visited, int row, int col, int side );

int *calculate_liberties( float *board );

void print_board( float *board, int swap, int *indexes );

void flip_board( float *board );

void predict_move( network net, float *board, float *move, int multi );

void remove_connected( float *b, int *lib, int p, int r, int c );

void move_go( float *b, int p, int r, int c );

int makes_safe_go( float *b, int *lib, int p, int r, int c );

int suicide_go( float *b, int p, int r, int c );

int legal_go( float *b, char *ko, int p, int r, int c );

int generate_move( network net
				, int player
				, float *board
				, int multi
				, float thresh
				, float temp
				, char *ko
				, int print );

void valid_go( char *cfgfile, char *weightfile, int multi );

void engine_go( char *filename, char *weightfile, int multi );

void test_go( char *cfg, char *weights, int multi );

float score_game( float *board );

void self_go( char *filename, char *weightfile, char *f2, char *w2, int multi );

void run_go( int argc, char **argv );

#ifdef __cplusplus
}
#endif

