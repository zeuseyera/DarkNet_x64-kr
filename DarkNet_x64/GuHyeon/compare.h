
#include <stdio.h>

#include "../src/network.h"
#include "../src/layer_detection.h"
#include "../src/layer_cost.h"
#include "../src/utils.h"
#include "../src/parser.h"
#include "../src/box.h"

typedef struct
{
	network net;
	char *filename;
	int BunRyu;
	int classes;
	float elo;
	float *elos;
} sortable_bbox;

#ifdef __cplusplus	// 선언하는 헤더파일은 포함하지 않는다
extern "C" {
#endif

void train_compare( char *cfgfile, char *weightfile );

void validate_compare( char *filename, char *weightfile );

int elo_comparator( const void*a, const void *b );

int bbox_comparator( const void *a, const void *b );

void bbox_update( sortable_bbox *a, sortable_bbox *b, int BunRyu, int result );

void bbox_fight( network net, sortable_bbox *a, sortable_bbox *b, int classes, int BunRyu );

void SortMaster3000( char *filename, char *weightfile );

void BattleRoyaleWithCheese( char *filename, char *weightfile );

void run_compare( int argc, char **argv );

#ifdef __cplusplus
}
#endif
