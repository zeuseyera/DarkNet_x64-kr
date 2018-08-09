
// 파일명: coco.h

#include <stdio.h>

#include "../src/network.h"
#include "../src/layer_detection.h"
#include "../src/layer_cost.h"
#include "../src/utils.h"
#include "../src/parser.h"
#include "../src/box.h"
#include "./demo.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

#ifdef __cplusplus	// 선언하는 헤더파일은 포함하지 않는다
extern "C" {
#endif
/*
char *coco_classes[] =
{
	"person",			"bicycle",		"car",				"motorcycle",		"airplane",
	"bus",				"train",		"truck",			"boat",				"traffic light",
	"fire hydrant",		"stop sign",	"parking meter",	"bench",			"bird",
	"cat",				"dog",			"horse",			"sheep",			"cow",
	"elephant",			"bear",			"zebra",			"giraffe",			"backpack",
	"umbrella",			"handbag",		"tie",				"suitcase",			"frisbee",
	"skis",				"snowboard",	"sports ball",		"kite",				"baseball bat",
	"baseball glove",	"skateboard",	"surfboard",		"tennis racket",	"bottle",
	"wine glass",		"cup",			"fork",				"knife",			"spoon",
	"bowl",				"banana",		"apple",			"sandwich",			"orange",
	"broccoli",			"carrot",		"hot dog",			"pizza",			"donut",
	"cake",				"chair",		"couch",			"potted plant",		"bed",
	"dining table",		"toilet",		"tv",				"laptop",			"mouse",
	"remote",			"keyboard",		"cell phone",		"microwave",		"oven",
	"toaster",			"sink",			"refrigerator",		"book",				"clock",
	"vase",				"scissors",		"teddy bear",		"hair drier",		"toothbrush"
};

int coco_ids[] =
{
	 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,13,14,15,16,17,18,19,20,
	21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,
	41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,
	61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90
};
*/
void train_coco( char *cfgfile, char *weightfile );

void print_cocos( FILE *fp
				, int image_id
				, box *boxes
				, float **probs
				, int num_boxes
				, int classes
				, int w
				, int h );

int get_coco_image_id( char *filename );

void validate_coco( char *cfgfile, char *weightfile );

void validate_coco_recall( char *cfgfile, char *weightfile );

void test_coco( char *cfgfile, char *weightfile, char *filename, float thresh );

void run_coco( int argc, char **argv );

#ifdef __cplusplus
}
#endif
