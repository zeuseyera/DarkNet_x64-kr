
// ���ϸ�: dice.h

#include "../src/network.h"
#include "../src/utils.h"
#include "../src/parser.h"

char *dice_labels[] =
{
	"face1"
	, "face2"
	, "face3"
	, "face4"
	, "face5"
	, "face6"
};

#ifdef __cplusplus	// �����ϴ� ��������� �������� �ʴ´�
extern "C" {
#endif

void train_dice( char *cfgfile, char *weightfile );

void validate_dice( char *filename, char *weightfile );

void test_dice( char *cfgfile, char *weightfile, char *filename );

void run_dice( int argc, char **argv );

#ifdef __cplusplus
}
#endif
