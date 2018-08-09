#ifndef OPTION_LIST_H
#define OPTION_LIST_H
#include "list.h"

typedef struct
{
	char *key;
	char *val;
	int used;
} kvp;


list *read_data_cfg( char *filename );
int read_option( char *s, list *options );
void option_insert( list *l, char *key, char *val );
char *option_find( list *l, char *key );
char *option_find_str( list *l, char *key, char *def );	// 목록에서 기호(key)를 찾고 해당기호의 문자값을 반환한다
int option_find_int( list *l, char *key, int def );		// 목록에서 기호(key)를 찾고 해당기호의 정수값을 반환한다
int option_find_int_quiet( list *l, char *key, int def );
float option_find_float( list *l, char *key, float def );	// 목록에서 기호(key)를 찾고 해당기호의 실수값을 반환한다
float option_find_float_quiet( list *l, char *key, float def );
void option_unused( list *l );

#endif
