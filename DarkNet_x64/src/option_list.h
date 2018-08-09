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
char *option_find_str( list *l, char *key, char *def );	// ��Ͽ��� ��ȣ(key)�� ã�� �ش��ȣ�� ���ڰ��� ��ȯ�Ѵ�
int option_find_int( list *l, char *key, int def );		// ��Ͽ��� ��ȣ(key)�� ã�� �ش��ȣ�� �������� ��ȯ�Ѵ�
int option_find_int_quiet( list *l, char *key, int def );
float option_find_float( list *l, char *key, float def );	// ��Ͽ��� ��ȣ(key)�� ã�� �ش��ȣ�� �Ǽ����� ��ȯ�Ѵ�
float option_find_float_quiet( list *l, char *key, float def );
void option_unused( list *l );

#endif
