#ifndef LIST_H
#define LIST_H
#include "darknet.h"

list *make_list();						// ��ϱ����� �޸𸮸� �Ҵ��Ѵ�
int list_find( list *l, void *val );

void list_insert( list *, void * );		// ����� ���ο� ���� �����ϰ� ���� �Ҵ��Ѵ�

void free_list_contents( list *l );		// ��ϳ��� �Ҵ�޸� ����

#endif
