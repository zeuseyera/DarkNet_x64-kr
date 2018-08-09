#ifndef LIST_H
#define LIST_H
#include "darknet.h"

list *make_list();						// 목록구조의 메모리를 할당한다
int list_find( list *l, void *val );

void list_insert( list *, void * );		// 목록의 새로운 마디를 생성하고 값을 할당한다

void free_list_contents( list *l );		// 목록내용 할당메모리 해제

#endif
