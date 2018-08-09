#include <stdlib.h>
#include <string.h>
#include "list.h"

// 목록구조의 메모리를 할당한다
list *make_list()
{
	list *Lst = malloc( sizeof( list ) );
	Lst->size = 0;
	Lst->front = 0;
	Lst->back = 0;

	return Lst;
}

/*
void transfer_node(list *s, list *d, node *n)
{
	node *prev, *next;
	prev = n->prev;
	next = n->next;
	if(prev) prev->next = next;
	if(next) next->prev = prev;
	--s->size;
	if(s->front == n) s->front = next;
	if(s->back == n) s->back = prev;
}
*/

void *list_pop( list *Lst )
{
	if ( !Lst->back ) return 0;

	node *MaDi	= Lst->back;
	void *val	= MaDi->val;
	Lst->back	= MaDi->prev;

	if ( Lst->back )
		Lst->back->next = 0;

	free( MaDi );

	--Lst->size;

	return val;
}
// 목록의 새로운 마디를 생성하고 값을 할당한다
void list_insert( list *Lst, void *val )
{
	node *MaDi	= malloc( sizeof( node ) );
	MaDi->val	= val;	// 목록의 값
	MaDi->next	= 0;

	if ( !Lst->back )
	{
		Lst->front	= MaDi;	// 제일 첫번째 목록
		MaDi->prev	= 0;
	}
	else
	{
		Lst->back->next	= MaDi;
		MaDi->prev		= Lst->back;
	}

	Lst->back = MaDi;	// 추가한 목록을 할당한다
	++Lst->size;
}
// 마디 할당메모리 해제
void free_node( node *MaDi )
{
	node *DaEum;

	while ( MaDi )
	{
		DaEum = MaDi->next;
		free( MaDi );
		MaDi = DaEum;
	}
}
// 목록 할당메모리 해제
void free_list( list *Lst )
{
	free_node( Lst->front );
	free( Lst );
}
// 목록내용 할당메모리 해제
void free_list_contents( list *Lst )
{
	node *Jeon = Lst->front;

	while ( Jeon )
	{
		free( Jeon->val );
		Jeon = Jeon->next;
	}
}
// 목록 배열메모리 할당
void **list_to_array( list *Lst )
{
	void **Sae	= calloc( Lst->size, sizeof( void* ) );
	int count	= 0;
	node *MaDi	= Lst->front;

	while ( MaDi )
	{
		Sae[count++] = MaDi->val;
		MaDi = MaDi->next;
	}

	return Sae;
}
