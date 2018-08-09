#include <stdlib.h>
#include <string.h>
#include "list.h"

list *make_list()
{
	list *l = malloc( sizeof(list) );
	l->size = 0;
	l->front = 0;
	l->back = 0;
	return l;
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

void *list_pop( list *l )
{
    if( !l->back ) return 0;

    node *b = l->back;
    void *val = b->val;
    l->back = b->prev;

    if( l->back )
		l->back->next = 0;

    free(b);
    --l->size;
    
    return val;
}
// ����� ���ο� ���� �����ϰ� ���� �Ҵ��Ѵ�
void list_insert( list *MogRog, void *Gab )
{
	// ����� ���ο� ���� �����Ѵ�
	node *SaeMaDi = malloc( sizeof(node) );
	SaeMaDi->val = Gab;	// ����� ��
	SaeMaDi->next = 0;

	if( !MogRog->back )
	{
		MogRog->front = SaeMaDi;	// ���� ù��° ���
		SaeMaDi->prev = 0;
	}
	else
	{
		MogRog->back->next = SaeMaDi;	
		SaeMaDi->prev = MogRog->back;
	}

	MogRog->back = SaeMaDi;	// �߰��� ����� �Ҵ��Ѵ�
	++MogRog->size;
}

void free_node( node *n )
{
	node *next;

	while( n )
	{
		next = n->next;
		free( n );
		n = next;
	}
}

void free_list( list *l )
{
	free_node( l->front );
	free( l );
}

void free_list_contents( list *l )
{
	node *n = l->front;

	while( n )
	{
		free( n->val );
		n = n->next;
	}
}
// ������� ���� ���� �޸𸮸� �Ҵ��ϰ� ���� �����Ѵ�
void **list_to_array( list *l )
{
    void **a = calloc( l->size, sizeof(void*) );
    int count = 0;
    node *n = l->front;

    while( n )
	{
        a[count++] = n->val;
        n = n->next;
    }

    return a;
}
/*
void **list_to_array( list *l, char *ChuGa )
{
	void **a = calloc( l->size, sizeof( void* ) );
	int count = 0;
	node *n = l->front;

	while ( n )
	{
		char buff[256];
		sprintf( buff, "%s/%s", ChuGa, n->val );

		a[count++] = buff;
		n = n->next;
	}

	return a;
}
*/