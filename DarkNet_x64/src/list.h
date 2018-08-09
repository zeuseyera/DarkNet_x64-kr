#ifndef LIST_H
#define LIST_H
// 목록의 마디 및 값
typedef struct node
{
    void *val;			// 목록의 값
    struct node *next;	// 현재목록
    struct node *prev;	// 이전목록
} node;
// 모든 목록
typedef struct list
{
    int size;		// 목록(마디) 개수
    node *front;	// 제일 첫번째 목록만 기록
    node *back;		// 현재목록
} list;

list *make_list();
int list_find(list *l, void *val);

void list_insert(list *, void *);	// 목록의 새로운 마디를 생성하고 값을 할당한다
// 목록으로 값에 대한 메모리를 할당하고 값을 추출한다
void **list_to_array( list *l );
//void **list_to_array( list *l, char *ChuGa );

void free_list(list *l);
void free_list_contents(list *l);

#endif
