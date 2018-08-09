#ifndef LIST_H
#define LIST_H
// ����� ���� �� ��
typedef struct node
{
    void *val;			// ����� ��
    struct node *next;	// ������
    struct node *prev;	// �������
} node;
// ��� ���
typedef struct list
{
    int size;		// ���(����) ����
    node *front;	// ���� ù��° ��ϸ� ���
    node *back;		// ������
} list;

list *make_list();
int list_find(list *l, void *val);

void list_insert(list *, void *);	// ����� ���ο� ���� �����ϰ� ���� �Ҵ��Ѵ�
// ������� ���� ���� �޸𸮸� �Ҵ��ϰ� ���� �����Ѵ�
void **list_to_array( list *l );
//void **list_to_array( list *l, char *ChuGa );

void free_list(list *l);
void free_list_contents(list *l);

#endif
