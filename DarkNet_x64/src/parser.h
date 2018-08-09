
// ���ϸ�: parser.h

#ifndef PARSER_H
#define PARSER_H
#include "network.h"

#ifdef __cplusplus	// �����ϴ� ��������� �������� �ʴ´�
extern "C" {
#endif

network parse_network_cfg( char *filename );	// �Ű�� �������Ϸ� �Ű���� �����Ѵ�

// �򰡸� ���� ����̸� dropout ���� dropout ������ �缳���Ѵ�
void SeolJeong_network_droupout( network *net );	//, float BiYul=1.0f );

void save_network( network net, char *filename );
// �޸��� ���߰��� ������ ���Ϸ� �����Ѵ� 
void save_weights( network net, char *filename );
void save_weights_upto( network net, char *filename, int cutoff );
void save_weights_double( network net, char *filename );
// ������ ���Ͽ��� ���߰��� �а� ������ ���� ���߰��� �����Ѵ� 
void load_weights( network *net, char *filename );
void load_weights_upto( network *net, char *filename, int cutoff );

#ifdef __cplusplus
}
#endif

#endif
