
// 파일명: parser.h

#ifndef PARSER_H
#define PARSER_H
#include "network.h"

#ifdef __cplusplus	// 선언하는 헤더파일은 포함하지 않는다
extern "C" {
#endif

network parse_network_cfg( char *filename );	// 신경망 설정파일로 신경망을 생성한다

// 평가를 위한 모드이면 dropout 층의 dropout 비율을 재설정한다
void SeolJeong_network_droupout( network *net );	//, float BiYul=1.0f );

void save_network( network net, char *filename );
// 메모리의 가중값을 선정한 파일로 저장한다 
void save_weights( network net, char *filename );
void save_weights_upto( network net, char *filename, int cutoff );
void save_weights_double( network net, char *filename );
// 선정한 파일에서 가중값을 읽고 지정한 망의 가중값을 변경한다 
void load_weights( network *net, char *filename );
void load_weights_upto( network *net, char *filename, int cutoff );

#ifdef __cplusplus
}
#endif

#endif
