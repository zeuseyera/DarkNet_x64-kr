#ifndef PARSER_H
#define PARSER_H
#include "darknet.h"
#include "network.h"

network *parse_network_cfg( char *filename );	// c2732 에러 [6/28/2018 jobs]
void save_weights( network *net, char *filename );	// c2732 에러 [6/28/2018 jobs]
void load_weights( network *net, char *filename );	// c2732 에러 [6/28/2018 jobs]
void save_weights_upto( network *net, char *filename, int cutoff );	// c2732 에러 [6/28/2018 jobs]
void load_weights_upto( network *net, char *filename, int start, int cutoff );	// c2732 에러 [6/28/2018 jobs]

void save_network(network net, char *filename);
void save_weights_double(network net, char *filename);

#endif
