// Oh boy, why am I about to do this....
#ifndef NETWORK_H
#define NETWORK_H
#include "darknet.h"

#include "image.h"
#include "layer.h"
#include "data.h"
#include "tree.h"

#ifdef __cplusplus
extern "C" {
	#endif

	#ifdef GPU
	// 장치의 신경망 출력단 출력값을 쥔장 출력값 메모리로 긁어온다
	void pull_network_output( network *net );
	#endif

	//network *load_network( char *cfg, char *weights, int clear );	// c2732 에러 [6/28/2018 jobs]
	//void set_batch_network( network *net, int b );				// c2732 에러 [6/28/2018 jobs]
	//void visualize_network( network *net );						// c2732 에러 [6/28/2018 jobs]
	//float *network_predict( network *net, float *input );			// c2732 에러 [6/28/2018 jobs]
	//int resize_network( network *net, int w, int h );

	void compare_networks( network *n1, network *n2, data d );
	char *get_layer_string( LAYER_TYPE a );

	network *make_network( int n );


	float network_accuracy_multi( network *net, data d, int n );
	int get_predicted_class_network( network *net );
	void print_network( network *net );
	void calc_network_cost( network *net );

	#ifdef __cplusplus
}
#endif

#endif

