#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "layer_activation.h"
#include "layer_logistic.h"
#include "layer_l2norm.h"
#include "activations.h"
#include "layer_avgpool.h"
#include "layer_batchnorm.h"
#include "blas.h"
#include "layer_connected.h"
#include "layer_deconvolutional.h"
#include "layer_convolutional.h"
#include "layer_cost.h"
#include "layer_crnn.h"
#include "layer_crop.h"
#include "layer_detection.h"
#include "layer_dropout.h"
#include "layer_gru.h"
#include "list.h"
#include "layer_local.h"
#include "layer_maxpool.h"
#include "layer_normalization.h"
#include "option_list.h"
#include "parser.h"
#include "layer_region.h"
#include "layer_yolo.h"
#include "layer_reorg.h"
#include "layer_rnn.h"
#include "layer_route.h"
#include "layer_upsample.h"
#include "layer_shortcut.h"
#include "layer_softmax.h"
#include "layer_lstm.h"
#include "utils.h"

typedef struct
{
	char *type;		// 토막(부문)이름
	list *options;	// 선택사항 목록
} section;	// 토막

list *read_cfg( char *filename );

LAYER_TYPE string_to_layer_type( char * type )
{

	if ( strcmp( type, "[shortcut]" )==0 )			return SHORTCUT;
	if ( strcmp( type, "[crop]" )==0 )				return CROP;
	if ( strcmp( type, "[cost]" )==0 )				return COST;
	if ( strcmp( type, "[detection]" )==0 )			return DETECTION;
	if ( strcmp( type, "[region]" )==0 )			return REGION;
	if ( strcmp( type, "[yolo]" )==0 )				return YOLO;
	if ( strcmp( type, "[local]" )==0 )				return LOCAL;
	if ( strcmp( type, "[conv]" )==0 ||
		 strcmp( type, "[convolutional]" )==0 )		return CONVOLUTIONAL;
	if ( strcmp( type, "[deconv]" )==0 ||
		 strcmp( type, "[deconvolutional]" )==0 )	return DECONVOLUTIONAL;
	if ( strcmp( type, "[activation]" )==0 )		return ACTIVE;
	if ( strcmp( type, "[logistic]" )==0 )			return LOGXENT;
	if ( strcmp( type, "[l2norm]" )==0 )			return L2NORM;
	if ( strcmp( type, "[net]" )==0 ||
		 strcmp( type, "[network]" )==0 )			return NETWORK;
	if ( strcmp( type, "[crnn]" )==0 )				return CRNN;
	if ( strcmp( type, "[gru]" )==0 )				return GRU;
	if ( strcmp( type, "[lstm]" ) == 0 )			return LSTM;
	if ( strcmp( type, "[rnn]" )==0 )				return RNN;
	if ( strcmp( type, "[conn]" )==0 ||
		 strcmp( type, "[connected]" )==0 )			return CONNECTED;
	if ( strcmp( type, "[max]" )==0 ||
		 strcmp( type, "[maxpool]" )==0 )			return MAXPOOL;
	if ( strcmp( type, "[reorg]" )==0 )				return REORG;
	if ( strcmp( type, "[avg]" )==0 ||
		 strcmp( type, "[avgpool]" )==0 )			return AVGPOOL;
	if ( strcmp( type, "[dropout]" )==0 )			return DROPOUT;
	if ( strcmp( type, "[lrn]" )==0 ||
		 strcmp( type, "[normalization]" )==0 )		return NORMALIZATION;
	if ( strcmp( type, "[batchnorm]" )==0 )			return BATCHNORM;
	if ( strcmp( type, "[soft]" )==0 ||
		 strcmp( type, "[softmax]" )==0 )			return SOFTMAX;
	if ( strcmp( type, "[route]" )==0 )				return ROUTE;
	if ( strcmp( type, "[upsample]" )==0 )			return UPSAMPLE;

	return BLANK;
}

void free_section( section *s )
{
	free( s->type );
	node *n = s->options->front;

	while ( n )
	{
		kvp *pair = (kvp *)n->val;
		free( pair->key );
		free( pair );
		node *next = n->next;
		free( n );
		n = next;
	}

	free( s->options );
	free( s );
}

void parse_data( char *data, float *a, int n )
{
	if ( !data ) return;

	char *curr = data;
	char *next = data;
	int done = 0;

	int ii;
	for ( ii=0; ii < n && !done; ++ii )
	{
		while ( *++next !='\0' && *next != ',' );
		if ( *next == '\0' ) done = 1;
		*next = '\0';
		//sscanf( curr, "%g", &a[i] );
		sscanf_s( curr, "%g", &a[ii] );
		curr = next+1;
	}
}

typedef struct size_params
{
	int batch;
	int inputs;
	int h;
	int w;
	int c;
	int index;		// 현재단 순번
	int time_steps;
	network *net;
} size_params;

local_layer parse_local( list *options, size_params params )
{
	int n		= option_find_int( options, "filters", 1 );
	int size	= option_find_int( options, "size", 1 );
	int stride	= option_find_int( options, "stride", 1 );
	int pad		= option_find_int( options, "pad", 0 );
	char *activation_s		= option_find_str( options, "activation", "logistic" );
	ACTIVATION activation	= get_activation( activation_s );

	int batch, h, w, c;
	h = params.h;
	w = params.w;
	c = params.c;
	batch=params.batch;

	if ( !(h && w && c) )
	//	error( "Layer before local layer must output image." );	//  [7/6/2018 jobs]
		error( "local층 앞은 반드시 이미지출력층 이여야 함!" );	//  [7/6/2018 jobs]

	local_layer Lyr = make_local_layer( batch, h, w, c, n, size, stride, pad, activation );

	return Lyr;
}

layer parse_deconvolutional( list *options, size_params params )
{
	int n		= option_find_int( options, "filters", 1 );
	int size	= option_find_int( options, "size", 1 );
	int stride	= option_find_int( options, "stride", 1 );

	char *activation_s		= option_find_str( options, "activation", "logistic" );
	ACTIVATION activation	= get_activation( activation_s );

	int batch, h, w, c;
	h = params.h;
	w = params.w;
	c = params.c;
	batch=params.batch;

	if ( !(h && w && c) )
	//	error( "Layer before deconvolutional layer must output image." );	//  [7/6/2018 jobs]
		error( "deconvolutional층 앞은 반드시 이미지출력층 이여야 함!" );	//  [7/6/2018 jobs]

	int batch_normalize	= option_find_int_quiet( options, "batch_normalize", 0 );
	int pad				= option_find_int_quiet( options, "pad", 0 );
	int padding			= option_find_int_quiet( options, "padding", 0 );
	if ( pad ) padding = size/2;

	layer Lyr = make_deconvolutional_layer( batch
										, h
										, w
										, c
										, n
										, size
										, stride
										, padding
										, activation
										, batch_normalize
										, params.net->adam );

	return Lyr;
}


convolutional_layer parse_convolutional( list *options, size_params params )
{
	int n		= option_find_int( options, "filters", 1 );
	int size	= option_find_int( options, "size", 1 );
	int stride	= option_find_int( options, "stride", 1 );
	int pad		= option_find_int_quiet( options, "pad", 0 );
	int padding	= option_find_int_quiet( options, "padding", 0 );
	int groups	= option_find_int_quiet( options, "groups", 1 );
	if ( pad ) padding = size/2;

	char *activation_s		= option_find_str( options, "activation", "logistic" );
	ACTIVATION activation	= get_activation( activation_s );

	int batch, h, w, c;
	h = params.h;
	w = params.w;
	c = params.c;
	batch=params.batch;

	if ( !(h && w && c) )
	//	error( "Layer before convolutional layer must output image." );	//  [7/6/2018 jobs]
		error( "convolutional층 앞은 반드시 이미지출력층 이여야 함!" );	//  [7/6/2018 jobs]

	int batch_normalize	= option_find_int_quiet( options, "batch_normalize", 0 );
	int binary			= option_find_int_quiet( options, "binary", 0 );
	int xnor			= option_find_int_quiet( options, "xnor", 0 );

	convolutional_layer Lyr	= make_convolutional_layer( batch
														, h
														, w
														, c
														, n
														, groups
														, size
														, stride
														, padding
														, activation
														, batch_normalize
														, binary
														, xnor
														, params.net->adam );
	Lyr.flipped	= option_find_int_quiet( options, "flipped", 0 );
	Lyr.dot		= option_find_float_quiet( options, "dot", 0 );

	return Lyr;
}

layer parse_crnn( list *options, size_params params )
{
	int output_filters	= option_find_int( options, "output_filters", 1 );
	int hidden_filters	= option_find_int( options, "hidden_filters", 1 );
	char *activation_s	= option_find_str( options, "activation", "logistic" );
	ACTIVATION activation	= get_activation( activation_s );
	int batch_normalize	= option_find_int_quiet( options, "batch_normalize", 0 );

	layer l = make_crnn_layer( params.batch
							, params.w
							, params.h
							, params.c
							, hidden_filters
							, output_filters
							, params.time_steps
							, activation
							, batch_normalize );

	l.shortcut = option_find_int_quiet( options, "shortcut", 0 );

	return l;
}

layer parse_rnn( list *options, size_params params )
{
	int output			= option_find_int( options, "output", 1 );
	char *activation_s	= option_find_str( options, "activation", "logistic" );
	ACTIVATION activation	= get_activation( activation_s );
	int batch_normalize = option_find_int_quiet( options, "batch_normalize", 0 );

	layer Lyr = make_rnn_layer( params.batch
							, params.inputs
							, output
							, params.time_steps
							, activation, batch_normalize
							, params.net->adam );

	Lyr.shortcut = option_find_int_quiet( options, "shortcut", 0 );

	return Lyr;
}

layer parse_gru( list *options, size_params params )
{
	int output = option_find_int( options, "output", 1 );
	int batch_normalize = option_find_int_quiet( options, "batch_normalize", 0 );

	layer Lyr = make_gru_layer( params.batch
							, params.inputs
							, output
							, params.time_steps
							, batch_normalize
							, params.net->adam );
	Lyr.tanh = option_find_int_quiet( options, "tanh", 0 );

	return Lyr;
}

layer parse_lstm( list *options, size_params params )
{
	int output = option_find_int( options, "output", 1 );
	int batch_normalize = option_find_int_quiet( options, "batch_normalize", 0 );

	layer Lyr = make_lstm_layer( params.batch
							, params.inputs
							, output
							, params.time_steps
							, batch_normalize
							, params.net->adam );

	return Lyr;
}

layer parse_connected( list *options, size_params params )
{
	int output			= option_find_int( options, "output", 1 );
	char *activation_s	= option_find_str( options, "activation", "logistic" );
	ACTIVATION activation	= get_activation( activation_s );
	int batch_normalize = option_find_int_quiet( options, "batch_normalize", 0 );

	layer l = make_connected_layer( params.batch
								, params.inputs
								, output
								, activation
								, batch_normalize
								, params.net->adam );
	return l;
}

softmax_layer parse_softmax( list *options, size_params params )
{
	int groups			= option_find_int_quiet( options, "groups", 1 );
	softmax_layer Lyr	= make_softmax_layer( params.batch, params.inputs, groups );
	Lyr.temperature		= option_find_float_quiet( options, "temperature", 1 );
	char *tree_file		= option_find_str( options, "tree", 0 );

	if ( tree_file ) Lyr.softmax_tree = read_tree( tree_file );

	Lyr.w = params.w;
	Lyr.h = params.h;
	Lyr.c = params.c;
	Lyr.spatial = option_find_float_quiet( options, "spatial", 0 );

	return Lyr;
}
// 욜로 마스크 문장분석
int *parse_yolo_mask( char *aa, int *num )
{
	int *mask = 0;
	if ( aa )
	{
		int len = strlen( aa );
		int nn = 1;
		int ii;

		for ( ii=0; ii < len; ++ii )
		{
			if ( aa[ii] == ',' ) ++nn;
		}

		mask = calloc( nn, sizeof( int ) );

		for ( ii=0; ii < nn; ++ii )
		{
			int val = atoi( aa );
			mask[ii] = val;
			aa = strchr( aa, ',' )+1;
		}

		*num = nn;
	}

	return mask;
}
// 욜로 문장분석
layer parse_yolo( list *options, size_params params )
{
	int classes	= option_find_int( options, "classes", 20 );// 분류개수(COCO: 80개, VOC: 20개)
	int total	= option_find_int( options, "num", 1 );		// 6개(앵커 쌍 개수???)
	int num		= total;									// 마스크 개수(3개: 3,4,5	또는 1,2,3)
	// mask = 3,4,5	또는 mask = 1,2,3
	char *msk	= option_find_str( options, "mask", 0 );	// 마스크 값 문자열
	int *mask	= parse_yolo_mask( msk, &num );				// 마스크 값(3개: 3,4,5	또는 1,2,3)
	layer Lyr	= make_yolo_layer( params.batch, params.w, params.h, num, total, mask, classes );
	assert( Lyr.outputs == params.inputs );

	Lyr.max_boxes = option_find_int_quiet( options, "max", 90 );	// 상자 최대개수
	Lyr.jitter		= option_find_float( options, "jitter", 0.2f );	// 조금씩 움직이는 비율

	Lyr.ignore_thresh	= option_find_float( options, "ignore_thresh", 0.5f );	// 무시 문턱값
	Lyr.truth_thresh	= option_find_float( options, "truth_thresh", 1 );		// 목표(신뢰) 문턱값
	Lyr.random		= option_find_int_quiet( options, "random", 0 );

	char *map_file	= option_find_str( options, "map", 0 );

	if ( map_file )
		Lyr.map = read_map( map_file );

	// anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
	char *anc = option_find_str( options, "anchors", 0 );

	if ( anc )
	{
		int len = strlen( anc );
		int nn = 1;
		int ii;

		for ( ii=0; ii < len; ++ii )
		{
			if ( anc[ii] == ',' ) ++nn;
		}

		for ( ii=0; ii < nn; ++ii )
		{
			float bias		= atof( anc );
			Lyr.biases[ii]	= bias;
			anc				= strchr( anc, ',' )+1;
		}
	}

	return Lyr;
}

layer parse_region( list *options, size_params params )
{
	int coords	= option_find_int( options, "coords", 4 );
	int classes	= option_find_int( options, "classes", 20 );
	int num		= option_find_int( options, "num", 1 );

	layer Lyr = make_region_layer( params.batch, params.w, params.h, num, classes, coords );
	assert( Lyr.outputs == params.inputs );

	Lyr.log		= option_find_int_quiet( options, "log", 0 );
	Lyr.sqrt	= option_find_int_quiet( options, "sqrt", 0 );

	Lyr.softmax		= option_find_int( options, "softmax", 0 );
	Lyr.background	= option_find_int_quiet( options, "background", 0 );
	Lyr.max_boxes	= option_find_int_quiet( options, "max", 30 );
	Lyr.jitter		= option_find_float( options, "jitter", 0.2f );
	Lyr.rescore		= option_find_int_quiet( options, "rescore", 0 );

	Lyr.thresh		= option_find_float( options, "thresh", 0.5f );
	Lyr.classfix	= option_find_int_quiet( options, "classfix", 0 );
	Lyr.absolute	= option_find_int_quiet( options, "absolute", 0 );
	Lyr.random		= option_find_int_quiet( options, "random", 0 );

	Lyr.coord_scale		= option_find_float( options, "coord_scale", 1 );
	Lyr.object_scale	= option_find_float( options, "object_scale", 1 );
	Lyr.noobject_scale	= option_find_float( options, "noobject_scale", 1 );
	Lyr.mask_scale		= option_find_float( options, "mask_scale", 1 );
	Lyr.class_scale		= option_find_float( options, "class_scale", 1 );
	Lyr.bias_match		= option_find_int_quiet( options, "bias_match", 0 );

	char *tree_file		= option_find_str( options, "tree", 0 );
	if ( tree_file )
		Lyr.softmax_tree	= read_tree( tree_file );
	char *map_file		= option_find_str( options, "map", 0 );
	if ( map_file )
		Lyr.map			= read_map( map_file );

	char *a = option_find_str( options, "anchors", 0 );
	if ( a )
	{
		int len = strlen( a );
		int n = 1;
		int i;
		for ( i = 0; i < len; ++i )
		{
			if ( a[i] == ',' ) ++n;
		}
		for ( i = 0; i < n; ++i )
		{
			float bias = atof( a );
			Lyr.biases[i] = bias;
			a = strchr( a, ',' )+1;
		}
	}
	return Lyr;
}

detection_layer parse_detection( list *options, size_params params )
{
	int coords	= option_find_int( options, "coords", 1 );
	int classes	= option_find_int( options, "classes", 1 );
	int rescore	= option_find_int( options, "rescore", 0 );
	int num		= option_find_int( options, "num", 1 );
	int side	= option_find_int( options, "side", 7 );

	detection_layer Lyr = make_detection_layer( params.batch
												, params.inputs
												, num
												, side
												, classes
												, coords
												, rescore );

	Lyr.softmax			= option_find_int( options, "softmax", 0 );
	Lyr.sqrt			= option_find_int( options, "sqrt", 0 );

	Lyr.max_boxes		= option_find_int_quiet( options, "max", 90 );
	Lyr.coord_scale		= option_find_float( options, "coord_scale", 1 );
	Lyr.forced			= option_find_int( options, "forced", 0 );
	Lyr.object_scale	= option_find_float( options, "object_scale", 1 );
	Lyr.noobject_scale	= option_find_float( options, "noobject_scale", 1 );
	Lyr.class_scale		= option_find_float( options, "class_scale", 1 );
	Lyr.jitter			= option_find_float( options, "jitter", 0.2f );
	Lyr.random			= option_find_int_quiet( options, "random", 0 );
	Lyr.reorg			= option_find_int_quiet( options, "reorg", 0 );

	return Lyr;
}

cost_layer parse_cost( list *options, size_params params )
{
	char *type_s		= option_find_str( options, "type", "sse" );
	COST_TYPE type		= get_cost_type( type_s );
	float scale			= option_find_float_quiet( options, "scale", 1 );
	cost_layer Lyr		= make_cost_layer( params.batch, params.inputs, type, scale );
	Lyr.ratio			=  option_find_float_quiet( options, "ratio", 0 );
	Lyr.noobject_scale	=  option_find_float_quiet( options, "noobj", 1 );
	Lyr.thresh			=  option_find_float_quiet( options, "thresh", 0 );

	return Lyr;
}

crop_layer parse_crop( list *options, size_params params )
{
	int crop_height		= option_find_int( options, "crop_height", 1 );
	int crop_width		= option_find_int( options, "crop_width", 1 );
	int flip			= option_find_int( options, "flip", 0 );
	float angle			= option_find_float( options, "angle", 0 );
	float saturation	= option_find_float( options, "saturation", 1 );
	float exposure		= option_find_float( options, "exposure", 1 );

	int batch, h, w, c;
	h = params.h;
	w = params.w;
	c = params.c;
	batch=params.batch;

	if ( !(h && w && c) )
	//	error( "Layer before crop layer must output image." );	//  [7/6/2018 jobs]
		error( "crop층 앞은 반드시 이미지출력층 이여야 함!" );	//  [7/6/2018 jobs]

	int noadjust = option_find_int_quiet( options, "noadjust", 0 );

	crop_layer Lyr = make_crop_layer( batch
								, h
								, w
								, c
								, crop_height
								, crop_width
								, flip
								, angle
								, saturation
								, exposure );
	Lyr.shift = option_find_float( options, "shift", 0 );
	Lyr.noadjust = noadjust;
	return Lyr;
}

layer parse_reorg( list *options, size_params params )
{
	int stride	= option_find_int( options, "stride", 1 );
	int reverse	= option_find_int_quiet( options, "reverse", 0 );
	int flatten	= option_find_int_quiet( options, "flatten", 0 );
	int extra	= option_find_int_quiet( options, "extra", 0 );

	int batch, h, w, c;
	h = params.h;
	w = params.w;
	c = params.c;
	batch=params.batch;

	if ( !(h && w && c) )
	//	error( "Layer before reorg layer must output image." );	//  [7/6/2018 jobs]
		error( "reorg층 앞은 반드시 이미지출력층 이여야 함!" );	//  [7/6/2018 jobs]

	layer Lyr = make_reorg_layer( batch, w, h, c, stride, reverse, flatten, extra );
	return Lyr;
}

maxpool_layer parse_maxpool( list *options, size_params params )
{
	int stride	= option_find_int( options, "stride", 1 );
	int size	= option_find_int( options, "size", stride );
	int padding	= option_find_int_quiet( options, "padding", (size-1)/2 );

	int batch, h, w, c;
	h = params.h;
	w = params.w;
	c = params.c;
	batch=params.batch;

	if ( !(h && w && c) )
	//	error( "Layer before maxpool layer must output image." );	//  [7/6/2018 jobs]
		error( "maxpool층 앞은 반드시 이미지출력층 이여야 함!" );	//  [7/6/2018 jobs]

	maxpool_layer Lyr = make_maxpool_layer( batch, h, w, c, size, stride, padding );
	return Lyr;
}

avgpool_layer parse_avgpool( list *options, size_params params )
{
	int batch, w, h, c;
	w = params.w;
	h = params.h;
	c = params.c;
	batch=params.batch;

	if ( !(h && w && c) )
	//	error( "Layer before avgpool layer must output image." );	//  [7/6/2018 jobs]
		error( "avgpool층 앞은 반드시 이미지출력층 이여야 함!" );	//  [7/6/2018 jobs]

	avgpool_layer Lyr = make_avgpool_layer( batch, w, h, c );
	return Lyr;
}

dropout_layer parse_dropout( list *options, size_params params )
{
	float probability	= option_find_float( options, "probability", 0.5f );
	dropout_layer Lyr = make_dropout_layer( params.batch, params.inputs, probability );
	Lyr.out_w = params.w;
	Lyr.out_h = params.h;
	Lyr.out_c = params.c;

	return Lyr;
}

layer parse_normalization( list *options, size_params params )
{
	float alpha	= option_find_float( options, "alpha", 0.0001f );
	float beta	= option_find_float( options, "beta", 0.75f );
	float kappa	= option_find_float( options, "kappa", 1 );
	int size	= option_find_int( options, "size", 5 );
	layer Lyr = make_normalization_layer( params.batch
									, params.w
									, params.h
									, params.c
									, size
									, alpha
									, beta
									, kappa );
	return Lyr;
}

layer parse_batchnorm( list *options, size_params params )
{
	layer l = make_batchnorm_layer( params.batch, params.w, params.h, params.c );
	return l;
}

layer parse_shortcut( list *options, size_params params, network *net )
{
	char *l = option_find( options, "from" );
	int index = atoi( l );
	if ( index < 0 ) index = params.index + index;

	int batch = params.batch;
	layer from = net->layers[index];

	layer Lyr = make_shortcut_layer( batch
								, index
								, params.w
								, params.h
								, params.c
								, from.out_w
								, from.out_h
								, from.out_c );

	char *activation_s	= option_find_str( options, "activation", "linear" );
	ACTIVATION activation = get_activation( activation_s );

	Lyr.activation	= activation;
	Lyr.alpha		= option_find_float_quiet( options, "alpha", 1 );
	Lyr.beta		= option_find_float_quiet( options, "beta", 1 );
	return Lyr;
}


layer parse_l2norm( list *options, size_params params )
{
	layer Lyr = make_l2norm_layer( params.batch, params.inputs );
	Lyr.h = Lyr.out_h = params.h;
	Lyr.w = Lyr.out_w = params.w;
	Lyr.c = Lyr.out_c = params.c;
	return Lyr;
}


layer parse_logistic( list *options, size_params params )
{
	layer Lyr = make_logistic_layer( params.batch, params.inputs );
	Lyr.h = Lyr.out_h = params.h;
	Lyr.w = Lyr.out_w = params.w;
	Lyr.c = Lyr.out_c = params.c;
	return Lyr;
}

layer parse_activation( list *options, size_params params )
{
	char *activation_s = option_find_str( options, "activation", "linear" );
	ACTIVATION activation = get_activation( activation_s );

	layer Lyr = make_activation_layer( params.batch, params.inputs, activation );

	Lyr.h = Lyr.out_h = params.h;
	Lyr.w = Lyr.out_w = params.w;
	Lyr.c = Lyr.out_c = params.c;

	return Lyr;
}

layer parse_upsample( list *options, size_params params, network *net )
{

	int stride = option_find_int( options, "stride", 2 );
	layer Lyr = make_upsample_layer( params.batch, params.w, params.h, params.c, stride );
	Lyr.scale = option_find_float_quiet( options, "scale", 1 );
	return Lyr;
}

route_layer parse_route( list *options, size_params params, network *net )
{
	char *l = option_find( options, "layers" );
	int len = strlen( l );

	if ( !l )
	//	error( "Route Layer must specify input layers" );	//  [7/6/2018 jobs]
		error( "노선(Route)층은 입력(input)층을 반드시 지정해야 한다!" );	//  [7/6/2018 jobs]

	int n = 1;
	int i;
	for ( i = 0; i < len; ++i )
	{
		if ( l[i] == ',' ) ++n;
	}

	int *layers = calloc( n, sizeof( int ) );
	int *sizes = calloc( n, sizeof( int ) );
	for ( i = 0; i < n; ++i )
	{
		int index = atoi( l );
		l = strchr( l, ',' )+1;
		if ( index < 0 ) index = params.index + index;
		layers[i] = index;
		sizes[i] = net->layers[index].outputs;
	}
	int batch = params.batch;

	route_layer Lyr = make_route_layer( batch, n, layers, sizes );

	convolutional_layer first = net->layers[layers[0]];

	Lyr.out_w = first.out_w;
	Lyr.out_h = first.out_h;
	Lyr.out_c = first.out_c;

	for ( i = 1; i < n; ++i )
	{
		int index = layers[i];
		convolutional_layer next = net->layers[index];

		if ( next.out_w == first.out_w && next.out_h == first.out_h )
		{
			Lyr.out_c += next.out_c;
		}
		else
		{
			Lyr.out_h = Lyr.out_w = Lyr.out_c = 0;
		}
	}

	return Lyr;
}

learning_rate_policy get_policy( char *str )
{
	if ( strcmp( str, "random" )	== 0 )	return RANDOM;
	if ( strcmp( str, "poly" )		== 0 )	return POLY;
	if ( strcmp( str, "constant" )	== 0 )	return CONSTANT;
	if ( strcmp( str, "step" )		== 0 )	return STEP;
	if ( strcmp( str, "exp" )		== 0 )	return EXP;
	if ( strcmp( str, "sigmoid" )	== 0 )	return SIG;
	if ( strcmp( str, "steps" )		== 0 )	return STEPS;

	//fprintf( stderr, "Couldn't find policy %s, going with constant\n", s );	//  [7/6/2018 jobs]
	fprintf( stderr, " %s 정책 없음, 상수(constant)로 지정함!\n", str );	//  [7/6/2018 jobs]

	return CONSTANT;
}

data_type get_jaryogubun( char *str )
{
	if ( strcmp( str, "detection" )		== 0 )	return DETECTION_DATA;
	if ( strcmp( str, "captcha" )		== 0 )	return CAPTCHA_DATA;
	if ( strcmp( str, "region" )		== 0 )	return REGION_DATA;
	if ( strcmp( str, "image" )			== 0 )	return IMAGE_DATA;
	if ( strcmp( str, "compare" )		== 0 )	return COMPARE_DATA;
	if ( strcmp( str, "writing" )		== 0 )	return WRITING_DATA;
	if ( strcmp( str, "swag" )			== 0 )	return SWAG_DATA;
	if ( strcmp( str, "tag" )			== 0 )	return TAG_DATA;
	if ( strcmp( str, "old_calss" )		== 0 )	return OLD_CLASSIFICATION_DATA;
	if ( strcmp( str, "study" )			== 0 )	return STUDY_DATA;
	if ( strcmp( str, "det" )			== 0 )	return DET_DATA;
	if ( strcmp( str, "super" )			== 0 )	return SUPER_DATA;
	if ( strcmp( str, "letterbox" )		== 0 )	return LETTERBOX_DATA;
	if ( strcmp( str, "regression" )	== 0 )	return REGRESSION_DATA;
	if ( strcmp( str, "segmentation" )	== 0 )	return SEGMENTATION_DATA;
	if ( strcmp( str, "instance" )		== 0 )	return INSTANCE_DATA;
	if ( strcmp( str, "mnist" )			== 0 )	return MNIST_DATA;
	if ( strcmp( str, "byeorim" )		== 0 )	return BYEORIM_DATA;

	fprintf( stderr, " %s 자료유형 없음, 분류(Classification)로 지정함!\n", str );	

	return CLASSIFICATION_DATA;
}

void parse_net_options( list *options, network *net )
{
	net->batch		= option_find_int( options, "batch", 1 );
	net->learning_rate = option_find_float( options, "learning_rate", 0.001f );
	net->momentum	= option_find_float( options, "momentum", 0.9f );
	net->decay		= option_find_float( options, "decay", 0.0001f );
	int subdivs		= option_find_int( options, "subdivisions", 1 );
	net->time_steps	= option_find_int_quiet( options, "time_steps", 1 );
	net->notruth	= option_find_int_quiet( options, "notruth", 0 );
	net->batch /= subdivs;
	net->batch *= net->time_steps;
	net->subdivisions = subdivs;
	net->random		= option_find_int_quiet( options, "random", 0 );

	net->adam		= option_find_int_quiet( options, "adam", 0 );
	if ( net->adam )
	{
		net->B1		= option_find_float( options, "B1", 0.9f );
		net->B2		= option_find_float( options, "B2", 0.999f );
		net->eps	= option_find_float( options, "eps", 0.0000001f );
	}

	net->h			= option_find_int_quiet( options, "height", 0 );
	net->w			= option_find_int_quiet( options, "width", 0 );
	net->c			= option_find_int_quiet( options, "channels", 0 );
	net->inputs		= option_find_int_quiet( options, "inputs", net->h * net->w * net->c );
	net->max_crop	= option_find_int_quiet( options, "max_crop", net->w*2 );
	net->min_crop	= option_find_int_quiet( options, "min_crop", net->w );
	net->max_ratio	= option_find_float_quiet( options, "max_ratio", (float)net->max_crop / net->w );
	net->min_ratio	= option_find_float_quiet( options, "min_ratio", (float)net->min_crop / net->w );
	net->center		= option_find_int_quiet( options, "center", 0 );
	net->clip		= option_find_float_quiet( options, "clip", 0 );

	net->angle		= option_find_float_quiet( options, "angle", 0 );
	net->aspect		= option_find_float_quiet( options, "aspect", 1 );
	net->saturation = option_find_float_quiet( options, "saturation", 1 );
	net->exposure	= option_find_float_quiet( options, "exposure", 1 );
	net->hue		= option_find_float_quiet( options, "hue", 0 );

	if ( !net->inputs && !(net->h && net->w && net->c) )
	//	error( "No input parameters supplied" );	//  [7/6/2018 jobs]
		error( "입력에 관한 참여(parameters)가 제공되지 않음!" );	//  [7/6/2018 jobs]

	char *policy_s	= option_find_str( options, "policy", "constant" );
	net->policy		= get_policy( policy_s );
	net->burn_in	= option_find_int_quiet( options, "burn_in", 0 );
	net->power		= option_find_float_quiet( options, "power", 4 );

	if ( net->policy == STEP )
	{
		net->step	= option_find_int( options, "step", 1 );
		net->scale	= option_find_float( options, "scale", 1 );
	}
	else if ( net->policy == STEPS )
	{
		char *l = option_find( options, "steps" );
		char *p = option_find( options, "scales" );

		if ( !l || !p )
		//	error( "STEPS policy must have steps and scales in cfg file" );	//  [7/6/2018 jobs]
			error( "STEPS 정책은 cfg 파일에 걸음(steps) 과 배율(scales) 이 반드시 있어야함!" );	//  [7/6/2018 jobs]

		int len	= strlen( l );
		int nn	= 1;

		int ii;
		for ( ii=0; ii < len; ++ii )
		{
			if ( l[ii] == ',' ) ++nn;
		}

		int *steps		= calloc( nn, sizeof( int ) );
		float *scales	= calloc( nn, sizeof( float ) );

		for ( ii=0; ii < nn; ++ii )
		{
			int step    = atoi( l );
			float scale = (float)atof( p );
			l = strchr( l, ',' )+1;
			p = strchr( p, ',' )+1;
			steps[ii] = step;
			scales[ii] = scale;
		}

		net->scales = scales;
		net->steps = steps;
		net->num_steps = nn;
	}
	else if ( net->policy == EXP )
	{
		net->gamma = option_find_float( options, "gamma", 1 );
	}
	else if ( net->policy == SIG )
	{
		net->gamma = option_find_float( options, "gamma", 1 );
		net->step = option_find_int( options, "step", 1 );
	}
	else if ( net->policy == POLY || net->policy == RANDOM )
	{
	}

	net->max_batches = option_find_int( options, "max_batches", 0 );

	char *jaryojong	= option_find_str( options, "jaryogubun", "class" );
	net->JaRyoJong	= get_jaryogubun( jaryojong );
}

int is_network( section *s )
{
	return (strcmp( s->type, "[net]" )==0
		|| strcmp( s->type, "[network]" )==0);
}
// 신경망 구성파일로 신경망을 만든다
network *parse_network_cfg( char *filename )
{
	list *sections = read_cfg( filename );
	node *nd = sections->front;

	if ( !nd )
		//error( "Config file has no sections" );	//  [7/6/2018 jobs]
		error( "구성파일 부분 없음!" );	//  [7/6/2018 jobs]

	network *net = make_network( sections->size - 1 );
	net->gpu_index = gpu_index;
	size_params params;

	section *s = (section *)nd->val;
	list *options = s->options;

	if ( !is_network( s ) )
		//error( "First section must be [net] or [network]" );	//  [7/6/2018 jobs]
		error( "첫번째 부분은 반드시 [net] 또는 [network] 여야 함! " );	//  [7/6/2018 jobs]

	parse_net_options( options, net );	// 망 선택사항 설정

	params.h = net->h;
	params.w = net->w;
	params.c = net->c;
	params.inputs = net->inputs;
	params.batch = net->batch;
	params.time_steps = net->time_steps;
	params.net = net;

	size_t workspace_size = 0;
	nd = nd->next;
	int count = 0;
	free_section( s );
	//fprintf( stderr, "layer     filters    size              input                output\n" );	//  [7/7/2018 jobs]
	fprintf( stderr, "층        포집(필터)  크기               입력                  출력\n" );	//  [7/7/2018 jobs]

	while ( nd )
	{
		params.index = count;
		fprintf( stderr, "%5d ", count );
		s = (section *)nd->val;
		options = s->options;

		layer Lyr = { 0 };
		LAYER_TYPE lt = string_to_layer_type( s->type );

		if		( lt == CONVOLUTIONAL )		{	Lyr = parse_convolutional( options, params );	}
		else if ( lt == DECONVOLUTIONAL )	{	Lyr = parse_deconvolutional( options, params );	}
		else if ( lt == LOCAL )				{	Lyr = parse_local( options, params );			}
		else if ( lt == ACTIVE )			{	Lyr = parse_activation( options, params );		}
		else if ( lt == LOGXENT )			{	Lyr = parse_logistic( options, params );		}
		else if ( lt == L2NORM )			{	Lyr = parse_l2norm( options, params );			}
		else if ( lt == RNN )				{	Lyr = parse_rnn( options, params );				}
		else if ( lt == GRU )				{	Lyr = parse_gru( options, params );				}
		else if ( lt == LSTM )				{	Lyr = parse_lstm( options, params );			}
		else if ( lt == CRNN )				{	Lyr = parse_crnn( options, params );			}
		else if ( lt == CONNECTED )			{	Lyr = parse_connected( options, params );		}
		else if ( lt == CROP )				{	Lyr = parse_crop( options, params );			}
		else if ( lt == COST )				{	Lyr = parse_cost( options, params );			}
		else if ( lt == REGION )			{	Lyr = parse_region( options, params );			}
		else if ( lt == YOLO )				{	Lyr = parse_yolo( options, params );			}
		else if ( lt == DETECTION )			{	Lyr = parse_detection( options, params );		}
		else if ( lt == SOFTMAX )
		{
												Lyr = parse_softmax( options, params );
			net->hierarchy = Lyr.softmax_tree;
		}
		else if ( lt == NORMALIZATION )		{	Lyr = parse_normalization( options, params );	}
		else if ( lt == BATCHNORM )			{	Lyr = parse_batchnorm( options, params );		}
		else if ( lt == MAXPOOL )			{	Lyr = parse_maxpool( options, params );			}
		else if ( lt == REORG )				{	Lyr = parse_reorg( options, params );			}
		else if ( lt == AVGPOOL )			{	Lyr = parse_avgpool( options, params );			}
		else if ( lt == ROUTE )				{	Lyr = parse_route( options, params, net );		}
		else if ( lt == UPSAMPLE )			{	Lyr = parse_upsample( options, params, net );	}
		else if ( lt == SHORTCUT )			{	Lyr = parse_shortcut( options, params, net );	}
		else if ( lt == DROPOUT )
		{
			Lyr = parse_dropout( options, params );
			Lyr.output		= net->layers[count-1].output;
			Lyr.delta		= net->layers[count-1].delta;

			#ifdef GPU
			Lyr.output_gpu	= net->layers[count-1].output_gpu;
			Lyr.delta_gpu	= net->layers[count-1].delta_gpu;
			#endif
		}
		else
		{
			//fprintf( stderr, "Type not recognized: %s\n", s->type );	//  [7/6/2018 jobs]
			fprintf( stderr, "층 유형 인식안됨: %s\n", s->type );	//  [7/6/2018 jobs]
		}

		Lyr.clip				= net->clip;
		Lyr.truth				= option_find_int_quiet( options, "truth", 0 );
		Lyr.onlyforward			= option_find_int_quiet( options, "onlyforward", 0 );
		Lyr.stopbackward		= option_find_int_quiet( options, "stopbackward", 0 );
		Lyr.dontsave			= option_find_int_quiet( options, "dontsave", 0 );
		Lyr.dontload			= option_find_int_quiet( options, "dontload", 0 );
		Lyr.dontloadscales		= option_find_int_quiet( options, "dontloadscales", 0 );
		Lyr.learning_rate_scale	= option_find_float_quiet( options, "learning_rate", 1 );
		Lyr.smooth				= option_find_float_quiet( options, "smooth", 0 );

		option_unused( options );	// 미사용 선택사항을 출력하여 알려준다

		net->layers[count] = Lyr;	// 생성한 현재단의 주소를 저장한다

		if ( Lyr.workspace_size > workspace_size )
			workspace_size = Lyr.workspace_size;

		free_section( s );
		nd = nd->next;
		++count;

		if ( nd )
		{
			params.h = Lyr.out_h;
			params.w = Lyr.out_w;
			params.c = Lyr.out_c;
			params.inputs = Lyr.outputs;
		}
	}

	free_list( sections );
	layer out		= get_network_output_layer( net );
	net->outputs	= out.outputs;
	net->truths		= out.outputs;

	if ( net->layers[net->n-1].truths )
		net->truths = net->layers[net->n-1].truths;

	net->output	= out.output;
	net->input	= calloc( net->inputs*net->batch, sizeof( float ) );
	net->truth	= calloc( net->truths*net->batch, sizeof( float ) );

	#ifdef GPU
	net->output_gpu	= out.output_gpu;
	net->input_gpu	= cuda_make_array( net->input, net->inputs*net->batch );
	net->truth_gpu	= cuda_make_array( net->truth, net->truths*net->batch );
	#endif

	if ( workspace_size )
	{
		//printf("%ld\n", workspace_size);
		#ifdef GPU
		if ( gpu_index >= 0 )
		{
			net->workspace = cuda_make_array( 0, (workspace_size-1)/sizeof( float )+1 );
		}
		else
		{
			net->workspace = calloc( 1, workspace_size );
		}
		#else
		net->workspace = calloc( 1, workspace_size );
		#endif
	}
	return net;
}

list *read_cfg( char *filename )
{
	//FILE *file = fopen( filename, "r" );
	FILE *file; fopen_s( &file, filename, "r" );

	if ( file == 0 ) file_error( filename );

	char *line;
	int nu = 0;
	list *options = make_list();
	section *current = 0;

	while ( (line=fgetl( file )) != 0 )
	{
		++nu;
		strip( line );
		switch ( line[0] )
		{
		case '[':
			current = malloc( sizeof( section ) );
			list_insert( options, current );
			current->options = make_list();
			current->type = line;
			break;
		case '\0':
		case '#':
		case ';':
			free( line );
			break;
		default:
			if ( !read_option( line, current->options ) )
			{
				fprintf( stderr
						//, "Config file error line %d, could parse: %s\n"	//  [7/6/2018 jobs]
						, "구성파일 오류발생 행: %d, 분석내용: %s\n"				//  [7/6/2018 jobs]
						, nu, line );
				free( line );
			}
			break;
		}
	}
	fclose( file );
	return options;
}

void save_convolutional_weights_binary( layer Lyr, FILE *fp )
{
	#ifdef GPU
	if ( gpu_index >= 0 )
	{
		pull_convolutional_layer( Lyr );
	}
	#endif

	binarize_weights( Lyr.weights, Lyr.n, Lyr.c*Lyr.size*Lyr.size, Lyr.binary_weights );
	int size = Lyr.c*Lyr.size*Lyr.size;
	int i, j, k;
	fwrite( Lyr.biases, sizeof( float ), Lyr.n, fp );

	if ( Lyr.batch_normalize )
	{
		fwrite( Lyr.scales, sizeof( float ), Lyr.n, fp );
		fwrite( Lyr.rolling_mean, sizeof( float ), Lyr.n, fp );
		fwrite( Lyr.rolling_variance, sizeof( float ), Lyr.n, fp );
	}

	for ( i = 0; i < Lyr.n; ++i )
	{
		float mean = Lyr.binary_weights[i*size];

		if ( mean < 0 ) mean = -mean;
		fwrite( &mean, sizeof( float ), 1, fp );

		for ( j = 0; j < size/8; ++j )
		{
			int index = i*size + j*8;
			unsigned char c = 0;

			for ( k = 0; k < 8; ++k )
			{
				if ( j*8 + k >= size ) break;
				if ( Lyr.binary_weights[index + k] > 0 ) c = (c | 1<<k);
			}
			fwrite( &c, sizeof( char ), 1, fp );
		}
	}
}

void save_convolutional_weights( layer Lyr, FILE *fp )
{
	if ( Lyr.binary )
	{
		//save_convolutional_weights_binary(l, fp);
		//return;
	}

	#ifdef GPU
	if ( gpu_index >= 0 )
	{
		pull_convolutional_layer( Lyr );
	}
	#endif

	int num = Lyr.nweights;
	fwrite( Lyr.biases, sizeof( float ), Lyr.n, fp );

	if ( Lyr.batch_normalize )
	{
		fwrite( Lyr.scales, sizeof( float ), Lyr.n, fp );
		fwrite( Lyr.rolling_mean, sizeof( float ), Lyr.n, fp );
		fwrite( Lyr.rolling_variance, sizeof( float ), Lyr.n, fp );
	}
	fwrite( Lyr.weights, sizeof( float ), num, fp );
}

void save_batchnorm_weights( layer Lyr, FILE *fp )
{
	#ifdef GPU
	if ( gpu_index >= 0 )
	{
		pull_batchnorm_layer( Lyr );
	}
	#endif

	fwrite( Lyr.scales, sizeof( float ), Lyr.c, fp );
	fwrite( Lyr.rolling_mean, sizeof( float ), Lyr.c, fp );
	fwrite( Lyr.rolling_variance, sizeof( float ), Lyr.c, fp );
}

void save_connected_weights( layer Lyr, FILE *fp )
{
	#ifdef GPU
	if ( gpu_index >= 0 )
	{
		pull_connected_layer( Lyr );
	}
	#endif

	fwrite( Lyr.biases, sizeof( float ), Lyr.outputs, fp );
	fwrite( Lyr.weights, sizeof( float ), Lyr.outputs*Lyr.inputs, fp );

	if ( Lyr.batch_normalize )
	{
		fwrite( Lyr.scales, sizeof( float ), Lyr.outputs, fp );
		fwrite( Lyr.rolling_mean, sizeof( float ), Lyr.outputs, fp );
		fwrite( Lyr.rolling_variance, sizeof( float ), Lyr.outputs, fp );
	}
}

void save_weights_upto( network *net, char *filename, int cutoff )
{
	#ifdef GPU
	if ( net->gpu_index >= 0 )
	{
		cuda_set_device( net->gpu_index );
	}
	#endif

	//fprintf( stderr, "Saving weights to %s\n", filename );	//  [7/6/2018 jobs]
	fprintf( stderr, "저장한 가중값 파일이름: %s\n", filename );	//  [7/6/2018 jobs]
	//FILE *fp = fopen( filename, "wb" );
	FILE *fp; fopen_s( &fp, filename, "wb" );

	if ( !fp ) file_error( filename );

	int major = 0;
	int minor = 2;
	int revision = 0;
	fwrite( &major, sizeof( int ), 1, fp );
	fwrite( &minor, sizeof( int ), 1, fp );
	fwrite( &revision, sizeof( int ), 1, fp );
	fwrite( net->seen, sizeof( size_t ), 1, fp );

	int ii;
	for ( ii=0; ii < net->n && ii < cutoff; ++ii )
	{
		layer Lyr = net->layers[ii];
		if ( Lyr.dontsave ) continue;

		if ( Lyr.type == CONVOLUTIONAL || Lyr.type == DECONVOLUTIONAL )
		{
			save_convolutional_weights( Lyr, fp );
		}

		if ( Lyr.type == CONNECTED )
		{
			save_connected_weights( Lyr, fp );
		}

		if ( Lyr.type == BATCHNORM )
		{
			save_batchnorm_weights( Lyr, fp );
		}

		if ( Lyr.type == RNN )
		{
			save_connected_weights( *(Lyr.input_layer), fp );
			save_connected_weights( *(Lyr.self_layer), fp );
			save_connected_weights( *(Lyr.output_layer), fp );
		}

		if ( Lyr.type == LSTM )
		{
			save_connected_weights( *(Lyr.wi), fp );
			save_connected_weights( *(Lyr.wf), fp );
			save_connected_weights( *(Lyr.wo), fp );
			save_connected_weights( *(Lyr.wg), fp );
			save_connected_weights( *(Lyr.ui), fp );
			save_connected_weights( *(Lyr.uf), fp );
			save_connected_weights( *(Lyr.uo), fp );
			save_connected_weights( *(Lyr.ug), fp );
		}

		if ( Lyr.type == GRU )
		{
			if ( 1 )
			{
				save_connected_weights( *(Lyr.wz), fp );
				save_connected_weights( *(Lyr.wr), fp );
				save_connected_weights( *(Lyr.wh), fp );
				save_connected_weights( *(Lyr.uz), fp );
				save_connected_weights( *(Lyr.ur), fp );
				save_connected_weights( *(Lyr.uh), fp );
			}
			else
			{
				save_connected_weights( *(Lyr.reset_layer), fp );
				save_connected_weights( *(Lyr.update_layer), fp );
				save_connected_weights( *(Lyr.state_layer), fp );
			}
		}

		if ( Lyr.type == CRNN )
		{
			save_convolutional_weights( *(Lyr.input_layer), fp );
			save_convolutional_weights( *(Lyr.self_layer), fp );
			save_convolutional_weights( *(Lyr.output_layer), fp );
		}

		if ( Lyr.type == LOCAL )
		{
			#ifdef GPU
			if ( gpu_index >= 0 )
			{
				pull_local_layer( Lyr );
			}
			#endif

			int locations = Lyr.out_w*Lyr.out_h;
			int size = Lyr.size*Lyr.size*Lyr.c*Lyr.n*locations;

			fwrite( Lyr.biases, sizeof( float ), Lyr.outputs, fp );
			fwrite( Lyr.weights, sizeof( float ), size, fp );
		}
	}

	fclose( fp );
}
void save_weights( network *net, char *filename )
{
	save_weights_upto( net, filename, net->n );
}

void transpose_matrix( float *a, int rows, int cols )
{
	float *transpose = calloc( rows*cols, sizeof( float ) );
	int x, y;
	for ( x = 0; x < rows; ++x )
	{
		for ( y = 0; y < cols; ++y )
		{
			transpose[y*rows + x] = a[x*cols + y];
		}
	}
	memcpy( a, transpose, rows*cols*sizeof( float ) );
	free( transpose );
}

void load_connected_weights( layer Lyr, FILE *fp, int transpose )
{
	fread( Lyr.biases, sizeof( float ), Lyr.outputs, fp );
	fread( Lyr.weights, sizeof( float ), Lyr.outputs*Lyr.inputs, fp );
	if ( transpose )
	{
		transpose_matrix( Lyr.weights, Lyr.inputs, Lyr.outputs );
	}

	//printf( "Biases: %f mean %f variance\n"
	//		, mean_array(Lyr.biases, Lyr.outputs)
	//		, variance_array(Lyr.biases, Lyr.outputs) );
	//printf( "Weights: %f mean %f variance\n"
	//		, mean_array(Lyr.weights, Lyr.outputs*Lyr.inputs)
	//		, variance_array(Lyr.weights, Lyr.outputs*Lyr.inputs) );

	if ( Lyr.batch_normalize && (!Lyr.dontloadscales) )
	{
		fread( Lyr.scales, sizeof( float ), Lyr.outputs, fp );
		fread( Lyr.rolling_mean, sizeof( float ), Lyr.outputs, fp );
		fread( Lyr.rolling_variance, sizeof( float ), Lyr.outputs, fp );
		//printf( "Scales: %f mean %f variance\n"
		//		, mean_array(Lyr.scales, Lyr.outputs)
		//		, variance_array(Lyr.scales, Lyr.outputs));
		//printf( "rolling_mean: %f mean %f variance\n"
		//		, mean_array( Lyr.rolling_mean, Lyr.outputs )
		//		, variance_array( Lyr.rolling_mean, Lyr.outputs ));
		//printf( "rolling_variance: %f mean %f variance\n"
		//		, mean_array( Lyr.rolling_variance, Lyr.outputs )
		//		, variance_array( Lyr.rolling_variance, Lyr.outputs ));
	}

	#ifdef GPU
	if ( gpu_index >= 0 )
	{
		push_connected_layer( Lyr );
	}
	#endif
}

void load_batchnorm_weights( layer Lyr, FILE *fp )
{
	fread( Lyr.scales, sizeof( float ), Lyr.c, fp );
	fread( Lyr.rolling_mean, sizeof( float ), Lyr.c, fp );
	fread( Lyr.rolling_variance, sizeof( float ), Lyr.c, fp );

	#ifdef GPU
	if ( gpu_index >= 0 )
	{
		push_batchnorm_layer( Lyr );
	}
	#endif
}

void load_convolutional_weights_binary( layer Lyr, FILE *fp )
{
	fread( Lyr.biases, sizeof( float ), Lyr.n, fp );

	if ( Lyr.batch_normalize && (!Lyr.dontloadscales) )
	{
		fread( Lyr.scales, sizeof( float ), Lyr.n, fp );
		fread( Lyr.rolling_mean, sizeof( float ), Lyr.n, fp );
		fread( Lyr.rolling_variance, sizeof( float ), Lyr.n, fp );
	}

	int size = Lyr.c*Lyr.size*Lyr.size;
	int i, j, k;
	for ( i = 0; i < Lyr.n; ++i )
	{
		float mean = 0;
		fread( &mean, sizeof( float ), 1, fp );

		for ( j = 0; j < size/8; ++j )
		{
			int index = i*size + j*8;
			unsigned char c = 0;
			fread( &c, sizeof( char ), 1, fp );

			for ( k = 0; k < 8; ++k )
			{
				if ( j*8 + k >= size ) break;
				Lyr.weights[index + k] = (c & 1<<k) ? mean : -mean;
			}
		}
	}

	#ifdef GPU
	if ( gpu_index >= 0 )
	{
		push_convolutional_layer( Lyr );
	}
	#endif
}

void load_convolutional_weights( layer Lyr, FILE *fp )
{
	if ( Lyr.binary )
	{
		//load_convolutional_weights_binary(l, fp);
		//return;
	}

	int num = Lyr.nweights;
	fread( Lyr.biases, sizeof( float ), Lyr.n, fp );

	if ( Lyr.batch_normalize && (!Lyr.dontloadscales) )
	{
		fread( Lyr.scales, sizeof( float ), Lyr.n, fp );
		fread( Lyr.rolling_mean, sizeof( float ), Lyr.n, fp );
		fread( Lyr.rolling_variance, sizeof( float ), Lyr.n, fp );
		if ( 0 )
		{
			int i;
			for ( i = 0; i < Lyr.n; ++i )
			{
				printf( "%g, ", Lyr.rolling_mean[i] );
			}
			printf( "\n" );
			for ( i = 0; i < Lyr.n; ++i )
			{
				printf( "%g, ", Lyr.rolling_variance[i] );
			}
			printf( "\n" );
		}
		if ( 0 )
		{
			fill_cpu( Lyr.n, 0, Lyr.rolling_mean, 1 );
			fill_cpu( Lyr.n, 0, Lyr.rolling_variance, 1 );
		}
		if ( 0 )
		{
			int i;
			for ( i = 0; i < Lyr.n; ++i )
			{
				printf( "%g, ", Lyr.rolling_mean[i] );
			}
			printf( "\n" );
			for ( i = 0; i < Lyr.n; ++i )
			{
				printf( "%g, ", Lyr.rolling_variance[i] );
			}
			printf( "\n" );
		}
	}

	fread( Lyr.weights, sizeof( float ), num, fp );
	//if(Lyr.c == 3) scal_cpu(num, 1./256, Lyr.weights, 1);
	if ( Lyr.flipped )
	{
		transpose_matrix( Lyr.weights, Lyr.c*Lyr.size*Lyr.size, Lyr.n );
	}
	//if (Lyr.binary) binarize_weights(Lyr.weights, Lyr.n, Lyr.c*Lyr.size*Lyr.size, Lyr.weights);

	#ifdef GPU
	if ( gpu_index >= 0 )
	{
		push_convolutional_layer( Lyr );
	}
	#endif
}


void load_weights_upto( network *net, char *filename, int start, int cutoff )
{
	#ifdef GPU
	if ( net->gpu_index >= 0 )
	{
		cuda_set_device( net->gpu_index );
	}
	#endif

	//fprintf( stderr, "Loading weights from %s...", filename );	//  [7/6/2018 jobs]
	fprintf( stderr, "탑재중인 가중값 파일이름: %s\n", filename );	//  [7/6/2018 jobs]
	fflush( stdout );

	//FILE *fp = fopen( filename, "rb" );
	FILE *fp; fopen_s( &fp, filename, "rb" );
	if ( !fp ) file_error( filename );

	int major;
	int minor;
	int revision;
	fread( &major, sizeof( int ), 1, fp );
	fread( &minor, sizeof( int ), 1, fp );
	fread( &revision, sizeof( int ), 1, fp );

	if ( (major*10 + minor) >= 2 && major < 1000 && minor < 1000 )
	{
		fread( net->seen, sizeof( size_t ), 1, fp );
	}
	else
	{
		int iseen = 0;
		fread( &iseen, sizeof( int ), 1, fp );
		*net->seen = iseen;
	}

	int transpose = (major > 1000) || (minor > 1000);

	int ii;
	for ( ii=start; ii < net->n && ii < cutoff; ++ii )
	{
		layer Lyr = net->layers[ii];
		if ( Lyr.dontload ) continue;
		if ( Lyr.type == CONVOLUTIONAL || Lyr.type == DECONVOLUTIONAL )
		{
			load_convolutional_weights( Lyr, fp );
		}
		if ( Lyr.type == CONNECTED )
		{
			load_connected_weights( Lyr, fp, transpose );
		}
		if ( Lyr.type == BATCHNORM )
		{
			load_batchnorm_weights( Lyr, fp );
		}
		if ( Lyr.type == CRNN )
		{
			load_convolutional_weights( *(Lyr.input_layer), fp );
			load_convolutional_weights( *(Lyr.self_layer), fp );
			load_convolutional_weights( *(Lyr.output_layer), fp );
		}
		if ( Lyr.type == RNN )
		{
			load_connected_weights( *(Lyr.input_layer), fp, transpose );
			load_connected_weights( *(Lyr.self_layer), fp, transpose );
			load_connected_weights( *(Lyr.output_layer), fp, transpose );
		}
		if ( Lyr.type == LSTM )
		{
			load_connected_weights( *(Lyr.wi), fp, transpose );
			load_connected_weights( *(Lyr.wf), fp, transpose );
			load_connected_weights( *(Lyr.wo), fp, transpose );
			load_connected_weights( *(Lyr.wg), fp, transpose );
			load_connected_weights( *(Lyr.ui), fp, transpose );
			load_connected_weights( *(Lyr.uf), fp, transpose );
			load_connected_weights( *(Lyr.uo), fp, transpose );
			load_connected_weights( *(Lyr.ug), fp, transpose );
		}
		if ( Lyr.type == GRU )
		{
			if ( 1 )
			{
				load_connected_weights( *(Lyr.wz), fp, transpose );
				load_connected_weights( *(Lyr.wr), fp, transpose );
				load_connected_weights( *(Lyr.wh), fp, transpose );
				load_connected_weights( *(Lyr.uz), fp, transpose );
				load_connected_weights( *(Lyr.ur), fp, transpose );
				load_connected_weights( *(Lyr.uh), fp, transpose );
			}
			else
			{
				load_connected_weights( *(Lyr.reset_layer), fp, transpose );
				load_connected_weights( *(Lyr.update_layer), fp, transpose );
				load_connected_weights( *(Lyr.state_layer), fp, transpose );
			}
		}
		if ( Lyr.type == LOCAL )
		{
			int locations = Lyr.out_w*Lyr.out_h;
			int size = Lyr.size*Lyr.size*Lyr.c*Lyr.n*locations;

			fread( Lyr.biases, sizeof( float ), Lyr.outputs, fp );
			fread( Lyr.weights, sizeof( float ), size, fp );

			#ifdef GPU
			if ( gpu_index >= 0 )
			{
				push_local_layer( Lyr );
			}
			#endif
		}
	}

	//fprintf( stderr, "Done!\n" );	//  [7/6/2018 jobs]
	fprintf( stderr, "가중값 탑재함!\n" );	//  [7/6/2018 jobs]
	fclose( fp );
}

void load_weights( network *net, char *filename )
{
	load_weights_upto( net, filename, 0, net->n );
}

