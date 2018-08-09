#include <stdio.h>
#include <time.h>
#include <assert.h>
#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "blas.h"

#include "layer_crop.h"
#include "layer_connected.h"
#include "layer_gru.h"
#include "layer_rnn.h"
#include "layer_crnn.h"
#include "layer_local.h"
#include "layer_convolutional.h"
#include "layer_activation.h"
#include "layer_detection.h"
#include "layer_region.h"
#include "layer_yolo.h"
#include "layer_normalization.h"
#include "layer_batchnorm.h"
#include "layer_maxpool.h"
#include "layer_reorg.h"
#include "layer_avgpool.h"
#include "layer_cost.h"
#include "layer_softmax.h"
#include "layer_dropout.h"
#include "layer_route.h"
#include "layer_upsample.h"
#include "layer_shortcut.h"
#include "parser.h"
#include "data.h"

load_args get_base_args( network *net )
{
	load_args args	= { 0 };
	args.w			= net->w;
	args.h			= net->h;
	args.size		= net->w;

	args.min		= net->min_crop;
	args.max		= net->max_crop;
	args.angle		= net->angle;
	args.aspect		= net->aspect;
	args.exposure	= net->exposure;
	args.center		= net->center;
	args.saturation	= net->saturation;
	args.hue		= net->hue;

	return args;
}

network *load_network( char *cfg, char *weights, int clear )
{
	network *net = parse_network_cfg( cfg );

	if ( weights && weights[0] != 0 )
	{
		load_weights( net, weights );
	}

	if ( clear ) (*net->seen) = 0;

	return net;
}

size_t get_current_batch( network *net )
{
	size_t batch_num = (*net->seen)/(net->batch*net->subdivisions);
	return batch_num;
}

void reset_network_state( network *net, int b )
{
	int i;
	for ( i = 0; i < net->n; ++i )
	{
		#ifdef GPU
		layer Lyr = net->layers[i];
		if ( Lyr.state_gpu )
		{
			fill_gpu( Lyr.outputs, 0, Lyr.state_gpu + Lyr.outputs*b, 1 );
		}
		if ( Lyr.h_gpu )
		{
			fill_gpu( Lyr.outputs, 0, Lyr.h_gpu + Lyr.outputs*b, 1 );
		}
		#endif
	}
}

void reset_rnn( network *net )
{
	reset_network_state( net, 0 );
}

float get_current_rate( network *net )
{
	size_t batch_num = get_current_batch( net );
	int ii;
	float rate;

	if ( batch_num < net->burn_in )
		return net->learning_rate * pow( (float)batch_num / net->burn_in, net->power );

	switch ( net->policy )
	{
	case CONSTANT:
		return net->learning_rate;
	case STEP:
		return net->learning_rate * pow( net->scale, (float)batch_num/net->step );
	case STEPS:
		rate = net->learning_rate;
		for ( ii=0; ii < net->num_steps; ++ii )
		{
			if ( net->steps[ii] > batch_num ) return rate;
			rate *= net->scales[ii];
		}
		return rate;
	case EXP:
		return net->learning_rate * pow( net->gamma, batch_num );
	case POLY:
		return net->learning_rate * pow( 1.0 - (float)batch_num / net->max_batches, net->power );
	case RANDOM:
		return net->learning_rate * pow( rand_uniform( 0.0f, 1.0f ), net->power );
	case SIG:
		return net->learning_rate * ( 1.0 / ( 1.0 + exp( net->gamma * ( batch_num - net->step ) ) ) );
	default:
		fprintf( stderr, "Policy is weird!\n" );
		return net->learning_rate;
	}
}

char *get_layer_string( LAYER_TYPE a )
{
	switch ( a )
	{
	case CONVOLUTIONAL:
		return "convolutional";
	case ACTIVE:
		return "activation";
	case LOCAL:
		return "local";
	case DECONVOLUTIONAL:
		return "deconvolutional";
	case CONNECTED:
		return "connected";
	case RNN:
		return "rnn";
	case GRU:
		return "gru";
	case LSTM:
		return "lstm";
	case CRNN:
		return "crnn";
	case MAXPOOL:
		return "maxpool";
	case REORG:
		return "reorg";
	case AVGPOOL:
		return "avgpool";
	case SOFTMAX:
		return "softmax";
	case DETECTION:
		return "detection";
	case REGION:
		return "region";
	case YOLO:
		return "yolo";
	case DROPOUT:
		return "dropout";
	case CROP:
		return "crop";
	case COST:
		return "cost";
	case ROUTE:
		return "route";
	case SHORTCUT:
		return "shortcut";
	case NORMALIZATION:
		return "normalization";
	case BATCHNORM:
		return "batchnorm";
	default:
		break;
	}
	return "none";
}

network *make_network( int n )
{
	network *net = calloc( 1, sizeof( network ) );
	net->n		= n;
	net->layers	= calloc( net->n, sizeof( layer ) );
	net->seen	= calloc( 1, sizeof( size_t ) );
	net->t	    = calloc( 1, sizeof( int ) );
	net->cost	= calloc( 1, sizeof( float ) );
	return net;
}
// �Ű���� ������� ���Ͽ� ��������
void forward_network( network *netp )
{
	#ifdef GPU
	if ( netp->gpu_index >= 0 )
	{
		forward_network_gpu( netp );	//
		return;
	}
	#endif

	network net = *netp;

	int ii;
	for ( ii=0; ii < net.n; ++ii )
	{
		net.index = ii;
		layer Lyr = net.layers[ii];
		// �ڷᰪ �迭���� ��ǥ���� �����´�
		if ( Lyr.delta )
		{
			fill_cpu( Lyr.outputs * Lyr.batch, 0, Lyr.delta, 1 );
		}

		Lyr.forward( Lyr, net );
		net.input = Lyr.output;

		if ( Lyr.truth )
		{
			net.truth = Lyr.output;	// ���� ��°��� ��ǥ������ ����
		}
	}

	calc_network_cost( netp );
}
// �Ű�� ���� ������ ����
void update_network( network *netp )
{
	#ifdef GPU
	if ( netp->gpu_index >= 0 )
	{
		update_network_gpu( netp );
		return;
	}
	#endif

	network net = *netp;

	update_args a	= { 0 };
	a.batch			= net.batch*net.subdivisions;
	a.learning_rate	= get_current_rate( netp );
	a.momentum		= net.momentum;
	a.decay			= net.decay;
	a.adam			= net.adam;
	a.B1			= net.B1;
	a.B2			= net.B2;
	a.eps			= net.eps;

	++*net.t;
	a.t = *net.t;

	int ii;
	for ( ii=0; ii < net.n; ++ii )
	{
		layer Lyr = net.layers[ii];
		if ( Lyr.update )
		{
			Lyr.update( Lyr, a );
		}
	}
}
// 
void calc_network_cost( network *netp )
{
	network net = *netp;

	int ii;
	float sum = 0;
	int count = 0;

	for ( ii=0; ii < net.n; ++ii )
	{
		if ( net.layers[ii].cost )
		{
			sum += net.layers[ii].cost[0];
			++count;
		}
	}

	*net.cost = sum/count;
}

int get_predicted_class_network( network *net )
{
	return max_index( net->output, net->outputs );
}

void backward_network( network *netp )
{
	#ifdef GPU
	if ( netp->gpu_index >= 0 )
	{
		backward_network_gpu( netp );
		return;
	}
	#endif

	network net = *netp;
	network orig = net;

	int ii;
	for ( ii=net.n-1; ii >= 0; --ii )
	{
		layer Lyr = net.layers[ii];

		if ( Lyr.stopbackward ) break;

		if ( ii == 0 )
		{
			net = orig;
		}
		else
		{
			layer prev = net.layers[ii-1];
			net.input = prev.output;
			net.delta = prev.delta;
		}

		net.index = ii;
		Lyr.backward( Lyr, net );
	}
}
// �ϳ��� �����ڷḦ ������
float train_network_datum( network *net )
{
	*net->seen += net->batch;
	net->train = 1;	// �����Ѵٰ� ����(��δ� ������)

	forward_network( net );		// �Է°� ������ ��� �� ����
	backward_network( net );	// ������ ������ ��� �� ����

	float error = *net->cost;

	if ( ( (*net->seen)/net->batch ) % net->subdivisions == 0 )
		update_network( net );	// �Ű�� ���� ������ ����

	return error;
}

float train_network_sgd( network *net, data d, int n )
{
	int batch = net->batch;

	float sum = 0;

	int i;
	for ( i = 0; i < n; ++i )
	{
		get_random_batch( d, batch, net->input, net->truth );
		float err = train_network_datum( net );
		sum += err;
	}

	return (float)sum/(n*batch);
}

float train_network( network *net, data dt )
{
	assert( dt.X.rows % net->batch == 0 );

	int batch = net->batch;
	int nn = dt.X.rows / batch;	// �ѻ縮�� �ڷᰳ��

	float sum = 0;

	int ii;
	for ( ii=0; ii < nn; ++ii )
	{
		// �ѻ縮��ŭ �ڷḦ �����´�
		get_next_batch( dt, batch, ii*batch, net->input, net->truth );

		float err = train_network_datum( net );

		sum += err;
	}

	return (float)sum / (nn*batch);
}

void set_temp_network( network *net, float t )
{
	int i;
	for ( i = 0; i < net->n; ++i )
	{
		net->layers[i].temperature = t;
	}
}


void set_batch_network( network *net, int b )
{
	net->batch = b;
	int i;
	for ( i = 0; i < net->n; ++i )
	{
		net->layers[i].batch = b;

		#ifdef CUDNN
		if ( net->layers[i].type == CONVOLUTIONAL )
		{
			cudnn_convolutional_setup( net->layers + i );
		}

		if ( net->layers[i].type == DECONVOLUTIONAL )
		{
			layer *Lyr = net->layers + i;
			cudnnSetTensor4dDescriptor( Lyr->dstTensorDesc
									, CUDNN_TENSOR_NCHW
									, CUDNN_DATA_FLOAT
									, 1
									, Lyr->out_c
									, Lyr->out_h
									, Lyr->out_w );
			cudnnSetTensor4dDescriptor( Lyr->normTensorDesc
									, CUDNN_TENSOR_NCHW
									, CUDNN_DATA_FLOAT
									, 1
									, Lyr->out_c
									, 1
									, 1 );
		}
		#endif
	}
}

int resize_network( network *net, int w, int h )
{
	#ifdef GPU
	cuda_set_device( net->gpu_index );
	cuda_free( net->workspace );
	#endif

	int ii;
	//if(w == net->w && h == net->h) return 0;
	net->w = w;
	net->h = h;
	int inputs = 0;
	size_t workspace_size = 0;
	//fprintf(stderr, "Resizing to %d x %d...\n", w, h);
	//fflush(stderr);

	for ( ii=0; ii < net->n; ++ii )
	{
		layer Lyr = net->layers[ii];

		if		( Lyr.type == CONVOLUTIONAL )	{	resize_convolutional_layer( &Lyr, w, h );		}
		else if ( Lyr.type == CROP )			{	resize_crop_layer( &Lyr, w, h );		}
		else if ( Lyr.type == MAXPOOL )			{	resize_maxpool_layer( &Lyr, w, h );		}
		else if ( Lyr.type == REGION )			{	resize_region_layer( &Lyr, w, h );		}
		else if ( Lyr.type == YOLO )			{	resize_yolo_layer( &Lyr, w, h );		}
		else if ( Lyr.type == ROUTE )			{	resize_route_layer( &Lyr, net );		}
		else if ( Lyr.type == SHORTCUT )		{	resize_shortcut_layer( &Lyr, w, h );		}
		else if ( Lyr.type == UPSAMPLE )		{	resize_upsample_layer( &Lyr, w, h );		}
		else if ( Lyr.type == REORG )			{	resize_reorg_layer( &Lyr, w, h );		}
		else if ( Lyr.type == AVGPOOL )			{	resize_avgpool_layer( &Lyr, w, h );		}
		else if ( Lyr.type == NORMALIZATION )	{	resize_normalization_layer( &Lyr, w, h );		}
		else if ( Lyr.type == COST )			{	resize_cost_layer( &Lyr, inputs );		}
		else
		{
			error( "Cannot resize this type of layer" );
		}

		if ( Lyr.workspace_size > workspace_size ) workspace_size = Lyr.workspace_size;
		if ( Lyr.workspace_size > 2000000000 ) assert( 0 );

		inputs = Lyr.outputs;
		net->layers[ii] = Lyr;
		w = Lyr.out_w;
		h = Lyr.out_h;

		if ( Lyr.type == AVGPOOL ) break;
	}

	layer out = get_network_output_layer( net );
	net->inputs = net->layers[0].inputs;
	net->outputs = out.outputs;
	net->truths = out.outputs;

	if ( net->layers[net->n-1].truths ) net->truths = net->layers[net->n-1].truths;

	net->output = out.output;
	free( net->input );
	free( net->truth );
	net->input = calloc( net->inputs*net->batch, sizeof( float ) );
	net->truth = calloc( net->truths*net->batch, sizeof( float ) );

	#ifdef GPU
	if ( gpu_index >= 0 )
	{
		cuda_free( net->input_gpu );
		cuda_free( net->truth_gpu );
		net->input_gpu = cuda_make_array( net->input, net->inputs*net->batch );
		net->truth_gpu = cuda_make_array( net->truth, net->truths*net->batch );
		if ( workspace_size )
		{
			net->workspace = cuda_make_array( 0, (workspace_size-1)/sizeof( float )+1 );
		}
	}
	else
	{
		free( net->workspace );
		net->workspace = calloc( 1, workspace_size );
	}
	#else
	free( net->workspace );
	net->workspace = calloc( 1, workspace_size );
	#endif

	//fprintf(stderr, " Done!\n");
	return 0;
}

layer get_network_detection_layer( network *net )
{
	int ii;
	for ( ii=0; ii < net->n; ++ii )
	{
		if ( net->layers[ii].type == DETECTION )
		{
			return net->layers[ii];
		}
	}

	fprintf( stderr, "Detection layer not found!!\n" );
	layer l = { 0 };

	return l;
}

image get_network_image_layer( network *net, int i )
{
	layer Lyr = net->layers[i];

	#ifdef GPU
	//cuda_pull_array(Lyr.output_gpu, Lyr.output, Lyr.outputs);
	#endif

	if ( Lyr.out_w && Lyr.out_h && Lyr.out_c )
	{
		return float_to_image( Lyr.out_w, Lyr.out_h, Lyr.out_c, Lyr.output );
	}

	image def = { 0 };

	return def;
}

image get_network_image( network *net )
{
	int ii;
	for ( ii=net->n-1; ii >= 0; --ii )
	{
		image m = get_network_image_layer( net, ii );
		if ( m.h != 0 ) return m;
	}

	image def = { 0 };

	return def;
}

void top_predictions( network *net, int k, int *index )
{
	top_k( net->output, net->outputs, k, index );
}

// �Ű�� ������ ���
float *network_predict( network *net, float *input )
{
	network orig = *net;	// ������ ��Ƶα�
	net->input	 = input;	// �Է°��ּ� ����
	net->truth	 = 0;		// ��ǥ�� ����
	net->train	 = 0;		// ���þ��� ����
	net->delta	 = 0;		// ������ ����

	forward_network( net );

	float *out	= net->output;	// ��°� �ּ� ��ȯ�� ����
	*net		= orig;			// ��Ƶ� ������ ��������

	return out;
}
// ����� ������ ��ȯ�Ѵ�
int num_detections( network *net, float thresh )
{
	int ii;
	int ss = 0;
	for ( ii=0; ii < net->n; ++ii )	// ���� �ݺ��Ѵ�
	{
		layer Lyr = net->layers[ii];

		if ( Lyr.type == YOLO )
		{
			ss += yolo_num_detections( Lyr, thresh );
		}

		if ( Lyr.type == DETECTION || Lyr.type == REGION )
		{
			ss += Lyr.w*Lyr.h*Lyr.n;
		}
	}

	return ss;
}
// ���ⰳ���� �˾Ƴ���, ������ŭ �޸𸮸� �Ҵ��ϰ�, �Ҵ��� �޸� �ּҸ� ��ȯ�Ѵ�
detection *make_network_boxes( network *net, float thresh, int *num )
{
	layer Lyr = net->layers[net->n - 1];

	int nboxes = num_detections( net, thresh );	// ���ΰ��� �Ѵ� ���ⰳ���� �˾Ƴ���
	if ( num ) *num = nboxes;

	detection *dets = calloc( nboxes, sizeof( detection ) );	// �޸� �Ҵ�

	int ii;
	for ( ii=0; ii < nboxes; ++ii )
	{
		dets[ii].prob = calloc( Lyr.classes, sizeof( float ) );

		if ( Lyr.coords > 4 )
		{
			dets[ii].mask = calloc( Lyr.coords-4, sizeof( float ) );
		}
	}

	return dets;
}

void fill_network_boxes( network *net	// �Ű��
					, int w				// �����ڷ� �ʺ�
					, int h				// �����ڷ� ����
					, float thresh		// ���ΰ�
					, float hier		//
					, int *map			//
					, int relative		//
					, detection *dets )	// ��������
{
	int j;
	for ( j = 0; j < net->n; ++j )
	{
		layer Lyr = net->layers[j];

		if ( Lyr.type == YOLO )
		{
			int count = get_yolo_detections( Lyr, w, h, net->w, net->h, thresh, map, relative, dets );
			dets += count;
		}

		if ( Lyr.type == REGION )
		{
			get_region_detections( Lyr, w, h, net->w, net->h, thresh, map, hier, relative, dets );
			dets += Lyr.w*Lyr.h*Lyr.n;
		}

		if ( Lyr.type == DETECTION )
		{
			get_detection_detections( Lyr, w, h, thresh, dets );
			dets += Lyr.w*Lyr.h*Lyr.n;
		}
	}
}
// ����� �з��� ���ڸ� ����� ��ȯ�Ѵ�
detection *get_network_boxes( network *net	// �Ű��
							, int w			// �����ڷ� �ʺ�
							, int h			// �����ڷ� ����
							, float thresh	// ���ΰ�
							, float hier	//
							, int *map		//
							, int relative	//
							, int *num )	// ���ⰳ��
{
	detection *dets = make_network_boxes( net, thresh, num );	// ���ⰳ���� �˾Ƴ��� �޸� �Ҵ�
	fill_network_boxes( net, w, h, thresh, hier, map, relative, dets );
	return dets;
}

void free_detections( detection *dets, int n )
{
	int i;
	for ( i = 0; i < n; ++i )
	{
		free( dets[i].prob );
		if ( dets[i].mask ) free( dets[i].mask );
	}
	free( dets );
}

float *network_predict_image( network *net, image im )
{
	image imr = letterbox_image( im, net->w, net->h );
	set_batch_network( net, 1 );
	float *p = network_predict( net, imr.data );
	free_image( imr );
	return p;
}

int network_width( network *net )
{
	return net->w;
}
int network_height( network *net )
{
	return net->h;
}

matrix network_predict_data_multi( network *net, data test, int n )
{
	int ii, jj, bb, mm;
	int kk = net->outputs;
	matrix pred = make_matrix( test.X.rows, kk );
	float *X = calloc( net->batch*test.X.rows, sizeof( float ) );

	for ( ii=0; ii < test.X.rows; ii += net->batch )
	{
		for ( bb=0; bb < net->batch; ++bb )
		{
			if ( ii+bb == test.X.rows ) break;
			memcpy( X+bb*test.X.cols, test.X.vals[ii+bb], test.X.cols*sizeof( float ) );
		}

		for ( mm=0; mm < n; ++mm )
		{
			float *out = network_predict( net, X );

			for ( bb=0; bb < net->batch; ++bb )
			{
				if ( ii+bb == test.X.rows ) break;

				for ( jj=0; jj < kk; ++jj )
				{
					pred.vals[ii+bb][jj] += out[jj+bb*kk]/n;
				}
			}
		}
	}

	free( X );

	return pred;
}

matrix network_predict_data( network *net, data test )
{
	int i, j, b;
	int k = net->outputs;

	matrix pred = make_matrix( test.X.rows, k );
	float *X = calloc( net->batch*test.X.cols, sizeof( float ) );

	for ( i = 0; i < test.X.rows; i += net->batch )
	{
		for ( b = 0; b < net->batch; ++b )
		{
			if ( i+b == test.X.rows ) break;
			memcpy( X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof( float ) );
		}

		float *out = network_predict( net, X );

		for ( b = 0; b < net->batch; ++b )
		{
			if ( i+b == test.X.rows ) break;

			for ( j = 0; j < k; ++j )
			{
				pred.vals[i+b][j] = out[j+b*k];
			}
		}
	}

	free( X );

	return pred;
}

void print_network( network *net )
{
	int ii, jj;
	for ( ii=0; ii < net->n; ++ii )
	{
		layer Lyr		= net->layers[ii];
		float *output	= Lyr.output;
		int nn			= Lyr.outputs;
		float mean		= mean_array( output, nn );
		float vari		= variance_array( output, nn );

		fprintf( stderr, "Layer %d - Mean: %f, Variance: %f\n", ii, mean, vari );

		if ( nn > 100 ) nn = 100;

		for ( jj=0; jj < nn; ++jj ) fprintf( stderr, "%f, ", output[jj] );

		if ( nn == 100 )	fprintf( stderr, ".....\n" );

		fprintf( stderr, "\n" );
	}
}

void compare_networks( network *n1, network *n2, data test )
{
	matrix g1 = network_predict_data( n1, test );
	matrix g2 = network_predict_data( n2, test );
	int i;
	int a, b, c, d;
	a = b = c = d = 0;

	for ( i = 0; i < g1.rows; ++i )
	{
		int truth	= max_index( test.y.vals[i], test.y.cols );
		int p1		= max_index( g1.vals[i], g1.cols );
		int p2		= max_index( g2.vals[i], g2.cols );
		if ( p1 == truth )
		{
			if ( p2 == truth ) ++d;
			else ++c;
		}
		else
		{
			if ( p2 == truth ) ++b;
			else ++a;
		}
	}

	printf( "%5d %5d\n%5d %5d\n", a, b, c, d );
	float num = pow( (abs( b - c ) - 1.0), 2.0 );
	float den = b + c;
	printf( "%f\n", num/den );
}

float network_accuracy( network *net, data d )
{
	matrix guess = network_predict_data( net, d );
	float acc = matrix_topk_accuracy( d.y, guess, 1 );
	free_matrix( guess );
	return acc;
}

float *network_accuracies( network *net, data d, int n )
{
	static float acc[2];
	matrix guess = network_predict_data( net, d );
	acc[0] = matrix_topk_accuracy( d.y, guess, 1 );
	acc[1] = matrix_topk_accuracy( d.y, guess, n );
	free_matrix( guess );
	return acc;
}
// �Ű�� ��´��� �ּҸ� �����´�
layer get_network_output_layer( network *net )
{
	int ii;
	for ( ii=net->n-1; ii >= 0; --ii )
	{
		// COST �� �մ��� ���� ��´��̴�...
		if ( net->layers[ii].type != COST )
			break;	// for �� Ż��???
		// COST ���� ��¿����� �ս��� ����ϱ� ���� ���̴�
	}

	return net->layers[ii];
}

float network_accuracy_multi( network *net, data d, int n )
{
	matrix guess = network_predict_data_multi( net, d, n );
	float acc = matrix_topk_accuracy( d.y, guess, 1 );
	free_matrix( guess );
	return acc;
}

void free_network( network *net )
{
	int i;
	for ( i = 0; i < net->n; ++i )
	{
		free_layer( net->layers[i] );
	}
	free( net->layers );

	if ( net->input ) free( net->input );
	if ( net->truth ) free( net->truth );

	#ifdef GPU
	if ( net->input_gpu ) cuda_free( net->input_gpu );
	if ( net->truth_gpu ) cuda_free( net->truth_gpu );
	#endif

	free( net );
}

// Some day...
// ^ What the hell is this comment for?


layer network_output_layer( network *net )
{
	int i;
	for ( i = net->n - 1; i >= 0; --i )
	{
		if ( net->layers[i].type != COST ) break;
	}
	return net->layers[i];
}

int network_inputs( network *net )
{
	return net->layers[0].inputs;
}

int network_outputs( network *net )
{
	return network_output_layer( net ).outputs;
}

float *network_output( network *net )
{
	return network_output_layer( net ).output;
}

#ifdef GPU
// �Ű�� ������ ������ ���
void forward_network_gpu( network *netp )
{
	network net = *netp;
	cuda_set_device( net.gpu_index );
	// ����޸𸮿��� ��ġ�޸𸮷� �Ű���Է°� �迭�� �����Ѵ�
	cuda_push_array( net.input_gpu, net.input, net.inputs*net.batch );

	if ( net.truth )	// ����޸𸮿��� ��ġ�޸𸮷� �Ű����ǥ�� �迭�� �����Ѵ�
	{
		cuda_push_array( net.truth_gpu, net.truth, net.truths*net.batch );
	}

	int ii;
	for ( ii=0; ii < net.n; ++ii )	// �Ű���� ����(��)�� �ݺ��Ѵ�
	{
		net.index = ii;
		layer Lyr = net.layers[ii];

		if ( Lyr.delta_gpu )
		{
			fill_gpu( Lyr.outputs * Lyr.batch, 0, Lyr.delta_gpu, 1 );
		}

		Lyr.forward_gpu( Lyr, net );
		net.input_gpu	= Lyr.output_gpu;
		net.input		= Lyr.output;

		if ( Lyr.truth )
		{
			net.truth_gpu	= Lyr.output_gpu;
			net.truth		= Lyr.output;
		}
	}

	pull_network_output( netp );	// ��ġ�� �Ű�� ��´� ��°��� ���� ��°� �޸𸮷� �ܾ�´�
	calc_network_cost( netp );
}

void backward_network_gpu( network *netp )
{
	int ii;
	network net = *netp;
	network orig = net;
	cuda_set_device( net.gpu_index );

	for ( ii=net.n-1; ii >= 0; --ii )
	{
		layer Lyr = net.layers[ii];

		if ( Lyr.stopbackward )
			break;	// �Ʒ� ����

		if ( ii == 0 )
		{
			net = orig;
		}
		else
		{
			layer prev = net.layers[ii-1];	// ������

			net.input		= prev.output;
			net.delta		= prev.delta;
			net.input_gpu	= prev.output_gpu;
			net.delta_gpu	= prev.delta_gpu;
		}

		net.index = ii;
		Lyr.backward_gpu( Lyr, net );
	}
}
// ������ �ʿ��� �������� �����Ѵ�(�����, �����, ��)
void update_network_gpu( network *netp )
{
	network net = *netp;
	cuda_set_device( net.gpu_index );
	int ii;
	update_args arg	= { 0 };
	arg.batch		= net.batch*net.subdivisions;
	arg.learning_rate = get_current_rate( netp );
	arg.momentum	= net.momentum;
	arg.decay		= net.decay;
	arg.adam		= net.adam;
	arg.B1			= net.B1;
	arg.B2			= net.B2;
	arg.eps			= net.eps;
	++*net.t;	// ����Ƚ�� ����
	arg.t			= (*net.t);

	for ( ii=0; ii < net.n; ++ii )
	{
		layer Lyr = net.layers[ii];
		if ( Lyr.update_gpu )
		{
			Lyr.update_gpu( Lyr, arg );
		}
	}
}

void harmless_update_network_gpu( network *netp )
{
	network net = *netp;
	cuda_set_device( net.gpu_index );
	int i;
	for ( i = 0; i < net.n; ++i )
	{
		layer Lyr = net.layers[i];
		if ( Lyr.weight_updates_gpu )	fill_gpu( Lyr.nweights, 0, Lyr.weight_updates_gpu, 1 );
		if ( Lyr.bias_updates_gpu )		fill_gpu( Lyr.nbiases, 0, Lyr.bias_updates_gpu, 1 );
		if ( Lyr.scale_updates_gpu )	fill_gpu( Lyr.nbiases, 0, Lyr.scale_updates_gpu, 1 );
	}
}

typedef struct
{
	network *net;
	data d;
	float *err;
} train_args;

void *train_thread( void *ptr )
{
	train_args args = *(train_args*)ptr;
	free( ptr );
	cuda_set_device( args.net->gpu_index );
	*args.err = train_network( args.net, args.d );
	return 0;
}
// ������� �Ű�� ����
pthread_t train_network_in_thread( network *net, data dt, float *err )
{
	pthread_t thread;
	train_args *ptr = (train_args *)calloc( 1, sizeof( train_args ) );
	ptr->net	= net;	// �Ű��
	ptr->d		= dt;	// ������ ���, ��ǥ��
	ptr->err	= err;	// ������ ������ �ּ�

	if ( pthread_create( &thread, 0, train_thread, ptr ) )
		//error( "Thread creation failed" );
		error( "���� ������ ��������!!!" );

	return thread;
}

void merge_weights( layer Lyr, layer base )
{
	if ( Lyr.type == CONVOLUTIONAL )
	{
		axpy_cpu( Lyr.n, 1, Lyr.bias_updates, 1, base.biases, 1 );
		axpy_cpu( Lyr.nweights, 1, Lyr.weight_updates, 1, base.weights, 1 );
		if ( Lyr.scales )
		{
			axpy_cpu( Lyr.n, 1, Lyr.scale_updates, 1, base.scales, 1 );
		}
	}
	else if ( Lyr.type == CONNECTED )
	{
		axpy_cpu( Lyr.outputs, 1, Lyr.bias_updates, 1, base.biases, 1 );
		axpy_cpu( Lyr.outputs*Lyr.inputs, 1, Lyr.weight_updates, 1, base.weights, 1 );
	}
}

void scale_weights( layer Lyr, float s )
{
	if ( Lyr.type == CONVOLUTIONAL )
	{
		scal_cpu( Lyr.n, s, Lyr.biases, 1 );
		scal_cpu( Lyr.nweights, s, Lyr.weights, 1 );
		if ( Lyr.scales )
		{
			scal_cpu( Lyr.n, s, Lyr.scales, 1 );
		}
	}
	else if ( Lyr.type == CONNECTED )
	{
		scal_cpu( Lyr.outputs, s, Lyr.biases, 1 );
		scal_cpu( Lyr.outputs*Lyr.inputs, s, Lyr.weights, 1 );
	}
}


void pull_weights( layer Lyr )
{
	if ( Lyr.type == CONVOLUTIONAL || Lyr.type == DECONVOLUTIONAL )
	{
		cuda_pull_array( Lyr.biases_gpu, Lyr.bias_updates, Lyr.n );
		cuda_pull_array( Lyr.weights_gpu, Lyr.weight_updates, Lyr.nweights );
		if ( Lyr.scales ) cuda_pull_array( Lyr.scales_gpu, Lyr.scale_updates, Lyr.n );
	}
	else if ( Lyr.type == CONNECTED )
	{
		cuda_pull_array( Lyr.biases_gpu, Lyr.bias_updates, Lyr.outputs );
		cuda_pull_array( Lyr.weights_gpu, Lyr.weight_updates, Lyr.outputs*Lyr.inputs );
	}
}

void push_weights( layer Lyr )
{
	if ( Lyr.type == CONVOLUTIONAL || Lyr.type == DECONVOLUTIONAL )
	{
		cuda_push_array( Lyr.biases_gpu, Lyr.biases, Lyr.n );
		cuda_push_array( Lyr.weights_gpu, Lyr.weights, Lyr.nweights );
		if ( Lyr.scales ) cuda_push_array( Lyr.scales_gpu, Lyr.scales, Lyr.n );
	}
	else if ( Lyr.type == CONNECTED )
	{
		cuda_push_array( Lyr.biases_gpu, Lyr.biases, Lyr.outputs );
		cuda_push_array( Lyr.weights_gpu, Lyr.weights, Lyr.outputs*Lyr.inputs );
	}
}

void distribute_weights( layer Lyr, layer base )
{
	if ( Lyr.type == CONVOLUTIONAL || Lyr.type == DECONVOLUTIONAL )
	{
		cuda_push_array( Lyr.biases_gpu, base.biases, Lyr.n );
		cuda_push_array( Lyr.weights_gpu, base.weights, Lyr.nweights );
		if ( base.scales ) cuda_push_array( Lyr.scales_gpu, base.scales, Lyr.n );
	}
	else if ( Lyr.type == CONNECTED )
	{
		cuda_push_array( Lyr.biases_gpu, base.biases, Lyr.outputs );
		cuda_push_array( Lyr.weights_gpu, base.weights, Lyr.outputs*Lyr.inputs );
	}
}

/*
void pull_updates( layer l )
{
	if ( Lyr.type == CONVOLUTIONAL )
	{
		cuda_pull_array( Lyr.bias_updates_gpu, Lyr.bias_updates, Lyr.n );
		cuda_pull_array( Lyr.weight_updates_gpu, Lyr.weight_updates, Lyr.nweights );
		if ( Lyr.scale_updates ) cuda_pull_array( Lyr.scale_updates_gpu, Lyr.scale_updates, Lyr.n );
	}
	else if ( Lyr.type == CONNECTED )
	{
		cuda_pull_array( Lyr.bias_updates_gpu, Lyr.bias_updates, Lyr.outputs );
		cuda_pull_array( Lyr.weight_updates_gpu, Lyr.weight_updates, Lyr.outputs*Lyr.inputs );
	}
}

void push_updates( layer l )
{
	if ( Lyr.type == CONVOLUTIONAL )
	{
		cuda_push_array( Lyr.bias_updates_gpu, Lyr.bias_updates, Lyr.n );
		cuda_push_array( Lyr.weight_updates_gpu, Lyr.weight_updates, Lyr.nweights );
		if ( Lyr.scale_updates ) cuda_push_array( Lyr.scale_updates_gpu, Lyr.scale_updates, Lyr.n );
	}
	else if ( Lyr.type == CONNECTED )
	{
		cuda_push_array( Lyr.bias_updates_gpu, Lyr.bias_updates, Lyr.outputs );
		cuda_push_array( Lyr.weight_updates_gpu, Lyr.weight_updates, Lyr.outputs*Lyr.inputs );
	}
}

void update_layer( layer l, network net )
{
	int update_batch = net.batch*net.subdivisions;
	float rate = get_current_rate( net );
	Lyr.t = get_current_batch( net );
	if ( Lyr.update_gpu )
	{
		Lyr.update_gpu( l, update_batch, rate*Lyr.learning_rate_scale, net.momentum, net.decay );
	}
}
void merge_updates( layer l, layer base )
{
	if ( Lyr.type == CONVOLUTIONAL )
	{
		axpy_cpu( Lyr.n, 1, Lyr.bias_updates, 1, base.bias_updates, 1 );
		axpy_cpu( Lyr.nweights, 1, Lyr.weight_updates, 1, base.weight_updates, 1 );
		if ( Lyr.scale_updates )
		{
			axpy_cpu( Lyr.n, 1, Lyr.scale_updates, 1, base.scale_updates, 1 );
		}
	}
	else if ( Lyr.type == CONNECTED )
	{
		axpy_cpu( Lyr.outputs, 1, Lyr.bias_updates, 1, base.bias_updates, 1 );
		axpy_cpu( Lyr.outputs*Lyr.inputs, 1, Lyr.weight_updates, 1, base.weight_updates, 1 );
	}
}

void distribute_updates( layer l, layer base )
{
	if ( Lyr.type == CONVOLUTIONAL || Lyr.type == DECONVOLUTIONAL )
	{
		cuda_push_array( Lyr.bias_updates_gpu, base.bias_updates, Lyr.n );
		cuda_push_array( Lyr.weight_updates_gpu, base.weight_updates, Lyr.nweights );
		if ( base.scale_updates ) cuda_push_array( Lyr.scale_updates_gpu, base.scale_updates, Lyr.n );
	}
	else if ( Lyr.type == CONNECTED )
	{
		cuda_push_array( Lyr.bias_updates_gpu, base.bias_updates, Lyr.outputs );
		cuda_push_array( Lyr.weight_updates_gpu, base.weight_updates, Lyr.outputs*Lyr.inputs );
	}
}
*/
void sync_layer( network **nets, int nn, int jj )
{
	int ii;
	network *net	= nets[0];
	layer base		= net->layers[jj];

	scale_weights( base, 0 );

	for ( ii=0; ii < nn; ++ii )
	{
		cuda_set_device( nets[ii]->gpu_index );
		layer ll = nets[ii]->layers[jj];
		pull_weights( ll );
		merge_weights( ll, base );
	}

	scale_weights( base, 1.0f/nn );

	for ( ii=0; ii < nn; ++ii )
	{
		cuda_set_device( nets[ii]->gpu_index );
		layer ll = nets[ii]->layers[jj];
		distribute_weights( ll, base );
	}
}

typedef struct
{
	network **nets;
	int n;	// GPU ����
	int j;	// �� ����
} sync_args;

void *sync_layer_thread( void *ptr )
{
	sync_args args = *(sync_args*)ptr;
	sync_layer( args.nets, args.n, args.j );
	free( ptr );

	return 0;
}

pthread_t sync_layer_in_thread( network **nets, int nn, int jj )
{
	pthread_t thread;
	sync_args *ptr = (sync_args *)calloc( 1, sizeof( sync_args ) );
	ptr->nets	= nets;
	ptr->n		= nn;	// GPU ����
	ptr->j		= jj;	// �� ����

	if ( pthread_create( &thread, 0, sync_layer_thread, ptr ) )
		//error( "Thread creation failed" );
		error( "������ ��������!!!" );

	return thread;
}

void sync_nets( network **nets
			, int nn		// GPU ����
			, int interval )
{
	int jj;
	int layers = nets[0]->n;
	pthread_t *threads = (pthread_t *)calloc( layers, sizeof( pthread_t ) );

	*(nets[0]->seen) += interval * (nn-1) * nets[0]->batch * nets[0]->subdivisions;

	for ( jj=0; jj < nn; ++jj )
	{
		*(nets[jj]->seen) = *(nets[0]->seen);
	}

	for ( jj=0; jj < layers; ++jj )
	{
		threads[jj] = sync_layer_in_thread( nets, nn, jj );
	}

	for ( jj=0; jj < layers; ++jj )
	{
		pthread_join( threads[jj], 0 );
	}

	free( threads );
}
// ����GPU�� ���߽Ű�� ����
float train_networks( network **nets
					, int nn	// GPU ����
					, data dt
					, int interval )
{
	int batch			= nets[0]->batch;
	int subdivisions	= nets[0]->subdivisions;
	assert( batch * subdivisions * nn == dt.X.rows );

	pthread_t *threads	= (pthread_t *)calloc( nn, sizeof( pthread_t ) );
	float *errors		= (float *)calloc( nn, sizeof( float ) );

	int ii;
	float sum = 0;
	for ( ii=0; ii < nn; ++ii )
	{
		data dp		= get_data_part( dt, ii, nn );
		threads[ii]	= train_network_in_thread( nets[ii], dp, errors + ii );
	}

	for ( ii=0; ii < nn; ++ii )
	{
		pthread_join( threads[ii], 0 );
		//printf("%f\n", errors[i]);
		sum += errors[ii];
	}

	//cudaDeviceSynchronize();

	if ( get_current_batch( nets[0] ) % interval == 0 )
	{
		//printf( "Syncing... " );
		printf( "GPU ����ȭ��... " );
		fflush( stdout );
		sync_nets( nets, nn, interval );
		//printf( "Done!\n" );
		printf( "GPU ����ȭ �Ϸ�!\n" );
	}

	//cudaDeviceSynchronize();

	free( threads );
	free( errors );

	return (float)sum/(nn);
}
// ��ġ�� �Ű�� ��´� ��°��� ���� ��°� �޸𸮷� �ܾ�´�
void pull_network_output( network *net )
{
	layer Lyr = get_network_output_layer( net );	// ��´��ּҸ� �����´�
	cuda_pull_array( Lyr.output_gpu, Lyr.output, Lyr.outputs*Lyr.batch );
}

// �Ű�� ���ܿ� ���� ��ġ�� �Ű�� ��°��� ���� ��°� �޸𸮷� �ܾ�´�
void pull_network_output_MoDu( network *net )
{
	int ii;
	for ( ii=0; ii<net->n; ++ii )
	{
		layer Lyr = net->layers[ii];	// ��´��ּҸ� �����´�
		cuda_pull_array( Lyr.output_gpu, Lyr.output, Lyr.outputs*Lyr.batch );
	}
}

void visualize_network( network *net )
{
	image *prev = 0;
	int ii;
	char buff[256];

	for ( ii=0; ii < net->n; ++ii )
	{
		layer Lyr = net->layers[ii];

		if ( Lyr.type == CONVOLUTIONAL )
		{
			//sprintf( buff
			sprintf_s( buff, 256
				, "���߰�- %d��(%s)-��� %d-���� %dx%dx%d"
				, ii, get_layer_string( Lyr.type ), Lyr.c, Lyr.size, Lyr.size, Lyr.n );	// â �̸�

			prev = visualize_convolutional_layer_weight( Lyr, buff, prev );
			prev = Lyr.BoJa_MuGeGab( Lyr, buff, prev );
		}
		else if ( Lyr.type == CONNECTED )
		{
			//sprintf( buff
			sprintf_s( buff, 256
				, "���߰�- %d��(%s)-��� %d-���� %dx%dx%d"
				, ii, get_layer_string( Lyr.type ), Lyr.c, Lyr.out_w, Lyr.out_h, Lyr.out_c );	// â �̸�

			//prev = visualize_connected_layer_weight( Lyr, buff, prev );
			prev = Lyr.BoJa_MuGeGab( Lyr, buff, prev );
		}
	}
}

// �Ű�� ���߰��� �ð�ȭ�Ͽ� ������
void visualize( char *cfgfile, char *weightfile )
{
	network *net = load_network( cfgfile, weightfile, 0 );

	visualize_network( net );

	#ifdef OPENCV
	cvWaitKey( 0 );
	#endif
}
//������ �Է°��� ���Ǿ� �����´�
image pull_SaBi_HanPan( int ww, int hh, int nn, float *SaBi )
{
	int cc = 1;				// 
	int bo = ww*hh*nn;		// ������ ��
	return float_to_image( ww, hh, cc, SaBi + bo );
}

image *pull_SaBi_image( network *net )
{
	image *out = calloc( net->c, sizeof( image ) );	// ������ ������ŭ �̹����޸� �Ҵ�

	int ii;
	// �����̹��� �ǰ��� �ݺ�
	for ( ii=0; ii < net->c; ++ii )
	{
		// ��Ƶ� �޸𸮸� �Ҵ��ϰ�, �ڷᰪ�� �����ϰ�, ��Ƶ� �ּҸ� ��ȯ�Ѵ�
		out[ii] = copy_image( pull_SaBi_HanPan( net->w, net->h, ii, net->input ) );
		normalize_image( out[ii] );	//�����Ǻ��� ���⸦ �ϸ� Ư¡ǥ���� �Ǵ°�???
	}

	return out;
}

image *visualize_SaBi_image( network *net, char *window, image *prev_out )
{
	// ������ ������ŭ �̹����޸𸮸� �Ҵ��ϰ� ��Ƶ� �ּҸ� �����Ѵ�
	image *single_out = pull_SaBi_image( net );
	// �̹����� ���簢������ �迭�� �����ϰ� ȭ�鿡 �����ش�
	show_images( single_out, net->c, window );

	char buff[256];

	sprintf_s( buff, 256, "%s: ���", window );

	return single_out;

}
// �Ű�� ��°��� �ð�ȭ�Ͽ� ������
void visualize_output( network *net )
{
	#ifdef GPU
	pull_network_output_MoDu( net );

	#endif

	image *prev = 0;
	int ii;
	char buff[256];

	// ���� �̹��� ���
	sprintf_s( buff, 256
		, "���°�-����-%dx%dx%d(����x����x����)"
		, net->w, net->h, net->c );	// â �̸�

	prev = visualize_SaBi_image( net, buff, prev );

	for ( ii=0; ii < net->n; ++ii )
	{
		layer Lyr = net->layers[ii];

		if ( Lyr.type == CROP )
		{
			sprintf_s( buff, 256
				, "���°�-%d��(%s)-%dx%dx%d(����x����x����)"
				, ii, get_layer_string( Lyr.type ), Lyr.out_w, Lyr.out_h, Lyr.out_c );	// â �̸�

			//prev = visualize_crop_layer_output( Lyr, buff, prev );
			prev = Lyr.BoJa_NaOnGab( Lyr, buff, prev );
		}
		else if ( Lyr.type == CONVOLUTIONAL )
		{
			sprintf_s( buff, 256
				, "���°�-%d��(%s)-%dx%dx%d(����x����x����)"
				, ii, get_layer_string( Lyr.type ), Lyr.out_w, Lyr.out_h, Lyr.out_c );	// â �̸�

			//prev = visualize_convolutional_layer_output( Lyr, buff, prev );
			prev = Lyr.BoJa_NaOnGab( Lyr, buff, prev );
		}
		else if ( Lyr.type == MAXPOOL )
		{
			sprintf_s( buff, 256
				, "���°�-%d��(%s)-%dx%dx%d(����x����x����)"
				, ii, get_layer_string( Lyr.type ), Lyr.out_w, Lyr.out_h, Lyr.out_c );	// â �̸�

			//prev = visualize_maxpool_layer_output( Lyr, buff, prev );
			prev = Lyr.BoJa_NaOnGab( Lyr, buff, prev );
		}
		else if ( Lyr.type == YOLO )
		{
			sprintf_s( buff, 256
				, "���°�-%d��(%s)-%dx%dx%d(����x����x����)"
				, ii, get_layer_string( Lyr.type ), Lyr.out_w, Lyr.out_h, Lyr.out_c );	// â �̸�

			//prev = visualize_yolo_layer_output( Lyr, buff, prev );
			prev = Lyr.BoJa_NaOnGab( Lyr, buff, prev );
		}
		else if ( Lyr.type == CONNECTED )
		{
			//sprintf( buff
			sprintf_s( buff, 256
				, "���°�-%d��(%s)-%dx%dx%d(����x����x����)"
				, ii, get_layer_string( Lyr.type ), Lyr.out_w, Lyr.out_h, Lyr.out_c );	// â �̸�

			//prev = visualize_connected_layer_output( Lyr, buff, prev );
			prev = Lyr.BoJa_NaOnGab( Lyr, buff, prev );
		}
		else if ( Lyr.type == SOFTMAX )
		{
			//sprintf( buff
			sprintf_s( buff, 256
				, "���°�-%d��(%s)-%dx%dx%d(����x����x����)"
				, ii, get_layer_string( Lyr.type ), Lyr.w, Lyr.h, Lyr.outputs );	// â �̸�

			//prev = visualize_soft_layer_output( Lyr, buff, prev );
			prev = Lyr.BoJa_NaOnGab( Lyr, buff, prev );
		}
		/*else if ( Lyr.type == COST )
		{
			//sprintf( buff
			sprintf_s( buff, 256
				, "���°�-%d��(%s)-%dx%dx%d(����x����x����)"
				, ii, get_layer_string( Lyr.type ), 1, 1, Lyr.outputs );	// â �̸�

			//prev = visualize_cost_layer_output( Lyr, buff, prev );
			prev = Lyr.BoJa_NaOnGab( Lyr, buff, prev );
		}*/
	}

}

// �Ű�� ���߰��� �ð�ȭ�Ͽ� ������
void visualize_out( int argc, char **argv )
{
	char *datacfg	= argv[3];						// �ڷᱸ�� ����
	char *cfgfile	= argv[4];						// �Ű������ �����̸�
	char *weightfile = (argc > 4) ? argv[5] : 0;	// ���߰� �����̸�
	char *filename	= (argc > 5) ? argv[6] : 0;		// ���⿡ ����� ����

	float thresh	= find_float_arg( argc, argv, "-thresh", 0.2f );	// ���ΰ�
	list *options	= read_data_cfg( datacfg );							// �ڷᱸ�� ���
	int classes		= option_find_int( options, "classes", 20 );		// �з���� ����

	char *eval		= option_find_str( options, "eval", "coco" );
	if ( strcmp( eval, "mnist") != 0 )
	{
		char *name_list	= option_find_str( options, "names", "data/names.list" );	// �з���� �����̸�
		char **names	= get_labels( name_list );									// �з��̸� ���
	}

	//char *train_list	= option_find_str( options, "train", "data/train.list" );
	char *train_list	= option_find_str( options, "valid", "data/valid.list" );

	image **alphabet = load_alphabet();		// ���ڱ׸� ���
//	list *plist		= get_paths( train_list );
//	char **paths	= (char **)list_to_array( plist );

	network *net	= load_network( cfgfile, weightfile, 0 );
	//net->batch		= 1;

	data mnistData;
	data *mnist	= NULL;

	if ( net->JaRyoJong == MNIST_DATA )
	{
//		mnistData	= make_data_mnist( paths );
		mnist		= &mnistData;
	}

	layer Lyr		= net->layers[net->n-1];

	set_batch_network( net, 1 );
	srand( 2222222 );
	clock_t time;
	char buff[256];
	char *input	= buff;
	float nms	= 0.4f;

	float *NaOnGab;

	while ( 1 )
	{
		float *X;

		if ( net->JaRyoJong == MNIST_DATA )
		{
			//int WiChi = (unsigned int)(UNIFORM_ZERO_THRU_ONE * net->max_batches);;
			int WiChi = (unsigned int)(UNIFORM_ZERO_THRU_ONE * mnistData.X.rows);;

			X = mnistData.X.vals[WiChi];
			printf( "�о�� �̹��� ��ȣ: %d, ��ǥ��: %d, ", WiChi, mnistData.labels[WiChi] );
		}
		else
		{
			if ( filename )
			{
				//strncpy( input, filename, 256 );
				strncpy_s( input, 256, filename, 256 );
			}
			else
			{
				printf( "����̹��� ���ϸ� �Է�: " );
				fflush( stdout );
				input = fgets( input, 256, stdin );

				if ( !input ) return;

				//strtok( input, "\n" );
				char *NaMeoJi;
				strtok_s( input, "\n", &NaMeoJi );
			}

			image im	= load_image_color( input, 0, 0 );
			image sized	= resize_image( im, net->w, net->h );
			X	= sized.data;
		}

		time = clock();
		float *NaOnGab = network_predict( net, X );

		int YeJiGab	= -1;
		float GoGab	= -999.0;

		int ii;
		for ( ii=0; ii<net->outputs; ++ii )
		{
			if ( GoGab < NaOnGab[ii] )
			{
				GoGab	= NaOnGab[ii];
				YeJiGab	= ii;
			}
		}

		printf( "������: %d\n", YeJiGab );

		net->input	 = X;	// ���������� ����� ������� ������

		visualize_output( net );

/*		int nboxes = 0;
		detection *dets = get_network_boxes( net, 1, 1, thresh, 0, 0, 0, &nboxes );

		if ( nms ) do_nms_sort( dets, Lyr.side*Lyr.side*Lyr.n, Lyr.classes, nms );

		draw_detections( im, dets, Lyr.side*Lyr.side*Lyr.n, thresh, names, alphabet, classes );
		save_image( im, "�������" );
		show_image( im, "�������" );
		free_detections( dets, nboxes );
		free_image( im );
		free_image( sized );
*/
		#ifdef OPENCV
		cvWaitKey( 0 );
		//cvWaitKey( 2000 );
		cvDestroyAllWindows();
		#endif

		if ( net->JaRyoJong != MNIST_DATA )
		{
			if ( filename ) break;
		}
	}

	if ( mnist )
		free_data( mnistData );
}

void run_visualize( int argc, char **argv )
{
	if ( argc < 4 )
	{
		fprintf( stderr
			, "�����: %s %s [���°�/���߰�/���Ⱚ] [�ڷᱸ������] [����������] [���߰�����(���û���)]\n"
			, argv[0], argv[1] );
		return;
	}

	char *cfg		= argv[4];					// �Ű������ �����̸�
	char *weights	= (argc > 4) ? argv[5] : 0;	// ���߰� �����̸�

	if		( 0==strcmp( argv[2], "���°�" ) )	visualize_out( argc, argv );	// �Ű�� ��°��� �ð�ȭ�Ͽ� ������
	else if ( 0==strcmp( argv[2], "���߰�" ) )	visualize( cfg, weights );		// �Ű�� ���߰��� �ð�ȭ�Ͽ� ������
	else
	{
		fprintf( stderr
			, "%s %s ���� �����׸� \"%s\"�� ���� �����ȵ�!...\n"
			, argv[0], argv[1], argv[2] );
	}

}


#endif
