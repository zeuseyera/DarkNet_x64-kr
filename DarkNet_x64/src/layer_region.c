#include "layer_region.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_region_layer( int batch
					, int w
					, int h
					, int n
					, int classes
					, int coords )
{
	layer Lyr = { 0 };
	Lyr.type = REGION;

	Lyr.n		= n;
	Lyr.batch	= batch;
	Lyr.h		= h;
	Lyr.w		= w;
	Lyr.c		= n*(classes + coords + 1);
	Lyr.out_w	= Lyr.w;
	Lyr.out_h	= Lyr.h;
	Lyr.out_c	= Lyr.c;
	Lyr.classes	= classes;
	Lyr.coords	= coords;
	Lyr.cost		= calloc( 1, sizeof( float ) );
	Lyr.biases		= calloc( n*2, sizeof( float ) );
	Lyr.bias_updates = calloc( n*2, sizeof( float ) );
	Lyr.outputs		= h*w*n*(classes + coords + 1);
	Lyr.inputs		= Lyr.outputs;
	Lyr.truths		= 30 * (Lyr.coords + 1);
	Lyr.delta		= calloc( batch*Lyr.outputs, sizeof( float ) );
	Lyr.output		= calloc( batch*Lyr.outputs, sizeof( float ) );

	int i;
	for ( i = 0; i < n*2; ++i )
	{
		Lyr.biases[i] = 0.5f;
	}

	Lyr.forward		= forward_region_layer;
	Lyr.backward	= backward_region_layer;

	#ifdef GPU
	Lyr.forward_gpu	= forward_region_layer_gpu;
	Lyr.backward_gpu = backward_region_layer_gpu;
	Lyr.output_gpu	= cuda_make_array( Lyr.output, batch*Lyr.outputs );
	Lyr.delta_gpu	= cuda_make_array( Lyr.delta, batch*Lyr.outputs );
	#endif

	fprintf( stderr, "detection\n" );
	srand( 0 );

	return Lyr;
}

void resize_region_layer( layer *Lyr, int w, int h )
{
	Lyr->w = w;
	Lyr->h = h;

	Lyr->outputs	= h*w*Lyr->n*(Lyr->classes + Lyr->coords + 1);
	Lyr->inputs		= Lyr->outputs;

	Lyr->output		= realloc( Lyr->output, Lyr->batch*Lyr->outputs*sizeof( float ) );
	Lyr->delta		= realloc( Lyr->delta, Lyr->batch*Lyr->outputs*sizeof( float ) );

	#ifdef GPU
	cuda_free( Lyr->delta_gpu );
	cuda_free( Lyr->output_gpu );

	Lyr->delta_gpu	= cuda_make_array( Lyr->delta, Lyr->batch*Lyr->outputs );
	Lyr->output_gpu	= cuda_make_array( Lyr->output, Lyr->batch*Lyr->outputs );
	#endif
}

box get_region_box( float *x
				, float *biases
				, int n
				, int index
				, int i
				, int j
				, int w
				, int h
				, int stride )
{
	box b;
	b.x = (i + x[index + 0*stride]) / w;
	b.y = (j + x[index + 1*stride]) / h;
	b.w = exp( x[index + 2*stride] ) * biases[2*n]   / w;
	b.h = exp( x[index + 3*stride] ) * biases[2*n+1] / h;
	return b;
}

float delta_region_box( box truth
					, float *x
					, float *biases
					, int n
					, int index
					, int i
					, int j
					, int w
					, int h
					, float *delta
					, float scale
					, int stride )
{
	box pred = get_region_box( x, biases, n, index, i, j, w, h, stride );
	float iou = box_iou( pred, truth );

	float tx = (truth.x*w - i);
	float ty = (truth.y*h - j);
	float tw = log( truth.w*w / biases[2*n] );
	float th = log( truth.h*h / biases[2*n + 1] );

	delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
	delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
	delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
	delta[index + 3*stride] = scale * (th - x[index + 3*stride]);

	return iou;
}

void delta_region_mask( float *truth
					, float *x
					, int n
					, int index
					, float *delta
					, int stride
					, int scale )
{
	int i;
	for ( i = 0; i < n; ++i )
	{
		delta[index + i*stride] = scale*(truth[i] - x[index + i*stride]);
	}
}


void delta_region_class( float *output
					, float *delta
					, int index
					, int class
					, int classes
					, tree *hier
					, float scale
					, int stride
					, float *avg_cat
					, int tag )
{
	int i, n;
	if ( hier )
	{
		float pred = 1;
		while ( class >= 0 )
		{
			pred *= output[index + stride*class];
			int g = hier->group[class];
			int offset = hier->group_offset[g];

			for ( i = 0; i < hier->group_size[g]; ++i )
			{
				delta[index + stride*(offset + i)] = scale * (0 - output[index + stride*(offset + i)]);
			}

			delta[index + stride*class] = scale * (1 - output[index + stride*class]);

			class = hier->parent[class];
		}

		*avg_cat += pred;
	}
	else
	{
		if ( delta[index] && tag )
		{
			delta[index + stride*class] = scale * (1 - output[index + stride*class]);
			return;
		}
		for ( n = 0; n < classes; ++n )
		{
			delta[index + stride*n] = scale * (((n == class) ? 1 : 0) - output[index + stride*n]);

			if ( n == class ) *avg_cat += output[index + stride*n];
		}
	}
}

float logit( float x )
{
	return log( x/(1.-x) );
}

float tisnan( float x )
{
	return (x != x);
}

int entry_index( layer Lyr, int batch, int location, int entry )
{
	int n	= location / (Lyr.w*Lyr.h);
	int loc	= location % (Lyr.w*Lyr.h);
	return batch*Lyr.outputs
		+ n*Lyr.w*Lyr.h*(Lyr.coords+Lyr.classes+1)
		+ entry*Lyr.w*Lyr.h
		+ loc;
}

void forward_region_layer( const layer Lyr, network net )
{
	int i, j, b, t, n;
	memcpy( Lyr.output, net.input, Lyr.outputs*Lyr.batch*sizeof( float ) );

	#ifndef GPU
	for ( b = 0; b < Lyr.batch; ++b )
	{
		for ( n = 0; n < Lyr.n; ++n )
		{
			int index = entry_index( Lyr, b, n*Lyr.w*Lyr.h, 0 );

			activate_array( Lyr.output + index, 2*Lyr.w*Lyr.h, LOGISTIC );

			index = entry_index( Lyr, b, n*Lyr.w*Lyr.h, Lyr.coords );

			if ( !Lyr.background )
				activate_array( Lyr.output + index, Lyr.w*Lyr.h, LOGISTIC );

			index = entry_index( Lyr, b, n*Lyr.w*Lyr.h, Lyr.coords + 1 );

			if ( !Lyr.softmax && !Lyr.softmax_tree )
				activate_array( Lyr.output + index, Lyr.classes*Lyr.w*Lyr.h, LOGISTIC );
		}
	}

	if ( Lyr.softmax_tree )
	{
		int i;
		int count = Lyr.coords + 1;

		for ( i = 0; i < Lyr.softmax_tree->groups; ++i )
		{
			int group_size = Lyr.softmax_tree->group_size[i];

			softmax_cpu( net.input + count
					, group_size
					, Lyr.batch
					, Lyr.inputs
					, Lyr.n*Lyr.w*Lyr.h
					, 1
					, Lyr.n*Lyr.w*Lyr.h
					, Lyr.temperature
					, Lyr.output + count );

			count += group_size;
		}
	}
	else if ( Lyr.softmax )
	{
		int index = entry_index( Lyr, 0, 0, Lyr.coords + !Lyr.background );

		softmax_cpu( net.input + index
					, Lyr.classes + Lyr.background
					, Lyr.batch*Lyr.n
					, Lyr.inputs/Lyr.n
					, Lyr.w*Lyr.h
					, 1
					, Lyr.w*Lyr.h
					, 1
					, Lyr.output + index );
	}
	#endif

	memset( Lyr.delta, 0, Lyr.outputs * Lyr.batch * sizeof( float ) );

	if ( !net.train ) return;

	float avg_iou = 0;
	float recall = 0;
	float avg_cat = 0;
	float avg_obj = 0;
	float avg_anyobj = 0;
	int count = 0;
	int class_count = 0;
	*(Lyr.cost) = 0;

	for ( b = 0; b < Lyr.batch; ++b )
	{
		if ( Lyr.softmax_tree )
		{
			int onlyclass = 0;
			for ( t = 0; t < 30; ++t )
			{
				box truth	= float_to_box( net.truth + t*(Lyr.coords + 1) + b*Lyr.truths, 1 );
				if ( !truth.x ) break;
				int class	= net.truth[t*(Lyr.coords + 1) + b*Lyr.truths + Lyr.coords];
				float maxp	= 0;
				int maxi	= 0;

				if ( truth.x > 100000 && truth.y > 100000 )
				{
					for ( n = 0; n < Lyr.n*Lyr.w*Lyr.h; ++n )
					{
						int class_index	= entry_index( Lyr, b, n, Lyr.coords + 1 );
						int obj_index	= entry_index( Lyr, b, n, Lyr.coords );
						float scale		=  Lyr.output[obj_index];
						Lyr.delta[obj_index] = Lyr.noobject_scale * (0 - Lyr.output[obj_index]);

						float p = scale * get_hierarchy_probability( Lyr.output + class_index
																, Lyr.softmax_tree
																, class
																, Lyr.w*Lyr.h );

						if ( p > maxp )
						{
							maxp = p;
							maxi = n;
						}
					}

					int class_index	= entry_index( Lyr, b, maxi, Lyr.coords + 1 );
					int obj_index	= entry_index( Lyr, b, maxi, Lyr.coords );

					delta_region_class( Lyr.output
									, Lyr.delta
									, class_index
									, class
									, Lyr.classes
									, Lyr.softmax_tree
									, Lyr.class_scale
									, Lyr.w*Lyr.h
									, &avg_cat
									, !Lyr.softmax );

					if ( Lyr.output[obj_index] < 0.3f )
						Lyr.delta[obj_index] = Lyr.object_scale * (.3 - Lyr.output[obj_index]);
					else  Lyr.delta[obj_index] = 0;

					Lyr.delta[obj_index] = 0;
					++class_count;
					onlyclass = 1;
					break;
				}
			}

			if ( onlyclass ) continue;
		}

		for ( j = 0; j < Lyr.h; ++j )
		{
			for ( i = 0; i < Lyr.w; ++i )
			{
				for ( n = 0; n < Lyr.n; ++n )
				{
					int box_index = entry_index( Lyr
											, b
											, n*Lyr.w*Lyr.h + j*Lyr.w + i
											, 0 );

					box pred = get_region_box( Lyr.output
											, Lyr.biases
											, n
											, box_index
											, i
											, j
											, Lyr.w
											, Lyr.h
											, Lyr.w*Lyr.h );

					float best_iou = 0;

					for ( t = 0; t < 30; ++t )
					{
						box truth = float_to_box( net.truth + t*(Lyr.coords + 1) + b*Lyr.truths, 1 );

						if ( !truth.x ) break;

						float iou = box_iou( pred, truth );

						if ( iou > best_iou )
						{
							best_iou = iou;
						}
					}

					int obj_index = entry_index( Lyr
											, b
											, n*Lyr.w*Lyr.h + j*Lyr.w + i
											, Lyr.coords );

					avg_anyobj += Lyr.output[obj_index];
					Lyr.delta[obj_index] = Lyr.noobject_scale * (0 - Lyr.output[obj_index]);

					if ( Lyr.background )
						Lyr.delta[obj_index] = Lyr.noobject_scale * (1 - Lyr.output[obj_index]);

					if ( best_iou > Lyr.thresh )
					{
						Lyr.delta[obj_index] = 0;
					}

					if ( *(net.seen) < 12800 )
					{
						box truth = { 0 };
						truth.x = (i + 0.5)/Lyr.w;
						truth.y = (j + 0.5)/Lyr.h;
						truth.w = Lyr.biases[2*n]/Lyr.w;
						truth.h = Lyr.biases[2*n+1]/Lyr.h;

						delta_region_box( truth
										, Lyr.output
										, Lyr.biases
										, n
										, box_index
										, i
										, j
										, Lyr.w
										, Lyr.h
										, Lyr.delta
										, 0.01
										, Lyr.w*Lyr.h );
					}
				}
			}
		}

		for ( t = 0; t < 30; ++t )
		{
			box truth = float_to_box( net.truth + t*(Lyr.coords + 1) + b*Lyr.truths, 1 );

			if ( !truth.x ) break;
			float best_iou = 0;
			int best_n = 0;
			i = (truth.x * Lyr.w);
			j = (truth.y * Lyr.h);
			box truth_shift = truth;
			truth_shift.x = 0;
			truth_shift.y = 0;

			for ( n = 0; n < Lyr.n; ++n )
			{
				int box_index = entry_index( Lyr
										, b
										, n*Lyr.w*Lyr.h + j*Lyr.w + i
										, 0 );
				box pred = get_region_box( Lyr.output
										, Lyr.biases
										, n
										, box_index
										, i
										, j
										, Lyr.w
										, Lyr.h
										, Lyr.w*Lyr.h );

				if ( Lyr.bias_match )
				{
					pred.w = Lyr.biases[2*n]/Lyr.w;
					pred.h = Lyr.biases[2*n+1]/Lyr.h;
				}

				pred.x = 0;
				pred.y = 0;
				float iou = box_iou( pred, truth_shift );

				if ( iou > best_iou )
				{
					best_iou = iou;
					best_n = n;
				}
			}

			int box_index = entry_index( Lyr
									, b
									, best_n*Lyr.w*Lyr.h + j*Lyr.w + i
									, 0 );

			float iou = delta_region_box( truth
										, Lyr.output
										, Lyr.biases
										, best_n
										, box_index
										, i
										, j
										, Lyr.w
										, Lyr.h
										, Lyr.delta
										, Lyr.coord_scale *  (2 - truth.w*truth.h)
										, Lyr.w*Lyr.h );

			if ( Lyr.coords > 4 )
			{
				int mask_index = entry_index( Lyr
									, b
									, best_n*Lyr.w*Lyr.h + j*Lyr.w + i
									, 4 );

				delta_region_mask( net.truth + t*(Lyr.coords + 1) + b*Lyr.truths + 5
								, Lyr.output
								, Lyr.coords - 4
								, mask_index
								, Lyr.delta
								, Lyr.w*Lyr.h
								, Lyr.mask_scale );
			}

			if ( iou > .5 ) recall += 1;
			avg_iou += iou;

			int obj_index = entry_index( Lyr
									, b
									, best_n*Lyr.w*Lyr.h + j*Lyr.w + i
									, Lyr.coords );

			avg_obj += Lyr.output[obj_index];
			Lyr.delta[obj_index] = Lyr.object_scale * (1 - Lyr.output[obj_index]);

			if ( Lyr.rescore )
			{
				Lyr.delta[obj_index] = Lyr.object_scale * (iou - Lyr.output[obj_index]);
			}

			if ( Lyr.background )
			{
				Lyr.delta[obj_index] = Lyr.object_scale * (0 - Lyr.output[obj_index]);
			}

			int class = net.truth[t*(Lyr.coords + 1) + b*Lyr.truths + Lyr.coords];

			if ( Lyr.map ) class = Lyr.map[class];

			int class_index = entry_index( Lyr
										, b
										, best_n*Lyr.w*Lyr.h + j*Lyr.w + i
										, Lyr.coords + 1 );

			delta_region_class( Lyr.output
							, Lyr.delta
							, class_index
							, class
							, Lyr.classes
							, Lyr.softmax_tree
							, Lyr.class_scale
							, Lyr.w*Lyr.h
							, &avg_cat
							, !Lyr.softmax );

			++count;
			++class_count;
		}
	}

	*(Lyr.cost) = pow( mag_array( Lyr.delta, Lyr.outputs * Lyr.batch ), 2 );

	printf( "Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n"
		, avg_iou/count
		, avg_cat/class_count
		, avg_obj/count
		, avg_anyobj/(Lyr.w*Lyr.h*Lyr.n*Lyr.batch)
		, recall/count
		, count );
}

void backward_region_layer( const layer Lyr, network net )
{
	/*
	int b;
	int size = Lyr.coords + Lyr.classes + 1;

	for ( b = 0; b < Lyr.batch*Lyr.n; ++b )
	{
		int index = (b*size + 4)*Lyr.w*Lyr.h;
		gradient_array( Lyr.output + index, Lyr.w*Lyr.h, LOGISTIC, Lyr.delta + index );
	}

	axpy_cpu( Lyr.batch*Lyr.inputs, 1, Lyr.delta, 1, net.delta, 1 );
	*/
}

void correct_region_boxes( detection *dets, int n, int w, int h, int netw, int neth, int relative )
{
	int i;
	int new_w=0;
	int new_h=0;
	if ( ((float)netw/w) < ((float)neth/h) )
	{
		new_w = netw;
		new_h = (h * netw)/w;
	}
	else
	{
		new_h = neth;
		new_w = (w * neth)/h;
	}
	for ( i = 0; i < n; ++i )
	{
		box b = dets[i].bbox;
		b.x =  (b.x - (netw - new_w) / 2.0f / netw) / ((float)new_w/netw);
		b.y =  (b.y - (neth - new_h) / 2.0f / neth) / ((float)new_h/neth);
		b.w *= (float)netw/new_w;
		b.h *= (float)neth/new_h;
		if ( !relative )
		{
			b.x *= w;
			b.w *= w;
			b.y *= h;
			b.h *= h;
		}
		dets[i].bbox = b;
	}
}

void get_region_detections( layer Lyr
						, int ww		// 검출자료 너비
						, int hh		// 검출자료 높이
						, int netw		// 망 사비단 너비
						, int neth		// 망 사비단 높이
						, float thresh
						, int *map
						, float tree_thresh
						, int relative
						, detection *dets )
{
	int ii, jj, nn, zz;
	float *predictions = Lyr.output;

	if ( Lyr.batch == 2 )
	{
		float *flip = Lyr.output + Lyr.outputs;

		for ( jj=0; jj < Lyr.h; ++jj )
		{
			for ( ii=0; ii < Lyr.w/2; ++ii )
			{
				for ( nn=0; nn < Lyr.n; ++nn )
				{
					for ( zz=0; zz < Lyr.classes + Lyr.coords + 1; ++zz )
					{
						int i1 = zz*Lyr.w*Lyr.h*Lyr.n + nn*Lyr.w*Lyr.h + jj*Lyr.w + ii;
						int i2 = zz*Lyr.w*Lyr.h*Lyr.n + nn*Lyr.w*Lyr.h + jj*Lyr.w + (Lyr.w - ii - 1);
						float swap = flip[i1];
						flip[i1] = flip[i2];
						flip[i2] = swap;

						if ( zz == 0 )
						{
							flip[i1] = -flip[i1];
							flip[i2] = -flip[i2];
						}
					}
				}
			}
		}

		for ( ii=0; ii < Lyr.outputs; ++ii )
		{
			Lyr.output[ii] = (Lyr.output[ii] + flip[ii])/2.;
		}
	}

	for ( ii=0; ii < Lyr.w*Lyr.h; ++ii )
	{
		int row = ii / Lyr.w;
		int col = ii % Lyr.w;

		for ( nn=0; nn < Lyr.n; ++nn )
		{
			int index = nn*Lyr.w*Lyr.h + ii;

			for ( jj=0; jj < Lyr.classes; ++jj )
			{
				dets[index].prob[jj] = 0;
			}

			int obj_index  = entry_index( Lyr, 0, nn*Lyr.w*Lyr.h + ii, Lyr.coords );
			int box_index  = entry_index( Lyr, 0, nn*Lyr.w*Lyr.h + ii, 0 );
			int mask_index = entry_index( Lyr, 0, nn*Lyr.w*Lyr.h + ii, 4 );
			float scale = Lyr.background ? 1 : predictions[obj_index];

			dets[index].bbox = get_region_box( predictions
											, Lyr.biases
											, nn
											, box_index
											, col
											, row
											, Lyr.w
											, Lyr.h
											, Lyr.w*Lyr.h );

			dets[index].objectness = scale > thresh ? scale : 0;

			if ( dets[index].mask )
			{
				for ( jj=0; jj < Lyr.coords - 4; ++jj )
				{
					dets[index].mask[jj] = Lyr.output[mask_index + jj*Lyr.w*Lyr.h];
				}
			}

			int class_index = entry_index( Lyr
										, 0
										, nn*Lyr.w*Lyr.h + ii
										, Lyr.coords + !Lyr.background );

			if ( Lyr.softmax_tree )
			{

				hierarchy_predictions( predictions + class_index
									, Lyr.classes
									, Lyr.softmax_tree
									, 0
									, Lyr.w*Lyr.h );

				if ( map )
				{
					for ( jj=0; jj < 200; ++jj )
					{
						int class_index = entry_index( Lyr
													, 0
													, nn*Lyr.w*Lyr.h + ii
													, Lyr.coords + 1 + map[jj] );

						float prob = scale*predictions[class_index];
						dets[index].prob[jj] = (prob > thresh) ? prob : 0;
					}
				}
				else
				{
					int idx =  hierarchy_top_prediction( predictions + class_index
													, Lyr.softmax_tree
													, tree_thresh
													, Lyr.w*Lyr.h );
					dets[index].prob[idx] = (scale > thresh) ? scale : 0;
				}
			}
			else
			{
				if ( dets[index].objectness )
				{
					for ( jj=0; jj < Lyr.classes; ++jj )
					{
						int class_index = entry_index( Lyr
													, 0
													, nn*Lyr.w*Lyr.h + ii
													, Lyr.coords + 1 + jj );

						float prob = scale*predictions[class_index];
						dets[index].prob[jj] = (prob > thresh) ? prob : 0;
					}
				}
			}
		}
	}

	correct_region_boxes( dets, Lyr.w*Lyr.h*Lyr.n, ww, hh, netw, neth, relative );
}

#ifdef GPU

void forward_region_layer_gpu( const layer Lyr, network net )
{
	copy_gpu( Lyr.batch*Lyr.inputs, net.input_gpu, 1, Lyr.output_gpu, 1 );

	int b, n;
	for ( b = 0; b < Lyr.batch; ++b )
	{
		for ( n = 0; n < Lyr.n; ++n )
		{
			int index = entry_index( Lyr, b, n*Lyr.w*Lyr.h, 0 );
			activate_array_gpu( Lyr.output_gpu + index, 2*Lyr.w*Lyr.h, LOGISTIC );

			if ( Lyr.coords > 4 )
			{
				index = entry_index( Lyr, b, n*Lyr.w*Lyr.h, 4 );
				activate_array_gpu( Lyr.output_gpu + index, (Lyr.coords - 4)*Lyr.w*Lyr.h, LOGISTIC );
			}

			index = entry_index( Lyr, b, n*Lyr.w*Lyr.h, Lyr.coords );

			if ( !Lyr.background )
				activate_array_gpu( Lyr.output_gpu + index, Lyr.w*Lyr.h, LOGISTIC );

			index = entry_index( Lyr, b, n*Lyr.w*Lyr.h, Lyr.coords + 1 );

			if ( !Lyr.softmax && !Lyr.softmax_tree )
				activate_array_gpu( Lyr.output_gpu + index, Lyr.classes*Lyr.w*Lyr.h, LOGISTIC );
		}
	}

	if ( Lyr.softmax_tree )
	{
		int index = entry_index( Lyr, 0, 0, Lyr.coords + 1 );
		softmax_tree( net.input_gpu + index
					, Lyr.w*Lyr.h
					, Lyr.batch*Lyr.n
					, Lyr.inputs/Lyr.n
					, 1
					, Lyr.output_gpu + index
					, *Lyr.softmax_tree );
	}
	else if ( Lyr.softmax )
	{
		int index = entry_index( Lyr, 0, 0, Lyr.coords + !Lyr.background );
		softmax_gpu( net.input_gpu + index
					, Lyr.classes + Lyr.background
					, Lyr.batch*Lyr.n
					, Lyr.inputs/Lyr.n
					, Lyr.w*Lyr.h
					, 1
					, Lyr.w*Lyr.h
					, 1
					, Lyr.output_gpu + index );
	}
	if ( !net.train || Lyr.onlyforward )
	{
		cuda_pull_array( Lyr.output_gpu, Lyr.output, Lyr.batch*Lyr.outputs );
		return;
	}

	cuda_pull_array( Lyr.output_gpu, net.input, Lyr.batch*Lyr.inputs );
	forward_region_layer( Lyr, net );
	//cuda_push_array(Lyr.output_gpu, Lyr.output, Lyr.batch*Lyr.outputs);
	if ( !net.train ) return;

	cuda_push_array( Lyr.delta_gpu, Lyr.delta, Lyr.batch*Lyr.outputs );
}

void backward_region_layer_gpu( const layer Lyr, network net )
{
	int b, n;
	for ( b = 0; b < Lyr.batch; ++b )
	{
		for ( n = 0; n < Lyr.n; ++n )
		{
			int index = entry_index( Lyr, b, n*Lyr.w*Lyr.h, 0 );
			gradient_array_gpu( Lyr.output_gpu + index, 2*Lyr.w*Lyr.h, LOGISTIC, Lyr.delta_gpu + index );

			if ( Lyr.coords > 4 )
			{
				index = entry_index( Lyr, b, n*Lyr.w*Lyr.h, 4 );
				gradient_array_gpu( Lyr.output_gpu + index
								, (Lyr.coords - 4)*Lyr.w*Lyr.h
								, LOGISTIC
								, Lyr.delta_gpu + index );
			}

			index = entry_index( Lyr, b, n*Lyr.w*Lyr.h, Lyr.coords );

			if ( !Lyr.background )
				gradient_array_gpu( Lyr.output_gpu + index, Lyr.w*Lyr.h, LOGISTIC, Lyr.delta_gpu + index );
		}
	}

	axpy_gpu( Lyr.batch*Lyr.inputs, 1, Lyr.delta_gpu, 1, net.delta_gpu, 1 );
}
#endif

void zero_objectness( layer Lyr )
{
	int i, n;
	for ( i = 0; i < Lyr.w*Lyr.h; ++i )
	{
		for ( n = 0; n < Lyr.n; ++n )
		{
			int obj_index = entry_index( Lyr, 0, n*Lyr.w*Lyr.h + i, Lyr.coords );
			Lyr.output[obj_index] = 0;
		}
	}
}

