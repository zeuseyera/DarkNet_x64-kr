#include "activations.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *get_activation_string( ACTIVATION act )
{
    switch( act )
	{
        case LOGISTIC:
            return "logistic";
        case LOGGY:
            return "loggy";
        case RELU:
            return "relu";
        case ELU:
            return "elu";
        case RELIE:
            return "relie";
        case RAMP:
            return "ramp";
        case LINEAR:
            return "linear";
        case TANH:
            return "tanh";
        case PLSE:
            return "plse";
        case LEAKY:
            return "leaky";
        case STAIR:
            return "stair";
        case HARDTAN:
            return "hardtan";
        case LHTAN:
            return "lhtan";
        default:
            break;
    }
    return "relu";
}

ACTIVATION get_activation( char *str )
{
	if ( strcmp( str, "logistic" )==0 )	return LOGISTIC;
	if ( strcmp( str, "loggy" )==0 )	return LOGGY;
	if ( strcmp( str, "relu" )==0 )		return RELU;
	if ( strcmp( str, "elu" )==0 )		return ELU;
	if ( strcmp( str, "relie" )==0 )	return RELIE;
	if ( strcmp( str, "plse" )==0 )		return PLSE;
	if ( strcmp( str, "hardtan" )==0 )	return HARDTAN;
	if ( strcmp( str, "lhtan" )==0 )	return LHTAN;
	if ( strcmp( str, "linear" )==0 )	return LINEAR;
	if ( strcmp( str, "ramp" )==0 )		return RAMP;
	if ( strcmp( str, "leaky" )==0 )	return LEAKY;
	if ( strcmp( str, "tanh" )==0 )		return TANH;
	if ( strcmp( str, "stair" )==0 )	return STAIR;

	fprintf( stderr, "Couldn't find activation function %s, going with ReLU\n", str );

	return RELU;
}

float activate( float xx, ACTIVATION act )
{
	switch ( act )
	{
		case LINEAR:
			return linear_activate( xx );
		case LOGISTIC:
			return logistic_activate( xx );
		case LOGGY:
			return loggy_activate( xx );
		case RELU:
			return relu_activate( xx );
		case ELU:
			return elu_activate( xx );
		case RELIE:
			return relie_activate( xx );
		case RAMP:
			return ramp_activate( xx );
		case LEAKY:
			return leaky_activate( xx );
		case TANH:
			return tanh_activate( xx );
		case PLSE:
			return plse_activate( xx );
		case STAIR:
			return stair_activate( xx );
		case HARDTAN:
			return hardtan_activate( xx );
		case LHTAN:
			return lhtan_activate( xx );
	}
	return 0;
}

void activate_array( float *xx, const int nn, const ACTIVATION act )
{
	int ii;
	for ( ii=0; ii < nn; ++ii )
	{
		xx[ii] = activate( xx[ii], act );
	}
}

float gradient( float xx, ACTIVATION act )
{
	switch ( act )
	{
		case LINEAR:
			return linear_gradient( xx );
		case LOGISTIC:
			return logistic_gradient( xx );
		case LOGGY:
			return loggy_gradient( xx );
		case RELU:
			return relu_gradient( xx );
		case ELU:
			return elu_gradient( xx );
		case RELIE:
			return relie_gradient( xx );
		case RAMP:
			return ramp_gradient( xx );
		case LEAKY:
			return leaky_gradient( xx );
		case TANH:
			return tanh_gradient( xx );
		case PLSE:
			return plse_gradient( xx );
		case STAIR:
			return stair_gradient( xx );
		case HARDTAN:
			return hardtan_gradient( xx );
		case LHTAN:
			return lhtan_gradient( xx );
	}
	return 0;
}

void gradient_array( const float *xx, const int nn, const ACTIVATION act, float *delta )
{
	int ii;
	for ( ii=0; ii < nn; ++ii )
	{
		delta[ii] *= gradient( xx[ii], act );
	}
}

