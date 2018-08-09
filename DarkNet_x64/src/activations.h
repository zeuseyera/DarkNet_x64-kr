#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include "darknet.h"
#include "cuda.h"
#include "math.h"

ACTIVATION get_activation( char *s );

char *get_activation_string( ACTIVATION a );
float activate( float x, ACTIVATION a );
float gradient( float x, ACTIVATION a );
void gradient_array( const float *x, const int n, const ACTIVATION a, float *delta );
void activate_array( float *x, const int n, const ACTIVATION a );
#ifdef GPU
void activate_array_gpu( float *x, int n, ACTIVATION a );
void gradient_array_gpu( float *x, int n, ACTIVATION a, float *delta );
#endif

static inline float stair_activate(float x)
{
    int n = floor(x);
    if (n%2 == 0) return (float)floor( x/2.0 );
    else return (float)((x - n) + floor( x/2.0 ));
}
static inline float hardtan_activate(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
static inline float linear_activate(float x)	{ return x; }
static inline float logistic_activate(float x)	{ return (float)( 1.0/(1.0 + exp(-x)) ); }
static inline float loggy_activate(float x)		{ return (float)( 2.0/(1.0 + exp(-x)) - 1 ); }
static inline float relu_activate(float x)		{ return x*(x>0); }
static inline float elu_activate(float x)		{ return (float)( (x >= 0)*x + (x < 0)*(exp(x)-1) ); }
static inline float relie_activate(float x)		{ return (x>0) ? x : 0.01f*x; }
static inline float ramp_activate(float x)		{ return x*(x>0)+0.1f*x; }
static inline float leaky_activate(float x)		{ return (x>0) ? x : 0.1f*x; }
static inline float tanh_activate(float x)		{ return (float)( (exp(2*x)-1)/(exp(2*x)+1) ); }
static inline float plse_activate(float x)
{
    if(x < -4) return 0.01f * (x + 4);
    if(x > 4)  return 0.01f * (x - 4) + 1;
    return 0.125f*x + 0.5f;
}

static inline float lhtan_activate(float x)
{
    if(x < 0) return 0.001f*x;
    if(x > 1) return 0.001f*(x-1) + 1;
    return x;
}
static inline float lhtan_gradient(float x)
{
    if(x > 0 && x < 1) return 1;
    return 0.001f;
}

static inline float hardtan_gradient(float x)
{
    if (x > -1 && x < 1) return 1;
    return 0;
}
static inline float linear_gradient(float x)	{ return 1; }
static inline float logistic_gradient(float x)	{ return (1-x)*x; }
static inline float loggy_gradient(float x)
{
    float y = (x+1.0f) / 2.0f;
    return 2 * ( 1-y ) * y;
}
static inline float stair_gradient(float x)
{
    if (floor(x) == x) return 0;
    return 1;
}
static inline float relu_gradient(float x)	{ return (x>0); }
static inline float elu_gradient(float x)	{ return (x >= 0) + (x < 0)*(x + 1); }
static inline float relie_gradient(float x)	{ return (x>0) ? 1 : 0.01f; }
static inline float ramp_gradient(float x)	{ return (x>0)+0.1f; }
static inline float leaky_gradient(float x)	{ return (x>0) ? 1 : 0.1f; }
static inline float tanh_gradient(float x)	{ return 1-x*x; }
static inline float plse_gradient(float x)	{ return (x < 0 || x > 1) ? 0.01f : 0.125f; }

#endif

