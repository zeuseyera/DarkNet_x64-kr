#include "box.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
// nms �񱳱�
int nms_comparator( const void *pa, const void *pb )
{
	detection a = *(detection *)pa;
	detection b = *(detection *)pb;

	float diff = 0;

	if ( b.sort_class >= 0 )
	{
		diff = a.prob[b.sort_class] - b.prob[b.sort_class];
	}
	else
	{
		diff = a.objectness - b.objectness;
	}

	if		( diff < 0 ) return 1;
	else if ( diff > 0 ) return -1;

	return 0;
}
// ���ⰳü�� ��°� ũ�� ������������ �����ϰ�, ���ΰ����� ���� ���������� �ʱ�ȭ �Ѵ�
void do_nms_obj( detection *dets
			, int total			// ����� ����
			, int classes		// �з�����
			, float thresh )	// ���ΰ�
{
	int ii, jj, kk;
	kk = total-1;
	// ���ⰳ�� �ݺ��Ͽ� ���Ⱚ�� ����ū ������ ����
	for ( ii=0; ii <= kk; ++ii )
	{
		if ( dets[ii].objectness == 0 )
		{
			detection swap	= dets[ii];	// �������(ii) ���� ��Ƶд�
			dets[ii]		= dets[kk];	// ���ʻ���(kk) ���� �������(ii) ���� ����
			dets[kk]		= swap;		// ���ʻ���(kk) ���� ��Ƶ� ���� ����
			--kk;
			--ii;
		}
	}

	total = kk+1;
	// ������ �����Ͽ����Ƿ� sort_class �� �ʱ�ȭ �Ѵ�
	for ( ii=0; ii < total; ++ii )
	{
		dets[ii].sort_class = -1;
	}
	// �����ڷḦ qsort �� �����Ѵ�
	qsort( dets, total, sizeof( detection ), nms_comparator );

	for ( ii=0; ii < total; ++ii )
	{
		if ( dets[ii].objectness == 0 ) continue;

		box ba = dets[ii].bbox;

		for ( jj=ii+1; jj < total; ++jj )
		{
			if ( dets[jj].objectness == 0 ) continue;

			box bb = dets[jj].bbox;

			if ( box_iou( ba, bb ) > thresh )
			{
				dets[jj].objectness = 0;

				for ( kk=0; kk < classes; ++kk )
				{
					dets[jj].prob[kk] = 0;
				}
			}
		}
	}
}


void do_nms_sort( detection *dets, int total, int classes, float thresh )
{
	int i, j, k;
	k = total-1;

	for ( i = 0; i <= k; ++i )
	{
		if ( dets[i].objectness == 0 )
		{
			detection swap = dets[i];
			dets[i] = dets[k];
			dets[k] = swap;
			--k;
			--i;
		}
	}

	total = k+1;

	for ( k = 0; k < classes; ++k )
	{
		for ( i = 0; i < total; ++i )
		{
			dets[i].sort_class = k;
		}

		qsort( dets, total, sizeof( detection ), nms_comparator );

		for ( i = 0; i < total; ++i )
		{
			if ( dets[i].prob[k] == 0 ) continue;

			box a = dets[i].bbox;

			for ( j = i+1; j < total; ++j )
			{
				box b = dets[j].bbox;

				if ( box_iou( a, b ) > thresh )
				{
					dets[j].prob[k] = 0;
				}
			}
		}
	}
}

box float_to_box( float *ff, int stride )
{
	box bx = { 0 };
	bx.x = ff[0];
	bx.y = ff[1*stride];
	bx.w = ff[2*stride];
	bx.h = ff[3*stride];

	return bx;
}

dbox derivative( box a, box b )
{
	dbox d;
	d.dx = 0;
	d.dw = 0;
	float l1 = a.x - a.w/2;
	float l2 = b.x - b.w/2;

	if ( l1 > l2 )	{	d.dx -= 1;		d.dw += 0.5f;	}

	float r1 = a.x + a.w/2;
	float r2 = b.x + b.w/2;

	if ( r1 < r2 )	{	d.dx += 1;		d.dw += 0.5f;	}
	if ( l1 > r2 )	{	d.dx = -1;		d.dw = 0;	}
	if ( r1 < l2 )	{	d.dx = 1;		d.dw = 0;	}

	d.dy = 0;
	d.dh = 0;
	float t1 = a.y - a.h/2;
	float t2 = b.y - b.h/2;

	if ( t1 > t2 )	{	d.dy -= 1;		d.dh += 0.5f;	}

	float b1 = a.y + a.h/2;
	float b2 = b.y + b.h/2;

	if ( b1 < b2 )	{	d.dy += 1;		d.dh += 0.5f;	}
	if ( t1 > b2 )	{	d.dy = -1;		d.dh = 0;	}
	if ( b1 < t2 )	{	d.dy = 1;		d.dh = 0;	}

	return d;
}
// �� ���� ��ģ(��ģ) ũ�⸦ ����ؼ� ��ȯ�Ѵ�
float overlap( float x1, float w1, float x2, float w2 )
{
	float l1	= x1 - w1/2;
	float l2	= x2 - w2/2;

	float left	= l1 > l2 ? l1 : l2;	// ������

	float r1	= x1 + w1/2;
	float r2	= x2 + w2/2;

	float right	= r1 < r2 ? r1 : r2;	// ������

	return right - left;
}
// �� ������ ������ ũ�⸦ ����ؼ� ��ȯ�Ѵ�
float box_intersection( box ba, box bb )
{
	float ww = overlap( ba.x, ba.w, bb.x, bb.w );	// ��ħ �ʺ�
	float hh = overlap( ba.y, ba.h, bb.y, bb.h );	// ��ħ ����

	if ( ww < 0 || hh < 0 ) return 0;

	float area = ww*hh;

	return area;
}
// �� ������ ���յ� ũ�⸦ ����ؼ� ��ȯ�Ѵ�
float box_union( box ba, box bb )
{
	float ii = box_intersection( ba, bb );	// �λ��ڰ� ������ ũ��

	float uu = ba.w*ba.h + bb.w*bb.h - ii;	// ����1 + ����2 - �λ��ڰ� ������ ũ��

	return uu;	// �� ������ ���յ� ũ��
}
// �� ���ڰ� �����ؼ� ��ģ ������ ����ؼ� ��ȯ�Ѵ�
float box_iou( box ba, box bb )
{
	return box_intersection( ba, bb ) / box_union( ba, bb );
}
//
float box_rmse( box a, box b )
{
	return sqrt( pow( a.x-b.x, 2 ) +
				 pow( a.y-b.y, 2 ) +
				 pow( a.w-b.w, 2 ) +
				 pow( a.h-b.h, 2 ) );
}

dbox dintersect( box a, box b )
{
	float w = overlap( a.x, a.w, b.x, b.w );
	float h = overlap( a.y, a.h, b.y, b.h );
	dbox dover = derivative( a, b );
	dbox di;

	di.dw = dover.dw*h;
	di.dx = dover.dx*h;
	di.dh = dover.dh*w;
	di.dy = dover.dy*w;

	return di;
}

dbox dunion( box a, box b )
{
	dbox du;

	dbox di = dintersect( a, b );
	du.dw = a.h - di.dw;
	du.dh = a.w - di.dh;
	du.dx = -di.dx;
	du.dy = -di.dy;

	return du;
}


void test_dunion()
{
	box a	= { 0,			 0,			 1,			 1 };
	box dxa	= { 0+0.0001f,	 0,			 1,			 1 };
	box dya	= { 0,			 0+0.0001f,	 1,			 1 };
	box dwa	= { 0,			 0,			 1+0.0001f,	 1 };
	box dha	= { 0,			 0,			 1,			 1+0.0001f };

	box b = { 0.5f, 0.5f, 0.2f, 0.2f };

	dbox di = dunion( a, b );
	printf( "Union: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh );

	float inter  = box_union( a, b );
	float xinter = box_union( dxa, b );
	float yinter = box_union( dya, b );
	float winter = box_union( dwa, b );
	float hinter = box_union( dha, b );

	xinter = (xinter - inter) / (0.0001f);
	yinter = (yinter - inter) / (0.0001f);
	winter = (winter - inter) / (0.0001f);
	hinter = (hinter - inter) / (0.0001f);

	printf( "Union Manual %f %f %f %f\n", xinter, yinter, winter, hinter );
}

void test_dintersect()
{
	box a	= { 0,			 0,			 1,			 1 };
	box dxa	= { 0+0.0001f,	 0,			 1,			 1 };
	box dya	= { 0,			 0+0.0001f,	 1,			 1 };
	box dwa	= { 0,			 0,			 1+0.0001f,	 1 };
	box dha	= { 0,			 0,			 1,			 1+0.0001f };

	box b = { 0.5f, 0.5f, 0.2f, 0.2f };

	dbox di = dintersect( a, b );
	printf( "Inter: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh );

	float inter =  box_intersection( a, b );
	float xinter = box_intersection( dxa, b );
	float yinter = box_intersection( dya, b );
	float winter = box_intersection( dwa, b );
	float hinter = box_intersection( dha, b );

	xinter = (xinter - inter) / (0.0001f);
	yinter = (yinter - inter) / (0.0001f);
	winter = (winter - inter) / (0.0001f);
	hinter = (hinter - inter) / (0.0001f);

	printf( "Inter Manual %f %f %f %f\n", xinter, yinter, winter, hinter );
}

void test_box()
{
	test_dintersect();
	test_dunion();
	box a	= { 0,			 0,			 1,			 1 };
	box dxa	= { 0+0.00001f,	 0,			 1,			 1 };
	box dya	= { 0,			 0+0.00001f, 1,			 1 };
	box dwa	= { 0,			 0,			 1+0.00001f, 1 };
	box dha	= { 0,			 0,			 1,			 1+0.00001f };

	box b = { .5, 0, .2, .2 };

	float iou = box_iou( a, b );
	iou = (1-iou)*(1-iou);
	printf( "%f\n", iou );
	dbox d = diou( a, b );
	printf( "%f %f %f %f\n", d.dx, d.dy, d.dw, d.dh );

	float xiou = box_iou( dxa, b );
	float yiou = box_iou( dya, b );
	float wiou = box_iou( dwa, b );
	float hiou = box_iou( dha, b );

	xiou = ((1-xiou)*(1-xiou) - iou) / (0.00001f);
	yiou = ((1-yiou)*(1-yiou) - iou) / (0.00001f);
	wiou = ((1-wiou)*(1-wiou) - iou) / (0.00001f);
	hiou = ((1-hiou)*(1-hiou) - iou) / (0.00001f);

	printf( "manual %f %f %f %f\n", xiou, yiou, wiou, hiou );
}

dbox diou( box a, box b )
{
	float u = box_union( a, b );
	float i = box_intersection( a, b );
	dbox di = dintersect( a, b );
	dbox du = dunion( a, b );
	dbox dd = { 0,0,0,0 };

	if ( i <= 0 || 1 )
	{
		dd.dx = b.x - a.x;
		dd.dy = b.y - a.y;
		dd.dw = b.w - a.w;
		dd.dh = b.h - a.h;
		return dd;
	}

	dd.dx = 2 * pow( (1-(i/u)), 1 ) * (di.dx*u - du.dx*i)/(u*u);
	dd.dy = 2 * pow( (1-(i/u)), 1 ) * (di.dy*u - du.dy*i)/(u*u);
	dd.dw = 2 * pow( (1-(i/u)), 1 ) * (di.dw*u - du.dw*i)/(u*u);
	dd.dh = 2 * pow( (1-(i/u)), 1 ) * (di.dh*u - du.dh*i)/(u*u);

	return dd;
}


void do_nms( box *boxes, float **probs, int total, int classes, float thresh )
{
	int i, j, k;
	for ( i = 0; i < total; ++i )
	{
		int any = 0;
		for ( k = 0; k < classes; ++k ) any = any || (probs[i][k] > 0);

		if ( !any )
		{
			continue;
		}

		for ( j = i+1; j < total; ++j )
		{
			if ( box_iou( boxes[i], boxes[j] ) > thresh )
			{
				for ( k = 0; k < classes; ++k )
				{
					if ( probs[i][k] < probs[j][k] ) probs[i][k] = 0;
					else probs[j][k] = 0;
				}
			}
		}
	}
}

box encode_box( box b, box anchor )
{
	box encode;

	encode.x = (b.x - anchor.x) / anchor.w;
	encode.y = (b.y - anchor.y) / anchor.h;
	encode.w = log2( b.w / anchor.w );
	encode.h = log2( b.h / anchor.h );

	return encode;
}

box decode_box( box b, box anchor )
{
	box decode;

	decode.x = b.x * anchor.w + anchor.x;
	decode.y = b.y * anchor.h + anchor.y;
	decode.w = pow( 2.0, b.w ) * anchor.w;
	decode.h = pow( 2.0, b.h ) * anchor.h;

	return decode;
}
