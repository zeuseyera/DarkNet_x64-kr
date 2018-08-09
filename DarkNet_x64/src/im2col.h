#ifndef IM2COL_H
#define IM2COL_H
// 사비값을 메모리에 수집(collect)한다
void im2col_cpu( float* data_im	// 수집할 자료값
				, int channels	// 사비 판수
				, int height	// 사비 세로
				, int width		// 사비 가로
				, int ksize		// 포집 가로,세로
				, int stride	// 포집 보
				, int pad		// 패딩
				, float* data_col );	// 수집 자료값

#ifdef GPU
// 사비값을 장치의 메모리에 수집(collect)한다
void im2col_ongpu( float *im	// 수집할 자료값
				, int channels	// 사비 판수
				, int height	// 사비 세로
				, int width		// 사비 가로
				, int ksize		// 포집 가로,세로
				, int stride	// 포집 보
				, int pad		// 패딩
				, float *data_col );	// 수집 자료값

#endif
#endif
