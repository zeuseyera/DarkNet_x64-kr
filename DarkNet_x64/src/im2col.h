#ifndef IM2COL_H
#define IM2COL_H
// ����� �޸𸮿� ����(collect)�Ѵ�
void im2col_cpu( float* data_im	// ������ �ڷᰪ
				, int channels	// ��� �Ǽ�
				, int height	// ��� ����
				, int width		// ��� ����
				, int ksize		// ���� ����,����
				, int stride	// ���� ��
				, int pad		// �е�
				, float* data_col );	// ���� �ڷᰪ

#ifdef GPU
// ����� ��ġ�� �޸𸮿� ����(collect)�Ѵ�
void im2col_ongpu( float *im	// ������ �ڷᰪ
				, int channels	// ��� �Ǽ�
				, int height	// ��� ����
				, int width		// ��� ����
				, int ksize		// ���� ����,����
				, int stride	// ���� ��
				, int pad		// �е�
				, float *data_col );	// ���� �ڷᰪ

#endif
#endif
