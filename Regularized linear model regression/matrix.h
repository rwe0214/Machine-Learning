#ifndef __MATRIX_H
#define __MATRIX_H

/*
 *	Matrix size: mxn
 */
typedef struct MATRIX{
	double **ele;
	int row_len;
	int col_len;
}matrix;

matrix *new_matrix();
matrix *new_matrix_size(int row, int col);
void set_matrix(matrix *a, double **val, int m, int n);
void clear_matrix(matrix *a);
void reset_matrix(matrix *a);
void free_matrix(matrix *a);
void print_matrix(matrix *a);
matrix *transpose_matrix(matrix *a);
matrix *multi_matrix(matrix *a, matrix *b);

#endif