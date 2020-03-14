#ifndef __MATRIX_H
#define __MATRIX_H

/*
 *	Matrix size: mxn
 */
typedef struct MATRIX {
    double **ele;
    int row_len;
    int col_len;
} matrix;

matrix *new_matrix();
matrix *new_matrix_size(int row, int col);
void set_matrix(matrix *a, int m, int n, double **val);
void set_matrixI(matrix *a, double k, int n);
void clear_matrix(matrix *a);
void reset_matrix(matrix *a);
void free_matrix(matrix *a);
void print_matrix(matrix *a);
double diff_vector(matrix *a, matrix *b);
matrix *add_matrix(matrix *a, matrix *b);
matrix *sub_matrix(matrix *a, matrix *b);
matrix *transpose_matrix(matrix *a);
matrix *multi_matrix(matrix *a, matrix *b);
void LU_decompose(matrix *a, matrix **l, matrix **u);
matrix *solve_linear_sys(matrix *A, matrix *b);
matrix *minor_matrix(matrix *a, int row, int col);
double determinant_matrix(matrix *a);
matrix *cof_matrix(matrix *a);
matrix *inverse_matrix(matrix *a);

#endif