#include "matrix.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

matrix *new_matrix()
{
    matrix *new = (matrix *) malloc(sizeof(matrix));
    new->row_len = 0;
    new->col_len = 0;
    new->ele = NULL;
    return new;
}

matrix *new_matrix_size(int row, int col)
{
    matrix *new = (matrix *) malloc(sizeof(matrix));
    new->row_len = row;
    new->col_len = col;
    new->ele = (double **) malloc(new->row_len * sizeof(double *));
    for (int i = 0; i < new->row_len; i++)
        new->ele[i] = (double *) malloc(new->col_len * sizeof(double));
    return new;
}

void set_matrix(matrix *a, int m, int n, double **val)
{
    if (!a) {
        printf("The matrix could not be empty.\n");
        return;
    }
    for (int i = 0; i < a->row_len; i++)
        free(a->ele[i]);

    a->row_len = m;
    a->col_len = n;
    a->ele = (double **) malloc(a->row_len * sizeof(double *));
    for (int i = 0; i < a->row_len; i++)
        a->ele[i] = (double *) malloc(a->col_len * sizeof(double));

    for (int i = 0; i < a->row_len; i++)
        for (int j = 0; j < a->col_len; j++)
            a->ele[i][j] = val[i][j];
}

void set_matrixI(matrix *a, double k, int n)
{
    if (!a) {
        printf("The matrix could not be empty.\n");
        return;
    }
    for (int i = 0; i < a->row_len; i++)
        free(a->ele[i]);

    a->row_len = a->col_len = n;
    a->ele = (double **) malloc(a->row_len * sizeof(double *));
    for (int i = 0; i < a->row_len; i++)
        a->ele[i] = (double *) malloc(a->col_len * sizeof(double));

    for (int i = 0; i < a->row_len; i++)
        a->ele[i][i] = k;
}

/*clear the ele of matrix*/
void clear_matrix(matrix *a)
{
    for (int i = 0; i < a->row_len; i++)
        for (int j = 0; j < a->col_len; j++)
            a->ele[i][j] = 0.0;
}

/*reset the matrix struct to initial*/
void reset_matrix(matrix *a)
{
    for (int i = 0; i < a->row_len; i++)
        free(a->ele[i]);
    free(a->ele);
    a->row_len = 0;
    a->col_len = 0;
}

/*free the memory space of matrix*/
void free_matrix(matrix *a)
{
    for (int i = 0; i < a->row_len; i++)
        free(a->ele[i]);
    free(a->ele);
    free(a);
}

void print_matrix(matrix *a)
{
    printf("[\n");
    for (int i = 0; i < a->row_len; i++) {
        for (int j = 0; j < a->col_len - 1; j++)
            printf("\t%.15f\t,", a->ele[i][j]);
        printf("\t%.15f\t", a->ele[i][a->col_len - 1]);
        printf(" \\ \n");
    }
    printf("]\n");
}

double diff_vector(matrix *a, matrix *b)
{
    double diff = 0.0;
    for (int i = 0; i < a->row_len; i++)
        diff += fabs(a->ele[i][0] - b->ele[i][0]);
    diff /= a->row_len;
    return diff;
}

matrix *add_matrix(matrix *a, matrix *b)
{
    matrix *c;
    c = new_matrix_size(a->row_len, a->col_len);

    for (int i = 0; i < a->row_len; i++)
        for (int j = 0; j < a->col_len; j++)
            c->ele[i][j] = a->ele[i][j] + b->ele[i][j];
    return c;
}

matrix *sub_matrix(matrix *a, matrix *b)
{
    matrix *c;
    c = new_matrix_size(a->row_len, a->col_len);

    for (int i = 0; i < a->row_len; i++)
        for (int j = 0; j < a->col_len; j++)
            c->ele[i][j] = a->ele[i][j] - b->ele[i][j];
    return c;
}

matrix *transpose_matrix(matrix *a)
{
    matrix *t = new_matrix_size(a->col_len, a->row_len);

    for (int i = 0; i < t->row_len; i++)
        for (int j = 0; j < t->col_len; j++)
            t->ele[i][j] = a->ele[j][i];
    return t;
}

matrix *multi_matrix(matrix *a, matrix *b)
{
    if (a->col_len != b->row_len) {
        printf("The two matrix could not multiplied!\n");
        return NULL;
    }

    matrix *sum = new_matrix_size(a->row_len, b->col_len);

    for (int i = 0; i < sum->row_len; i++)
        for (int j = 0; j < sum->col_len; j++)
            for (int k = 0; k < a->col_len; k++)
                sum->ele[i][j] += (a->ele[i][k] * b->ele[k][j]);
    return sum;
}

void LU_decompose(matrix *a, matrix **l, matrix **u)
{
    if (a->row_len != a->col_len) {
        printf("The martix is not square.\n");
        return;
    }
    int n = a->row_len;
    // if(!(*l))
    *l = new_matrix_size(n, n);
    // if(!(*u))
    *u = new_matrix_size(n, n);

    for (int i = 0; i < n; i++) {
        (*u)->ele[0][i] = a->ele[0][i];
        (*l)->ele[i][0] = a->ele[i][0] / (*u)->ele[0][0];
        (*l)->ele[i][i] = 1.0;
    }

    double sum;
    for (int i = 1; i < n; i++) {
        for (int j = i; j < n; j++) {
            sum = 0.0;
            for (int k = 0; k < i; k++)
                sum += (*l)->ele[i][k] * (*u)->ele[k][j];
            (*u)->ele[i][j] = a->ele[i][j] - sum;
        }
        /*Need to get U matrix first!*/
        for (int j = i; j < n; j++) {
            sum = 0.0;
            for (int k = 0; k < i; k++)
                sum += (*l)->ele[j][k] * (*u)->ele[k][i];
            (*l)->ele[j][i] = (a->ele[j][i] - sum) / (*u)->ele[i][i];
        }
    }
    return;
}
/*Only work at square matrix A*/
matrix *solve_linear_sys(matrix *A, matrix *b)
{
    /* Ax = LUx = b
     * let y = Ux
     * Ly = b, y = L^(-1)b
     * solve Ux = y
     */
    if (A->row_len != A->col_len) {
        printf("A need to be square matrix!\n");
        return NULL;
    }
    matrix *L_i, *U;
    LU_decompose(A, &L_i, &U);
    for (int i = 0; i < L_i->row_len; i++)
        for (int j = 0; j < i; j++)
            L_i->ele[i][j] *= -1;
    matrix *y = multi_matrix(L_i, b);
    matrix *x = new_matrix_size(A->col_len, 1);

    double sum;
    x->ele[x->row_len - 1][0] =
        y->ele[y->row_len - 1][0] / U->ele[U->row_len - 1][U->row_len - 1];
    for (int i = x->row_len - 2; i >= 0; i--) {
        sum = 0.0;
        for (int j = x->row_len - 1; j > i; j--)
            sum += U->ele[i][j] * x->ele[j][0];
        x->ele[i][0] = (y->ele[i][0] - sum) / U->ele[i][i];
    }
    return x;
}

matrix *minor_matrix(matrix *a, int row, int col)
{
    matrix *m = new_matrix_size(a->row_len - 1, a->col_len - 1);
    int r = 0, l = 0;
    //	printf("aa\n");
    for (int i = 0; i < a->row_len; i++)
        for (int j = 0; j < a->col_len; j++)
            if (i != row && j != col) {
                //				printf("%d, %d\t%d, %d\n", i, j, r, l);
                m->ele[r][l++] = a->ele[i][j];
                if (l == m->col_len) {
                    l = 0;
                    r++;
                }
            }

    return m;
}

double determinant_matrix(matrix *a)
{
    if (a->row_len != a->col_len) {
        printf("Only square matrix could be calculate determinant!\n");
        exit(-1);
    }
    if (a->row_len == 1)
        return a->ele[0][0];
    else if (a->row_len == 2)
        return a->ele[0][0] * a->ele[1][1] - a->ele[1][0] * a->ele[0][1];
    else {
        double sum = 0.0;
        for (int j = 0; j < a->col_len; j++)
            sum += (pow(-1, 0 + j) * a->ele[0][j] *
                    determinant_matrix(minor_matrix(a, 0, j)));
        return sum;
    }
}

matrix *cof_matrix(matrix *a)
{
    if (!a) {
        printf("A is empty!\n");
        return NULL;
    }

    matrix *cof = new_matrix_size(a->row_len, a->col_len);
    for (int i = 0; i < a->row_len; i++)
        for (int j = 0; j < a->col_len; j++)
            cof->ele[i][j] =
                (pow(-1, i + j) * determinant_matrix(minor_matrix(a, i, j)));

    return cof;
}

matrix *inverse_matrix(matrix *a)
{
    matrix *adj = transpose_matrix(cof_matrix(a));
    matrix *inverse = new_matrix_size(a->row_len, a->col_len);

    for (int i = 0; i < a->row_len; i++)
        for (int j = 0; j < a->col_len; j++)
            inverse->ele[i][j] = adj->ele[i][j] / determinant_matrix(a);
    return inverse;
}