#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "matrix.h"

int fread_val(double ***x, double ***y, int n);

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: ./linear_regression [n] [lambda]\n");
        return EXIT_FAILURE;
    }

    matrix *A, *b, *At, *AtA, *Atb, *lI, *W, *rlse;
    int m, n = atoi(argv[1]);
    double lambda = atof(argv[2]);
    double **x, **y;

    /*
     * rLSE method
     * min | b - Ax |^2
     * (AtA - lI)x = Atb
     * LUx = Atb
     * Ly = Atb
     * y = L^(-1)Atb
     * Ux = y
    */
    m = fread_val(&x, &y, n);
    A = new_matrix();
    set_matrix(A, m, n, x);
    b = new_matrix();
    set_matrix(b, m, 1, y);
    At = transpose_matrix(A);
    AtA = multi_matrix(At, A);
    lI = new_matrix();
    set_matrixI(lI, lambda, m);
    W = add_matrix(AtA, lI);
    Atb = multi_matrix(At, b);
    rlse = solve_linear_sys(W, Atb);

    print_matrix(rlse);
    /*
     * Newton's Method
     * f(x) = | b - Ax |^2
     * f'(x) = 2AtAx - 2Atb
     * f"(x) = 2AtA
     * x_(n+1) = x_n - (AtA)^(-1)*(AtAx_n - Atb)
    */
    matrix *X = new_matrix_size(n, 1);
    for (int i = 0; i < n; i++)
        X->ele[i][0] = 0.0;

    matrix *df = sub_matrix(multi_matrix(AtA, X), Atb);
    matrix *hf = inverse_matrix(AtA);
    matrix *Xn = new_matrix_size(n, 1);
    Xn = sub_matrix(X, multi_matrix(hf, df));
    while (diff_vector(Xn, X) > 0.0005) {
        X = Xn;
        df = sub_matrix(multi_matrix(AtA, X), Atb);
        Xn = sub_matrix(X, multi_matrix(hf, df));
    }
    print_matrix(Xn);

    free_matrix(A);
    free_matrix(b);
    free_matrix(At);
    free_matrix(AtA);
    free_matrix(Atb);
    free_matrix(lI);
    free_matrix(W);
    free_matrix(rlse);
    free_matrix(X);
    free_matrix(df);
    free_matrix(hf);
    free_matrix(Xn);
    return 0;
}

int fread_val(double ***x, double ***y, int n)
{
    size_t buf_size = 0;
    char *buf = malloc(buf_size);
    int m = 0;
    char *FILENAME = "input.txt";

    FILE *fp = fopen(FILENAME, "r");
    if (!fp) {
        fprintf(stderr, "Error opening file '%s'\n", FILENAME);
        return EXIT_FAILURE;
    }

    while (getline(&buf, &buf_size, fp) != -1)
        m++;

    fseek(fp, 0, SEEK_SET);
    *x = (double **) malloc(m * sizeof(double *));
    for (int i = 0; i < m; i++)
        (*x)[i] = (double *) malloc(n * sizeof(double));
    *y = (double **) malloc(m * sizeof(double));
    for (int i = 0; i < m; i++)
        (*y)[i] = (double *) malloc(sizeof(double));

    for (int i = 0; i < m; i++) {
        fscanf(fp, "%lf,%lf\n", &(*x)[i][n - 2], &(*y)[i][0]);
        (*x)[i][n - 1] = 1.0;
        for (int j = n - 3; j >= 0; j--)
            (*x)[i][j] = (*x)[i][j + 1] * (*x)[i][n - 2];
    }

    fclose(fp);
    return m;
}