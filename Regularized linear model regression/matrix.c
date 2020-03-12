#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>

matrix *new_matrix(){
	matrix *new = (matrix *)malloc(sizeof(matrix));
	new->row_len = 0;
	new->col_len = 0;
	new->ele = NULL;
	return new;
}

matrix *new_matrix_size(int row, int col){
	matrix *new = (matrix *)malloc(sizeof(matrix));
	new->row_len = row;
	new->col_len = col;
	new->ele = (double **)malloc(new->row_len * sizeof(double *));
	for(int i=0; i< new->row_len; i++)
		new->ele[i] = (double *)malloc(new->col_len * sizeof(double));
	return new;
}

void set_matrix(matrix *a, double **val, int m, int n){
	if(!a){
		printf("The matrix could not be empty.\n");
		return;
	}
	for(int i=0; i<a->row_len; i++)
		free(a->ele[i]);
		
	a->row_len = m;
	a->col_len = n;
	a->ele = (double **)malloc(a->row_len * sizeof(double *));
	for(int i=0; i< a->row_len; i++)
		a->ele[i] = (double *)malloc(a->col_len * sizeof(double));
		
	for(int i=0; i<a->row_len; i++)
		for(int j=0; j<a->col_len; j++)
			a->ele[i][j] = val[i][j];
}

/*clear the ele of matrix*/
void clear_matrix(matrix *a){
	for(int i=0; i<a->row_len; i++)
		for(int j=0; j<a->col_len; j++)
			a->ele[i][j] = 0.0;
}

/*reset the matrix struct to initial*/
void reset_matrix(matrix *a){
	for(int i=0; i<a->row_len; i++)
		free(a->ele[i]);
	free(a->ele);
	a->row_len = 0;
	a->col_len = 0;
}

/*free the memory space of matrix*/
void free_matrix(matrix *a){
	for(int i=0; i<a->row_len; i++)
		free(a->ele[i]);
	free(a->ele);
	free(a);
}

void print_matrix(matrix *a){
    printf("[\n");
	for(int i=0; i<a->row_len; i++){
		for(int j=0; j<a->col_len-1; j++)
			printf("\t%.20f\t,", a->ele[i][j]);
        printf("\t%.20f\t", a->ele[i][a->col_len-1]);
		printf(" \\ \n");
	}
    printf("]\n");
}

matrix *transpose_matrix(matrix *a){
	matrix *t = new_matrix_size(a->col_len, a->row_len);
	
	for(int i=0; i<t->row_len; i++)
		for(int j=0; j<t->col_len; j++)
			t->ele[i][j] = a->ele[j][i];
	return t;
}

matrix *multi_matrix(matrix *a, matrix *b){
	if(a->col_len!=b->row_len){
		printf("The two matrix could not multiplied!\n");
		return NULL;
	}

	matrix *sum = new_matrix_size(a->row_len, b->col_len);

	for(int i=0; i<sum->row_len; i++)
		for(int j=0; j<sum->col_len; j++)
			for(int k=0; k<a->col_len; k++)
				sum->ele[i][j] += (a->ele[i][k] * b->ele[k][i]);
	return sum;
}