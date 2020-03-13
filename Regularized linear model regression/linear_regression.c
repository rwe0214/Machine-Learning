#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include "matrix.h"

#define RIO_BUFSIZE 8192 
typedef struct {
    int rio_fd;                /* Descriptor for this internal buf */
    int rio_cnt;               /* Unread bytes in internal buf */
    char *rio_bufptr;          /* Next unread byte in internal buf */
    char rio_buf[RIO_BUFSIZE]; /* Internal buffer */
} rio_t;

void rio_readinitb(rio_t *rp, int fd);
static ssize_t rio_read(rio_t *rp, char *usrbuf, size_t n);
ssize_t rio_readlineb(rio_t *rp, void *usrbuf, size_t maxlen);
int fread_val(double ***x, double ***y, int n);


int main(int argc, char **argv){
	if(argc != 3){
		fprintf(stderr, "usage: ./linear_regression [n] [lambda]\n");
		return EXIT_FAILURE;
	}

	matrix *A, *b, *At, *AtA, *lI, *W, *rlse;
	int m, n = atoi(argv[1]);
	double lambda = atof(argv[2]);
	double **x, **y;

	m = fread_val(&x, &y, n);
	A = new_matrix();
	set_matrix(A, x, m, n);
	b = new_matrix();
	set_matrix(b, y, m, 1);	
	At = transpose_matrix(A);
	AtA = multi_matrix(At, A);
	lI =new_matrix();
	set_matrixI(lI, lambda, m);
	W = add_matrix(AtA, lI);
	b = multi_matrix(At, b);
	rlse = solve_linear_sys(W, b);

	print_matrix(rlse);

	free_matrix(A);
	free_matrix(b);
	free_matrix(At);
	free_matrix(AtA);
	free_matrix(lI);
	free_matrix(W);
	free_matrix(rlse);
	return 0;
}

void rio_readinitb(rio_t *rp, int fd) {
    rp->rio_fd = fd;
    rp->rio_cnt = 0;
    rp->rio_bufptr = rp->rio_buf;
}

static ssize_t rio_read(rio_t *rp, char *usrbuf, size_t n)
{
    int cnt;

    while (rp->rio_cnt <= 0) {  /* refill if buf is empty */
	rp->rio_cnt = read(rp->rio_fd, rp->rio_buf, 
			   sizeof(rp->rio_buf));
	if (rp->rio_cnt < 0) {
		return -1;
	}
	else if (rp->rio_cnt == 0)  /* EOF */
	    return 0;
	else 
	    rp->rio_bufptr = rp->rio_buf; /* reset buffer ptr */
    }

    /* Copy min(n, rp->rio_cnt) bytes from internal buf to user buf */
    cnt = n;          
    if (rp->rio_cnt < n)   
	cnt = rp->rio_cnt;
    memcpy(usrbuf, rp->rio_bufptr, cnt);
    rp->rio_bufptr += cnt;
    rp->rio_cnt -= cnt;
    return cnt;
}

ssize_t rio_readlineb(rio_t *rp, void *usrbuf, size_t maxlen) 
{
    int n, rc;
    char c, *bufp = usrbuf;

    for (n = 1; n < maxlen; n++) { 
    	if ((rc = rio_read(rp, &c, 1)) == 1) {
    	    *bufp++ = c;
    	    if (c == '\n')
    		    break;
    	} else if (rc == 0) {
    	    if (n == 1)
    		    return 0; /* EOF, no data read */
    	    else
    		    break;    /* EOF, some data was read */
    	} else
    	    return -1;	  /* error */
    }
    *bufp = 0;
    return n;
}

int fread_val(double ***x, double ***y, int n){
	
	size_t buf_size = 0;
	char *buf = malloc(buf_size);
	int m=0;
	char *FILENAME = "input.txt";

	FILE *fp = fopen(FILENAME, "r");
	if(!fp){
		fprintf(stderr, "Error opening file '%s'\n", FILENAME);
    	return EXIT_FAILURE;
	}

	while(getline(&buf, &buf_size, fp) != -1)
		m++;

	fseek(fp, 0, SEEK_SET);
	*x = (double **)malloc(m * sizeof(double *));
	for(int i=0; i< m; i++)
		(*x)[i] = (double *)malloc(n * sizeof(double));
	*y = (double **)malloc(m * sizeof(double));
	for(int i=0; i< m; i++)
		(*y)[i] = (double *)malloc(sizeof(double));

	for(int i=0; i<m; i++){
		fscanf(fp, "%lf,%lf\n", &(*x)[i][n-2], &(*y)[i][0]);
		(*x)[i][n-1] = 1.0;
		for(int j=n-3; j>=0; j--)
			(*x)[i][j] = (*x)[i][j+1] * (*x)[i][n-2];
	}

	fclose(fp);
	return m;
}