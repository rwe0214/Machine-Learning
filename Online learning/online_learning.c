#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define PATH "online_learning.data"

char **data;
int line_num = 0, print = 0;
double marginal = 0.0;
char *out;
size_t line_buf_size = 32;
char *line_buf;
FILE *f_config;
int *a, *b;
double *p;

double mygamma(int a)
{
    if (a == 1 || a == 2)
        return 1;
    return (a - 1) * mygamma(a - 1);
}

double beta(int a, int b)
{
    return mygamma(a) * mygamma(b) / mygamma(a + b);
}

double beta_distribution(double p, int a, int b)
{
    double beta_inverse = 1 / beta(a, b);
    return pow(p, a - 1) * pow(1 - p, b - 1) * beta_inverse;
}

double factorial(int n)
{
    if (n == 1 || n == 0)
        return 1;
    return n * (factorial(n - 1));
}

double likelihood(int m, int n, double p)
{
    return factorial(n) / (factorial(m) * factorial(n - m)) * pow(p, m) *
           pow(1 - p, n - m);
}

void load_data(char *path, char ***data, int *line_num)
{
    FILE *fp = fopen(path, "r");
    size_t buf_size = 0;
    char *tmp = NULL;

    if (!fp) {
        fprintf(stderr, "Error opening file '%s'\n", path);
        exit(-1);
    }

    while (getline(&tmp, &buf_size, fp) != -1)
        (*line_num)++;
    fseek(fp, 0, SEEK_SET);
    free(tmp);

    *data = (char **) malloc((*line_num) * sizeof(char *));
    for (int i = 0; i < (*line_num); i++) {
        (*data)[i] = (char *) malloc(512 * sizeof(char));
        fscanf(fp, "%s", (*data)[i]);
    }
    fclose(fp);
}

void init(int line_num)
{
    a = malloc((line_num + 1) * sizeof(int));
    b = malloc((line_num + 1) * sizeof(int));
    p = malloc((line_num) * sizeof(double));
    out = malloc(64 * sizeof(char));
    line_buf = malloc(line_buf_size * sizeof(char));
    f_config = fopen("config.txt", "r");
    if (!!access("output", 0)) {
        printf("mkdir output/ ...\n");
        mkdir("output", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
}

void online_learning()
{
    /*read config.txt to get a0 and b0*/
    while (getline(&line_buf, &line_buf_size, f_config) > 0) {
        if (print++ > 0)
            printf("\n===============================\n\n");
        sscanf(line_buf, "%d,%d\n", &a[0], &b[0]);
        printf("a0 = %d, b0 = %d\n", a[0], b[0]);
        /*run 11 cases*/
        for (int i = 1; i <= line_num; i++) {
            sprintf(out, "output/a%d_b%d_%d.txt", a[0], b[0], i);

            /*for plotting .png*/
            FILE *fp = fopen(out, "w");
            if (!fp) {
                fprintf(stderr, "Could not open %s\n", out);
                exit(-1);
            }
            int m = 0, n;
            for (n = 0; data[i - 1][n] != '\0'; n++)
                if (data[i - 1][n] == '1')
                    m++;
            marginal += (double) m / (double) n;
            a[i] = a[i - 1] + m, b[i] = b[i - 1] + (n - m);

            for (int k = 1; k <= 100; k++)
                fprintf(fp, "%.2f,%.10f\n", (double) k / 100.0,
                        beta_distribution((double) k / 100.0, a[i], b[i]));
            p[i - 1] = likelihood(m, n, (double) m / (double) n);
            printf("case %d: %s\n", i, data[i - 1]);

            /*output*/
            printf("Likelihood(MLE): %.17f\n", p[i - 1]);
            printf("Beta prior:\t a=%d, b=%d\n", a[i - 1], b[i - 1]);
            if (i == line_num)
                printf("Beta posterior:\t a=%d, b=%d\n", a[i], b[i]);
            else
                printf("Beta posterior:\t a=%d, b=%d\n\n\n\n", a[i], b[i]);
            fclose(fp);
        }
    }
};

void free_mem()
{
    for (int i = 0; i < line_num; i++)
        free(data[i]);
    free(data);
    free(a);
    free(b);
    free(p);
    free(line_buf);
    fclose(f_config);
    free(out);
}

int main()
{
    load_data(PATH, &data, &line_num);
    init(line_num);
    online_learning();
    free_mem();
    return 0;
}
