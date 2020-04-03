#include <stdio.h>
#include <stdlib.h>

#include "mnist.h"
#include "naive_bayes.h"

double **train_data;
double **test_data;

void set_local(mnist_info *info, mnist_data *data)
{
    train_data = (double **) malloc(info->train_size * sizeof(double *));

    test_data = (double **) malloc(info->test_size * sizeof(double *));

    for (int i = 0; i < info->train_size; i++) {
        train_data[i] = (double *) malloc(info->image_size * sizeof(double));
    }
    for (int i = 0; i < info->test_size; i++) {
        test_data[i] = (double *) malloc(info->image_size * sizeof(double));
    }

    for (int i = 0; i < info->train_size; i++) {
        for (int j = 0; j < info->image_size; j++)
            train_data[i][j] = (data->train_image[i][j]);
    }
    for (int i = 0; i < info->test_size; i++) {
        for (int j = 0; j < info->image_size; j++)
            test_data[i][j] = (data->test_image[i][j]);
    }
}

void free_mem(mnist_info *info, mnist_data *data)
{
    for (int i = 0; i < info->train_size; i++) {
        free(train_data[i]);
    }
    for (int i = 0; i < info->test_size; i++) {
        free(test_data[i]);
    }
    free(train_data);
    free(test_data);
}

int main()
{
    mnist_info *info;
    mnist_data *data;
    double error_rate[2];
    load_mnist();
    info = get_mnist_info();
    data = get_mnist_data();
    set_local(info, data);

    for (int i = 0; i < 2; i++) {
        error_rate[i] = run_naive_bayes_classifier(
            train_data, data->train_label, test_data, data->test_label,
            info->train_size, info->test_size, info->image_size,
            info->label_class_size, i);
        printf("\n");
    }

    printf("=====================================\n");
    printf("Continuous Mode:\n\tAccurancy:  %.4f\n\terror rate: %.4f\n\n",
           1 - error_rate[0], error_rate[0]);
    printf("Discrete Mode:\n\tAccurancy:  %.4f\n\terror rate: %.4f\n",
           1 - error_rate[1], error_rate[1]);
    free_mem(info, data);
    free_mnist_data(data);
    free_mnist_info(info);
    return 0;
}
