#include "naive_bayes.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int *count_class;
double **train_class_mu;    // mean
double **train_class_sd;    // standard deviation
double ***train_class_bin;  // for discrete mode
double **predict;

double gaussian(double x, double mu, double sd)
{
    double e = exp(pow((x - mu), 2) / (2 * sd) * (-1));
    return e / sqrt(2 * M_PI * sd);
}

void statistic_data(double **data,
                    uint8_t *target,
                    int num,
                    int size,
                    int num_class,
                    int mode)
{
    /*init*/
    count_class = calloc(num_class, sizeof(int));
    train_class_mu = (double **) malloc(num_class * sizeof(double *));
    for (int i = 0; i < num_class; i++) {
        train_class_mu[i] = (double *) calloc(size, sizeof(double));
    }
    train_class_sd = (double **) malloc(num_class * sizeof(double *));
    for (int i = 0; i < num_class; i++) {
        train_class_sd[i] = (double *) calloc(size, sizeof(double));
    }
    train_class_bin = (double ***) malloc(num_class * sizeof(double **));
    for (int i = 0; i < num_class; i++) {
        train_class_bin[i] = (double **) malloc(size * sizeof(double *));
        for (int j = 0; j < size; j++) {
            train_class_bin[i][j] = (double *) calloc(size, sizeof(double));
            for (int k = 0; k < 32; k++)
                train_class_bin[i][j][k] = 0.0000001;
        }
    }

    /*count*/
    for (int i = 0; i < num; i++) {
        count_class[target[i]]++;
    }

    /*get mean*/
    for (int i = 0; i < num; i++) {
        int class = target[i];
        for (int j = 0; j < size; j++) {
            if (mode == 0)
                train_class_mu[class][j] += data[i][j];
            else
                train_class_mu[class][j] += ((int) data[i][j] >> 3);
        }
    }

    for (int i = 0; i < 10; i++)
        for (int j = 0; j < size; j++)
            train_class_mu[i][j] /= count_class[i];

    /*get varience*/
    for (int i = 0; i < num; i++) {
        int class = target[i];
        for (int j = 0; j < size; j++) {
            train_class_sd[class][j] +=
                pow(data[i][j] - train_class_mu[class][j], 2);
        }
    }
    for (int i = 0; i < 10; i++)
        for (int j = 0; j < size; j++) {
            train_class_sd[i][j] /= (double) count_class[i];
            if (train_class_sd[i][j] < 1840)
                train_class_sd[i][j] = 1840;
        }
    /*count bin*/
    for (int i = 0; i < num; i++) {
        int class = target[i];
        for (int j = 0; j < size; j++) {
            train_class_bin[class][j][(int) data[i][j] >> 3]++;
        }
    }
}

double prediction(double **test,
                  uint8_t *target,
                  int num_train,
                  int num_test,
                  int size_feature,
                  int num_class,
                  int mode)
{
    if (mode == 0)
        printf("Continuous Mode:\n");
    else
        printf("Discrete Mode:\n");
    double count = 0.0;
    predict = (double **) malloc(num_test * sizeof(double *));
    for (int i = 0; i < num_test; i++) {
        predict[i] = (double *) malloc(num_class * sizeof(double));
        memset(predict[i], 0, num_class * sizeof(double));
    }

    for (int i = 0; i < num_test; i++) {
        printf("No.%d test:\n", i + 1);

        double marginal = 0.0;
        for (int j = 0; j < num_class; j++) {
            for (int k = 0; k < size_feature; k++) {
                double p;
                if (mode == 0) {
                    p = gaussian(test[i][k], train_class_mu[j][k],
                                 train_class_sd[j][k]);
                    if (p <= pow(10, -10))
                        p = pow(10, -10);
                } else
                    p = train_class_bin[j][k][(int) test[i][k] >> 3];
                predict[i][j] += log(p);
            }
            if (mode == 0)
                predict[i][j] +=
                    log((double) count_class[j] / (double) num_train);
            else /*need to calculate probabiblity at each pixel*/
                predict[i][j] -= (size_feature - 1) * log(count_class[j]);
            marginal += predict[i][j];
        }
        for (int j = 0; j < num_class; j++) {
            printf("%d: %.17f\n", j, predict[i][j] / marginal);
        }
        int result = -1;
        double max = -99999999999999;
        for (int j = 0; j < num_class; j++) {
            if (max < predict[i][j]) {
                max = predict[i][j];
                result = j;
            }
        }

        printf("Prediction: %d, Ans: %d\n", result, target[i]);
        count += (result == target[i]) ? 1 : 0;
        printf("Error: %4.0f/%4d\n", i - count + 1, i + 1);
        printf("\033[13A\033[K");
    }

    printf("\033[J");
    printf("* Imagination of numbers in Bayesian classifier\n");
    for (int i = 0; i < num_class; i++) {
        for (int j = 0; j < size_feature; j++) {
            if (j % 28 == 0)
                printf("\n");
            if (mode == 0) {
                if (train_class_mu[i][j] >= 128)
                    printf("1");
                else
                    printf("0");
            } else {
                if (train_class_mu[i][j] >= 16)
                    printf("1");
                else
                    printf("0");
            }
        }
        printf("\n");
    }
    printf("\n\n  Accuracy: %.4f\n", count / num_test);
    printf("Error rate: %.4f\n", (1 - (count / num_test)));
    return (1 - (count / num_test));
}

void free_NB()
{
    free(count_class);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 784; j++)
            free(train_class_bin[i][j]);
        free(train_class_mu[i]);
        free(train_class_sd[i]);
        free(train_class_bin[i]);
    }
    for (int i = 0; i < 10000; i++)
        free(predict[i]);
    free(train_class_mu);
    free(train_class_sd);
    free(train_class_bin);
    free(predict);
}

/*
 * mode: 0 -> Continuous mode
 *       1 -> Discrete mdoe
 */
double run_naive_bayes_classifier(double **train_data,
                                  uint8_t *train_class,
                                  double **test_data,
                                  uint8_t *test_class,
                                  int num_train,
                                  int num_test,
                                  int size_feature,
                                  int num_class,
                                  int mode)
{
    double ret;
    printf("Starting Naive Bayes Classifier...\n\n");
    statistic_data(train_data, train_class, num_train, size_feature, num_class,
                   mode);
    ret = prediction(test_data, test_class, num_train, num_test, size_feature,
                     num_class, mode);
    free_NB();
    printf("Finishing Naive Bayes Classifier...\n\n");
    return ret;
}
