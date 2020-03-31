#ifndef __NAIVE_BAYES_H
#define __NAIVE_BAYES_H
#include <stdint.h>

double gaussian(double x, double mu, double sd);
void statistic_data(double **data,
                    uint8_t *target,
                    int num,
                    int size,
                    int num_class,
                    int mode);
double prediction(double **test,
                  uint8_t *target,
                  int num_train,
                  int num_test,
                  int size_feature,
                  int num_class,
                  int mode);
double run_naive_bayes_classifier(double **train_data,
                                  uint8_t *train_class,
                                  double **test_data,
                                  uint8_t *test_class,
                                  int num_train,
                                  int num_test,
                                  int size_feature,
                                  int num_class,
                                  int mode);
void free_NB();

#endif
