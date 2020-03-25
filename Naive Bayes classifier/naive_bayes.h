#ifndef __NAIVE_BAYES_H
#define __NAIVE_BAYES_H

double get_gaussian(double x, double mu, double sd);
void train_model();
void print_model();
void run_naive_bayes_classifier(uint8_t *train_data[],
                                uint8_t train_target[],
                                uint8_t *test_data[],
                                uint8_t test_target[]);

#endif
