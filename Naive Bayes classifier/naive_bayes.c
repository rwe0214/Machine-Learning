#include <math.h>
#include <stdio.h>

#include "mnist.h"

/*
 * P(A,B,C,D|X) = P(A|X)P(B|X)P(C|X)P(D|X)P(X) / P(A,B,C,D)
 *
 * => p(label|pixels) = product( p(pixel|label) )p(label) / p(pixels) = a/b
 *	 p(~label|pixels) = c/b
 *
 * p(label|pixels) / (p(label|pixels)+p(~label|pixels))
 * = (a/b) / (a/b + c/b)
 * = a / (a+c)
 *
 * a = product( p(pixel|label) )p(label)
 * c = product( p(pixel|~label) )p(~label)
 *
 */

double get_gaussian(double x, double mu, double sd)
{
    double e = pow(x - mu, 2) / (2 * sd * sd) * (-1);
    return exp(e) / (sd * sqrt(2 * M_PI));
}

/*TODO*/
void train_model()
{
    ;
}
/*TODO*/
void print_model()
{
    ;
}
/*TODO*/
void run_naive_bayes_classifier(uint8_t *train_data[],
                                uint8_t train_target[],
                                uint8_t *test_data[],
                                uint8_t test_target[])
{
    ;
}
