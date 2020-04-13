#include "SequentialEstimator.h"

#include <cmath>

SequentialEstimator::SequentialEstimator(double mean, double varience)
{
    this->mean = mean;
    this->varience = varience;
    this->n = 1.0;
}

void SequentialEstimator::setMean(double mean)
{
    this->mean = mean;
}

void SequentialEstimator::setVarience(double varience)
{
    this->varience = varience;
}

double SequentialEstimator::getMean()
{
    return this->mean;
}

double SequentialEstimator::getVarience()
{
    return this->varience;
}

double SequentialEstimator::updateMean(double x)
{
    return ((this->n - 1) * this->mean + x) / this->n;
}

double SequentialEstimator::updateVarience(double x)
{
    return ((this->n - 1) / this->n) *
           (this->varience + (pow(x - this->mean, 2) / this->n));
}

void SequentialEstimator::addNewData(double data)
{
    this->n++;
    setVarience(updateVarience(data));
    setMean(updateMean(data));
}