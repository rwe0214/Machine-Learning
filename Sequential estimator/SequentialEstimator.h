#ifndef __SEQUENTIALESTIMATOR_H
#define __SEQUENTIALESTIMATOR_H

class SequentialEstimator
{
public:
    SequentialEstimator(double, double);
    void setMean(double);
    void setVarience(double);
    double getMean();
    double getVarience();
    double updateMean(double);
    double updateVarience(double);
    void addNewData(double);

private:
    double mean;
    double varience;
    double n;  // numbers of data
};

#endif