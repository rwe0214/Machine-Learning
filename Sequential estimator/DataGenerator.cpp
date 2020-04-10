#include "DataGenerator.h"

#include <cmath>
#include <ctime>
#include <random>

DataGenerator::DataGenerator(double mean, double varience)
{
    this->mode = 0;
    setMean(mean);
    setVarience(varience);
};

/*
 * TODO:
 * Add exception handler when weight size is less then basis
 */
DataGenerator::DataGenerator(int basis,
                             double varience,
                             std::vector<double> weight)
{
    this->mode = 1;
    setMean(0.0);
    setBasis(basis);
    setVarience(varience);
    setWeight(weight);
};

void DataGenerator::setMean(double mean)
{
    this->mean = mean;
};

void DataGenerator::setVarience(double varience)
{
    this->varience = varience;
};

void DataGenerator::setBasis(int basis)
{
    this->basis = basis;
};

void DataGenerator::setWeight(std::vector<double> weight)
{
    this->weight = weight;
};

double DataGenerator::getMean()
{
    return this->mean;
};

double DataGenerator::getVarience()
{
    return this->varience;
};

int DataGenerator::getBasis()
{
    return this->basis;
};

std::vector<double> DataGenerator::getWeight()
{
    return this->weight;
};

std::vector<double> DataGenerator::marsaglia()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> unid(-1.0, 1.0);
    double u, v, s;
    do {
        u = unid(gen);
        v = unid(gen);
        s = pow(u, 2) + pow(v, 2);
    } while (s >= 1);

    s = sqrt(-2 * log(s) / s);

    std::vector<double> ret;
    ret.push_back(u * s);
    ret.push_back(v * s);
    return ret;
}

std::vector<double> DataGenerator::getPolynomial()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> unid(-1.0, 1.0);
    double x = unid(gen);
    std::vector<double> ret(2);

    ret[0] = x;
    for (int i = 0; i < this->basis; i++)
        ret[1] += this->weight.at(i) * pow(x, i);
    return ret;
}

std::vector<std::vector<double> > DataGenerator::polynomial(int num)
{
    std::vector<std::vector<double> > ret(num);
    if (mode == 1) {
        std::vector<double> poly;
        DataGenerator error(0, this->varience);
        std::vector<double> e = error.random(num);

        for (int i = 0; i < num; i++) {
            poly = this->getPolynomial();
            ret[i].push_back(poly.at(0));
            ret[i].push_back(poly.at(1) + e.at(i));
            ret[i].push_back(poly.at(1));
        }
    } else
        printf("This is not a polynomial linear mode generator\n");
    return ret;
}

std::vector<double> DataGenerator::random(int num)
{
    std::vector<double> ret;
    if (mode == 0) {
        double sigma = sqrt(varience);
        std::vector<double> d;
        for (int i = 0; i < num; i++) {
            if (i % 2 == 0)
                d = marsaglia();
            ret.push_back(mean + sigma * d[i % 2]);
        }
    } else
        printf("This is not a random data generator\n");
    return ret;
}
