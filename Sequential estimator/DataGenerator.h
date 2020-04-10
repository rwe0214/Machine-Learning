#include <vector>
/*
 * DataGenerator(mean, varience) or
 * DataGenerator(basis, varience, weight) and assume mean=0
 *
 *
 */
class DataGenerator
{
public:
    DataGenerator(double, double);
    DataGenerator(int, double, std::vector<double>);
    void setMean(double);
    void setVarience(double);
    void setBasis(int);
    void setWeight(std::vector<double>);
    double getMean();
    double getVarience();
    std::vector<double> getWeight();
    int getBasis();
    std::vector<double> random(int);
    std::vector<std::vector<double> > polynomial(int);

private:
    double mean;
    double varience;
    int basis;
    int mode;
    std::vector<double> weight;
    std::vector<double> marsaglia();
    std::vector<double> getPolynomial();
};
