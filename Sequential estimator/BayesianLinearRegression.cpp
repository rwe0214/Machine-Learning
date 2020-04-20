#include "BayesianLinearRegression.h"

#include <cmath>

BayesianLinearRegression::BayesianLinearRegression(unsigned basis,
                                                   double a,
                                                   double b)
{
    this->basis = basis;
    this->a = 1 / a;
    this->prior.push_back(Matrix(basis, 1));
    this->prior.push_back(Matrix(basis, basis));
    for (unsigned i = 0; i < this->prior.at(1).getRow(); i++)
        for (unsigned j = 0; j < this->prior.at(1).getCol(); j++)
            if (i == j)
                prior.at(1)(i, j) = b;
    this->posterior.push_back(Matrix(basis, 1));
    this->posterior.push_back(Matrix(basis, basis));
    this->predict.resize(3);
};

void BayesianLinearRegression::setA(double a)
{
    this->a = a;
};

void BayesianLinearRegression::setPrior(vector<Matrix> p)
{
    this->prior = p;
};

void BayesianLinearRegression::setPosterior(vector<Matrix> p)
{
    this->posterior = p;
};

void BayesianLinearRegression::setBasis(unsigned basis)
{
    this->basis = basis;
};

void BayesianLinearRegression::setPredict(double mean,
                                          double varience,
                                          double predict)
{
    this->predict[0] = mean;
    this->predict[1] = varience;
    this->predict[2] = predict;
};

double BayesianLinearRegression::getA()
{
    return this->a;
};

vector<Matrix> BayesianLinearRegression::BayesianLinearRegression::getPrior()
{
    return this->prior;
};

vector<Matrix> BayesianLinearRegression::getPosterior()
{
    return this->posterior;
};

unsigned BayesianLinearRegression::getBasis()
{
    return this->basis;
};

vector<double> BayesianLinearRegression::getPredict()
{
    return this->predict;
};

vector<Matrix> BayesianLinearRegression::update(vector<double> data)
{
    // b_new = a * x_t * x + b

    Matrix x(1, this->basis);
    for (double i = 0; i < x.getCol(); i++)
        // x = [x^0, x^1, ..., x^{n-1}]
        x(0, i) = pow(data[0], i);

    Matrix x_t = x.transpose();
    Matrix b = this->prior[1].inverse();

    Matrix b_new = (x_t * x * this->a) + b;

    /*
     * mu_new = a * b_new^-1 * x_t * y + b_new^-1 * b * mu
     *        = b_new^-1 * ( a*x_t*y + b*mu )
     * where a = constant
     */

    Matrix tmp = b * this->prior[0];
    tmp = x_t * data[1] * this->a + tmp;
    Matrix mu_new = b_new.inverse() * tmp;

    vector<Matrix> ret;
    ret.push_back(mu_new);
    ret.push_back(b_new.inverse());

    double predict = (x * mu_new)(0, 0);
    setPredict((x * ret.at(0))(0, 0), 1 / this->a + (x * ret.at(1) * x_t)(0, 0),
               predict);
    return ret;
};

void BayesianLinearRegression::addNewData(vector<double> data)
{
    setPosterior(update(data));
    setPrior(this->posterior);
};

vector<vector<double> > BayesianLinearRegression::showModel()
{
    vector<vector<double> > ret(40);
    for (int i = 0; i < 40; i++) {
        Matrix x(1, this->basis);
        for (int j = 0; j < x.getCol(); j++)
            x(0, j) = pow(-2 + (double) (0.1 * i), j);
        Matrix x_t = x.transpose();
        ret.at(i).push_back(x(0, 1));
        ret.at(i).push_back((x * this->posterior.at(0))(0, 0));
        ret.at(i).push_back(1 / this->a +
                            (x * this->posterior.at(1) * x_t)(0, 0));
    }
    return ret;
}