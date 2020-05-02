#include "LogisticRegressor.h"

#include <cmath>

#include "Matrix.h"

LogisticRegressor::LogisticRegressor(int newton,
                                     int n,
                                     double converage,
                                     vector<vector<double> > data,
                                     vector<int> targets)
{
    setNewton(newton);
    setSize(n);
    setConverage(converage);
    setX(data);
    this->W.resize(3);
    this->W[0].resize(1);
    this->W[1].resize(1);
    this->W[2].resize(1);
    setTarget(targets);
    this->predict.resize(n);
}


void LogisticRegressor::run()
{
    Matrix x(this->X);
    Matrix xt = x.transpose();
    Matrix w(this->W);
    Matrix y(this->target);
    Matrix tmp(this->size, 1);
    Matrix tmp2(3, this->size);
    Matrix eye(this->size, 1);
    Matrix d(this->size, this->size);
    Matrix I(this->size, 1, 1);
    Matrix hf(3, 3);
    Matrix gradient(3, 1);
    double diff = 9999;
    int t = 0;
    int cur_conv = 0;
    int max_conv = 10;
    int max_t = pow(10, 4);

    while (1) {
        printf("iteration: %5d/%5d\r", t++, max_t);
        tmp = x * w;
        tmp = tmp * (-1);
        tmp = tmp.expMatrix();
        tmp = I + tmp;
        tmp = tmp.inverse();
        eye = tmp;
        tmp = tmp - y;
        gradient = xt * tmp;

        for (unsigned i = 0; i < this->size; i++)
            eye(i, 0) = pow(eye(i, 0), 2);
        for (unsigned i = 0; i < d.getRowSize(); i++)
            d(i, i) = eye(i, 0);
        tmp2 = xt * d;
        hf = tmp2 * x;
        if (hf.determinant() != 0 && this->newton_enable) {
            hf = hf.inverse();
            gradient = hf * gradient;
        }
        diff = gradient(0, 0) + gradient(1, 0) + gradient(2, 0);
        w = w - gradient;
        if (diff < converage) {
            cur_conv++;
            if (cur_conv == max_conv)
                break;
        } else
            cur_conv = 0;

        if (t == max_t)
            break;
    }
    printf("iteration: %5d", t);
    for (int i = 0; i < this->size; i++) {
        double p = x(i, 0) * w(0, 0) + x(i, 1) * w(1, 0) + x(i, 2) * w(2, 0);
        this->predict[i] = (p > 0.5 ? 1 : 0);
    }
    vector<vector<double> > retw;
    retw.resize(3);
    for (int i = 0; i < 3; i++)
        retw[i].push_back(w(i, 0));
    setW(retw);
}

void LogisticRegressor::setNewton(int n)
{
    this->newton_enable = n;
}

void LogisticRegressor::setSize(int n)
{
    this->size = n;
}

void LogisticRegressor::setConverage(double converage)
{
    this->converage = converage;
}

/*
 * X = [
 * [x0, y0, 1],
 * [x1, y1, 1],
 * ...
 * [xn-1, yn-1, 1]
 * ]
 */
void LogisticRegressor::setX(vector<vector<double> > data)
{
    this->X.resize(this->size);
    for (int i = 0; i < this->size; i++) {
        X[i].resize(3);
        for (int j = 0; j < data[i].size(); j++)
            this->X[i][j] = data[i][j];
        this->X[i][2] = 1;
    }
}

/*
 * W = [
 * [Wx],
 * [Wy],
 * [W0]
 * ]
 */
void LogisticRegressor::setW(vector<vector<double> > w)
{
    this->W = w;
}

void LogisticRegressor::setTarget(vector<int> targets)
{
    this->target = targets;
}

void LogisticRegressor::setPredict(vector<int> predicts)
{
    this->predict = predicts;
}

int LogisticRegressor::getSize()
{
    return this->size;
}

double LogisticRegressor::getConverage()
{
    return this->converage;
}

vector<vector<double> > LogisticRegressor::getX()
{
    return this->X;
}

vector<vector<double> > LogisticRegressor::getW()
{
    return this->W;
}

vector<int> LogisticRegressor::getTarget()
{
    return this->target;
}

vector<int> LogisticRegressor::getPredict()
{
    return this->predict;
}