#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include "DataGenerator.h"
#include "SequentialEstimator.h"
#define ITER_MAX 50000
using namespace std;

int main()
{
    /*data generator*/
    double u_0 = 3, v_0 = 5;
    vector<double> w;
    int basis = 3;
    for (int i = basis; i >= 0; i--)
        w.push_back(i);

    DataGenerator rgen(u_0, v_0);
    DataGenerator pgen(basis, 25, w);
    vector<double> r = rgen.random(ITER_MAX);
    vector<vector<double> > p = pgen.polynomial(10000);

    ofstream myfile;
    myfile.open("randomGaussian.data");
    for (int i = 0; i < r.size(); i++)
        myfile << r.at(i) << endl;
    myfile.close();
    myfile.open("polynomialLinearModel.data");
    for (int i = 0; i < p.size(); i++)
        myfile << (p.at(i)).at(0) << ',' << (p.at(i)).at(1) << ','
               << (p.at(i)).at(2) << endl;
    myfile.close();

    /*sequential estimator*/
    double converge = 0.0001;
    int i = 0;
    double delta_u, delta_v;
    printf("Data point source function: N(%f, %f)\n\n", u_0, v_0);
    SequentialEstimator se(r.at(i++), 0);

    do {
        delta_u = se.getMean();
        delta_v = se.getVarience();
        se.addNewData(r.at(i++));
        delta_u = abs(delta_u - se.getMean());
        delta_v = abs(delta_v - se.getVarience());
        if ((i - 1) % (ITER_MAX / 15) == 0 && i != ITER_MAX) {
            printf("[%6d] Add data point: %.15f\n", i - 1, r.at(i - 1));
            printf("\t Mean = %.15f,\t", se.getMean());
            printf("Varience = %.15f\n", se.getVarience());
        }
    } while ((i < ITER_MAX) && ((delta_u > converge) || (delta_v > converge)));
    printf("[%6d] Add data point: %.15f\n", i - 1, r.at(i - 1));
    printf("\t Mean = %.15f,\t", se.getMean());
    printf("Varience = %.15f\n", se.getVarience());
}