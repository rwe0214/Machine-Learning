#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include "DataGenerator.h"
#include "SequentialEstimator.h"
#include "Matrix.h"
#include "BayesianLinearRegression.h"
#define ITER_MAX 10000
using namespace std;

int main()
{
    /*data generator*/
    double u_0 = 3, v_0 = 5, varience = 1.0, b=1.0;
    vector<double> w;
    int basis = 4;
    for (int i = 1; i <= basis; i++)
        w.push_back(i);

    DataGenerator rgen(u_0, v_0);
    DataGenerator pgen(basis, varience, w);
    vector<double> r = rgen.random(ITER_MAX);
    vector<vector<double> > p = pgen.polynomial(ITER_MAX);

    ofstream myfile;
    myfile.open("randomGaussian.data");
    for (int i = 0; i < r.size(); i++)
        myfile << r.at(i) << endl;
    myfile.close();
    myfile.open("polynomialLinearModel.data");
    for (int i = 0; i < basis-1; i++)
        myfile << w.at(i) << ',';
    myfile << w.at(basis-1) << endl;
    myfile << varience << endl;
    myfile.close();
    myfile.open("polynomialLinearModelPredict.data");
    for (int i = 0; i < p.size(); i++){
        if((p.at(i)).at(0) < 1 && (p.at(i)).at(0) > -1){
            myfile << (p.at(i)).at(0) << ',';
            myfile << (p.at(i)).at(1) << endl;
        }
    }
    myfile.close();

    /*sequential estimator*/
    double converge = 0.001;
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

    /*bayesian linear regression*/
    BayesianLinearRegression blr(basis, varience, b); 
    Matrix diff_u(basis, 1);
    double delta = 1;
    i = 0;
    converge = 0.0000001;
    do{
        printf("[%6d]Add data point (%.5f, %.5f):\n\n", i+1, (p.at(i)).at(0), (p.at(i)).at(1));
        diff_u = blr.getPosterior().at(0);
        blr.addNewData(p.at(i++));
        diff_u = diff_u - blr.getPosterior().at(0);
        printf("Postirior mean:\n");
        blr.getPosterior().at(0).print();
        printf("\nPosterior variance:\n");
        blr.getPosterior().at(1).print();
        printf("--------------------------------------------------\n");

        if(i>1){
            delta = 0.0;
            for(unsigned i=0; i<basis; i++)
                delta += abs(diff_u(i, 0));
            delta /= basis;
        }
    }while((i<ITER_MAX) && delta>converge);


}