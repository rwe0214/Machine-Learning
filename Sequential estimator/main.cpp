#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include "BayesianLinearRegression.h"
#include "DataGenerator.h"
#include "Matrix.h"
#include "SequentialEstimator.h"
#define ITER_MAX 10000
using namespace std;

int main()
{
    /*data generator*/
    double u_0 = 3, v_0 = 5, varience = 1.0, b = 1.0;
    vector<double> w;
    int basis = 4;
    for (int i = 1; i <= basis; i++)
        w.push_back(i);

    DataGenerator rgen(u_0, v_0);
    DataGenerator pgen(basis, varience, w);
    vector<double> r = rgen.random(ITER_MAX);
    vector<vector<double> > p = pgen.polynomial(ITER_MAX);

    ofstream myfile;
    if (!!access("output", 0))
        mkdir("output", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    myfile.open("output/polynomialLinearModel.data");
    for (int i = 0; i < basis - 1; i++)
        myfile << w.at(i) << ',';
    myfile << w.at(basis - 1) << endl;
    myfile << varience << endl;
    myfile.close();

    /*sequential estimator*/
    printf(
        "======================= Sequential Estimator Start "
        "=======================\n");
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
    printf(
        "======================== Sequential Estimator End "
        "========================\n\n");

    /*bayesian linear regression*/
    printf("b=%f, n=%d, a=%f, w=[", b, basis, varience);
    for (int i = 0; i < basis - 1; i++)
        printf("%f, ", w.at(i));
    printf("%f]\n", w.at(basis - 1));
    printf(
        "==================== Bayesian Linear Regression Start "
        "====================\n");
    BayesianLinearRegression blr(basis, varience, b);
    Matrix diff_u(basis, 1);
    double delta = 1;
    i = 0;
    converge = 0.0000001;
    myfile.open("output/bayesianLinearRegressionPredict.data");
    ofstream income_10, income_50;
    ofstream income_10_model, income_50_model;
    income_10.open("output/bayesianLinearRegressionPredict10.data");
    income_50.open("output/bayesianLinearRegressionPredict50.data");
    income_10_model.open("output/bayesianLinearRegressionPredictModel10.data");
    income_50_model.open("output/bayesianLinearRegressionPredictModel50.data");

    do {
        diff_u = blr.getPosterior().at(0);
        myfile << p.at(i).at(0) << "," << p.at(i).at(1) << endl;
        if (i < 10)
            income_10 << p.at(i).at(0) << "," << p.at(i).at(1) << endl;
        if (i < 50)
            income_50 << p.at(i).at(0) << "," << p.at(i).at(1) << endl;
        blr.addNewData(p.at(i++));
        diff_u = diff_u - blr.getPosterior().at(0);
        if (i % (ITER_MAX / 15) == 0 && i != ITER_MAX) {
            printf("[%6d]Add data point (% .5f, % .5f):\n\n", i,
                   (p.at(i - 1)).at(0), (p.at(i - 1)).at(1));
            printf("Postirior mean:\n");
            blr.getPosterior().at(0).print();
            printf("\nPosterior variance:\n");
            blr.getPosterior().at(1).print();
            printf("\nPredict distribution ~ N(% .5f, % .5f)\n",
                   blr.getPredict().at(0), blr.getPredict().at(1));
            printf(
                "--------------------------------------------------------------"
                "----------\n");
        }
        if (i > 1) {
            delta = 0.0;
            for (unsigned i = 0; i < basis; i++)
                delta += abs(diff_u(i, 0));
        }
        if (i == 10) {
            for (unsigned i = 0; i < basis - 1; i++)
                income_10_model << blr.getPosterior().at(0)(i, 0) << ",";
            income_10_model << blr.getPosterior().at(0)(basis - 1, 0) << endl;
            income_10_model << blr.getPredict().at(1) << endl;
        }
        if (i == 50) {
            for (unsigned i = 0; i < basis - 1; i++)
                income_50_model << blr.getPosterior().at(0)(i, 0) << ",";
            income_50_model << blr.getPosterior().at(0)(basis - 1, 0) << endl;
            income_50_model << blr.getPredict().at(1) << endl;
        }
    } while ((i < ITER_MAX) && delta > 3 * converge);
    printf("[%6d]Add data point (%.5f, %.5f):\n\n", i, (p.at(i - 1)).at(0),
           (p.at(i - 1)).at(1));
    printf("Postirior mean:\n");
    blr.getPosterior().at(0).print();
    printf("\nPosterior variance:\n");
    blr.getPosterior().at(1).print();
    printf("\nPredict distribution ~ N(% .5f, % .5f)\n", blr.getPredict().at(0),
           blr.getPredict().at(1));
    printf(
        "----------------------------------------------------------------------"
        "--\n");
    printf(
        "===================== Bayesian Linear Regression End "
        "=====================\n");
    myfile.close();
    myfile.open("output/bayesianLinearRegressionPredictModel.data");
    for (unsigned i = 0; i < basis - 1; i++)
        myfile << blr.getPosterior().at(0)(i, 0) << ",";
    myfile << blr.getPosterior().at(0)(basis - 1, 0) << endl;
    myfile << blr.getPredict().at(1) << endl;
    myfile.close();
    income_10.close();
    income_50.close();
    income_10_model.close();
    income_50_model.close();
}