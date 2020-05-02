#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

#include "DataGenerator.h"
#include "LogisticRegressor.h"
#include "Matrix.h"

using namespace std;

int n;
double mx1, my1, mx2, my2;
double vx1, vy1, vx2, vy2;
double converage;

vector<double> x1, y1;
vector<double> x2, y2;
vector<vector<double> > train_data;
vector<int> target;

Matrix *weight_gradient;
Matrix *weight_newton;
vector<int> predict_gradient;
vector<int> predict_newton;

void read_config();
void init_train();
void output(LogisticRegressor, LogisticRegressor);

int main()
{
    read_config();
    init_train();
    LogisticRegressor lr_gradient(0, 2 * n, converage, train_data, target);
    LogisticRegressor lr_newton(1, 2 * n, converage, train_data, target);

    output(lr_gradient, lr_newton);
}

void read_config()
{
    ifstream fin;
    fin.open("config.data");
    char line[10];
    fin >> n;
    fin >> line;
    sscanf(line, "%lf,%lf,%lf,%lf", &mx1, &my1, &mx2, &my2);
    fin >> line;
    sscanf(line, "%lf,%lf,%lf,%lf", &vx1, &vy1, &vx2, &vy2);
    fin.close();
}

void init_train()
{
    converage = 0.0000001;
    DataGenerator gx1(mx1, vx1), gy1(my1, vy1);
    DataGenerator gx2(mx2, vx2), gy2(my2, vy2);

    x1 = gx1.random(n);
    y1 = gy1.random(n);
    x2 = gx2.random(n);
    y2 = gy2.random(n);
    train_data.resize(2 * n);

    for (int i = 0; i < n; i++) {
        train_data[i].push_back(x1.at(i));
        train_data[i].push_back(y1.at(i));
        target.push_back(0);
    }

    for (int i = 0; i < n; i++) {
        train_data[i + n].push_back(x2.at(i));
        train_data[i + n].push_back(y2.at(i));
        target.push_back(1);
    }
}

void output(LogisticRegressor gradient, LogisticRegressor newton)
{
    ofstream fout;
    if (!!access("output", 0))
        mkdir("output", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    fout.open("output/train.data");

    fout << 2 * n << endl;
    for (int i = 0; i < n; i++)
        fout << x1.at(i) << "," << y1.at(i) << ",0" << endl;
    for (int i = 0; i < n; i++)
        fout << x2.at(i) << "," << y2.at(i) << ",1" << endl;
    fout.close();

    printf("Gradient descent:\n\n");
    gradient.run();
    predict_gradient = gradient.getPredict();
    double tp = 0.0, fp = 0.0, tn = 0.0, fn = 0.0;
    for (int i = 0; i < 2 * n; i++) {
        if (predict_gradient.at(i) == target.at(i) &&
            predict_gradient.at(i) == 1)
            tp++;
        else if (predict_gradient.at(i) != target.at(i) &&
                 predict_gradient.at(i) == 1)
            fp++;
        else if (predict_gradient.at(i) == target.at(i) &&
                 predict_gradient.at(i) == 0)
            tn++;
        else if (predict_gradient.at(i) != target.at(i) &&
                 predict_gradient.at(i) == 0)
            fn++;
    }
    printf("\n\nw:\n");
    weight_gradient = new Matrix(gradient.getW());
    weight_gradient->print();
    printf("\nConfusion Matrix:\n");
    printf("\t\tPredict cluster 1 Predict cluster 2\n");
    printf("Is cluster 1\t%17.0f %17.0f\n", tn, fp);
    printf("Is cluster 2\t%17.0f %17.0f\n\n", fn, tp);
    printf("Sensitivity (Successfully predict cluster 1): %lf\n",
           (tn / (fp + tn)));
    printf("Specificity (Successfully predict cluster 2): %lf\n\n",
           (tp / (tp + fn)));
    printf(
        "-----------------------------------------------------------------\n");
    fout.open("output/predict_gradient.data");
    fout << (*weight_gradient)(0, 0) << "," << (*weight_gradient)(1, 0) << ","
         << (*weight_gradient)(2, 0) << endl;
    for (int i = 0; i < n; i++)
        fout << x1.at(i) << "," << y1.at(i) << "," << predict_gradient.at(i)
             << endl;
    for (int i = 0; i < n; i++)
        fout << x2.at(i) << "," << y2.at(i) << "," << predict_gradient.at(n + i)
             << endl;
    fout.close();

    printf("Newton's method:\n\n");
    newton.run();
    predict_newton = newton.getPredict();
    tp = 0.0, fp = 0.0, tn = 0.0, fn = 0.0;
    for (int i = 0; i < 2 * n; i++) {
        if (predict_newton.at(i) == target.at(i) && predict_newton.at(i) == 1)
            tp++;
        else if (predict_newton.at(i) != target.at(i) &&
                 predict_newton.at(i) == 1)
            fp++;
        else if (predict_newton.at(i) == target.at(i) &&
                 predict_newton.at(i) == 0)
            tn++;
        else if (predict_newton.at(i) != target.at(i) &&
                 predict_newton.at(i) == 0)
            fn++;
    }
    printf("\n\nw:\n");
    weight_newton = new Matrix(newton.getW());
    weight_newton->print();
    printf("\nConfusion Matrix:\n");
    printf("\t\tPredict cluster 1 Predict cluster 2\n");
    printf("Is cluster 1\t%17.0f %17.0f\n", tn, fp);
    printf("Is cluster 2\t%17.0f %17.0f\n\n", fn, tp);
    printf("Sensitivity (Successfully predict cluster 1): %lf\n",
           (tn / (fp + tn)));
    printf("Specificity (Successfully predict cluster 2): %lf\n\n",
           (tp / (tp + fn)));
    printf(
        "-----------------------------------------------------------------\n");
    fout.open("output/predict_newton.data");
    fout << (*weight_newton)(0, 0) << "," << (*weight_newton)(1, 0) << ","
         << (*weight_newton)(2, 0) << endl;
    for (int i = 0; i < n; i++)
        fout << x1.at(i) << "," << y1.at(i) << "," << predict_newton.at(i)
             << endl;
    for (int i = 0; i < n; i++)
        fout << x2.at(i) << "," << y2.at(i) << "," << predict_newton.at(n + i)
             << endl;
    fout.close();
}