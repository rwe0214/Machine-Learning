#include <fstream>
#include <iostream>
#include <vector>

#include "DataGenerator.h"

using namespace std;

int main()
{
    vector<double> w;
    int basis = 3;
    for (int i = basis; i >= 0; i--)
        w.push_back(i);

    DataGenerator rgen(10, 25);
    DataGenerator pgen(basis, 25, w);
    vector<double> r = rgen.random(10000);
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
}