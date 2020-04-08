#include "DataGenerator.h"
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

int main(){
    DataGenerator gen(5, 100);

    vector<double> v1 = gen.random(10000);

    ofstream myfile;
    myfile.open ("randomGaussian.data");
    for(int i=0; i< v1.size(); i++)
        myfile << v1.at(i) << endl;
    myfile.close();
}