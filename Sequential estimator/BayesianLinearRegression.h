#include "Matrix.h"
#include <vector>
using std::vector;

class BayesianLinearRegression{
    public:
        BayesianLinearRegression(unsigned, double, double);
        void setA(double);
        void setPrior(vector<Matrix>);
        void setPosterior(vector<Matrix>);
        void setBasis(unsigned);
        void setPredict(double, double);
        double getA();
        vector<Matrix> getPrior();
        vector<Matrix> getPosterior();
        unsigned getBasis();
        vector<double> getPredict();
        vector<Matrix> update(vector<double>);
        void addNewData(vector<double>);
    private:
        double a;
        vector<Matrix> prior;
        vector<Matrix> posterior;
        unsigned basis;
        vector<double> predict;
};