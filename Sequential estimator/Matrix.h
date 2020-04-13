#include <vector>
using std::vector;

class Matrix{
    public:
        Matrix(unsigned, unsigned, double);
        Matrix transpose();
        Matrix inverse();
        Matrix operator+(Matrix &);
        Matrix operator-(Matrix &);
        Matrix operator*(Matrix &);
        double& operator()(const unsigned &, const unsigned &);
        void operator()(const unsigned &, const unsigned &, double);
        unsigned getRow();
        unsigned getCol();
        
    private:
        unsigned row;
        unsigned col;
        vector<vector<double> > ele;
};