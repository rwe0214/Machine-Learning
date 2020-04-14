#ifndef __MATRIX_H
#define __MATRIX_H

#include <vector>
using std::vector;

class Matrix{
    public:
        Matrix(unsigned, unsigned);
        Matrix(unsigned, unsigned, double);
        Matrix operator+(Matrix &);
        Matrix operator-(Matrix &);
        Matrix operator*(Matrix &);
        Matrix operator*(double);
        double& operator()(const unsigned &, const unsigned &);
        Matrix transpose();
        Matrix minor(unsigned, unsigned);
        double determinant();
        Matrix adjugate();
        Matrix inverse();
        void print();
        unsigned getRow();
        unsigned getCol();
        
    private:
        unsigned row_size;
        unsigned col_size;
        vector<vector<double> > ele;
};
#endif