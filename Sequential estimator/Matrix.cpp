#include "Matrix.h"

#include <cmath>

Matrix::Matrix(unsigned row, unsigned col)
{
    this->row_size = row;
    this->col_size = col;
    this->ele.resize(this->row_size);
    for (unsigned i = 0; i < this->ele.size(); i++)
        ele[i].resize(this->col_size, 0.0);
};

Matrix::Matrix(unsigned row, unsigned col, double init_val)
{
    this->row_size = row;
    this->col_size = col;
    this->ele.resize(this->row_size);
    for (unsigned i = 0; i < this->ele.size(); i++)
        ele[i].resize(this->col_size, init_val);
};

double &Matrix::operator()(const unsigned &row, const unsigned &col)
{
    return this->ele[row][col];
};

Matrix Matrix::operator+(Matrix &B)
{
    Matrix sum(this->row_size, this->col_size);
    for (unsigned i = 0; i < this->row_size; i++) {
        for (unsigned j = 0; j < this->col_size; j++)
            sum(i, j) = this->ele[i][j] + B(i, j);
    }
    return sum;
};

Matrix Matrix::operator-(Matrix &B)
{
    Matrix diff(this->row_size, this->col_size);
    for (unsigned i = 0; i < this->row_size; i++) {
        for (unsigned j = 0; j < this->col_size; j++)
            diff(i, j) = this->ele[i][j] - B(i, j);
    }
    return diff;
};

Matrix Matrix::operator*(Matrix &B)
{
    Matrix product(this->row_size, B.getCol());
    if (this->col_size != B.getRow())
        throw "The two matries cannot multiply";
    for (unsigned i = 0; i < this->row_size; i++) {
        for (unsigned j = 0; j < B.getCol(); j++)
            for (unsigned k = 0; k < this->col_size; k++)
                product(i, j) += this->ele[i][k] * B(k, j);
    }
    return product;
};

Matrix Matrix::operator*(double scalar)
{
    Matrix s(this->row_size, this->col_size);
    for (unsigned i = 0; i < this->row_size; i++) {
        for (unsigned j = 0; j < this->col_size; j++)
            s(i, j) = this->ele[i][j] * scalar;
    }
    return s;
};

Matrix Matrix::transpose()
{
    Matrix m_t(this->col_size, this->row_size);
    for (unsigned i = 0; i < this->row_size; i++)
        for (unsigned j = 0; j < this->col_size; j++)
            m_t(j, i) = this->ele[i][j];
    return m_t;
};

Matrix Matrix::minor(unsigned row, unsigned col)
{
    Matrix m(this->row_size - 1, this->col_size - 1);
    unsigned r = 0, l = 0;
    for (unsigned i = 0; i < this->row_size; i++)
        for (unsigned j = 0; j < this->col_size; j++)
            if (i != row && j != col) {
                m(r, l++) = this->ele[i][j];
                if (l == m.col_size) {
                    l = 0;
                    r++;
                }
            }
    return m;
};

double Matrix::determinant()
{
    if (this->row_size != this->col_size)
        throw "This is not a square matrix";
    if (this->row_size == 1)
        return this->ele[0][0];
    else if (this->row_size == 2)
        return this->ele[0][0] * this->ele[1][1] -
               this->ele[0][1] * this->ele[1][0];
    else {
        double sum = 0.0;
        for (int j = 0; j < this->col_size; j++)
            sum += (pow(-1, 0 + j) * this->ele[0][j] *
                    (this->minor(0, j)).determinant());
        return sum;
    }
};

Matrix Matrix::adjugate()
{
    Matrix cof(this->row_size, this->col_size);
    for (unsigned i = 0; i < this->row_size; i++)
        for (unsigned j = 0; j < this->col_size; j++)
            cof(i, j) = (pow(-1, i + j) * this->minor(i, j).determinant());
    return cof.transpose();
};

/*only work on square matrix*/
Matrix Matrix::inverse()
{
    if (this->determinant() == 0)
        throw "This matrix do not have inverse";
    Matrix adj = this->adjugate();
    double det = this->determinant();
    Matrix m_i(this->row_size, this->col_size);
    for (unsigned i = 0; i < this->row_size; i++)
        for (unsigned j = 0; j < this->col_size; j++)
            m_i(i, j) = adj(i, j) / det;
    return m_i;
};

void Matrix::print()
{
    for (unsigned i = 0; i < this->row_size; i++) {
        printf("\t");
        for (unsigned j = 0; j < this->col_size - 1; j++)
            printf("% .10f,\t", this->ele[i][j]);
        printf("% .10f\n", this->ele[i][this->col_size - 1]);
    }
};

unsigned Matrix::getRow()
{
    return this->row_size;
};

unsigned Matrix::getCol()
{
    return this->col_size;
};