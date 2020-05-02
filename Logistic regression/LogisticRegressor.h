#ifndef __LOGISTICREGRESSOR_H
#define __LOGISTICREGRESSOR_H
#include <vector>

using std::vector;

class LogisticRegressor
{
public:
    LogisticRegressor(int, int, double, vector<vector<double> >, vector<int>);
    void run();
    void setNewton(int);
    void setSize(int);
    void setConverage(double);
    void setX(vector<vector<double> >);
    void setW(vector<vector<double> >);
    void setTarget(vector<int>);
    void setPredict(vector<int>);
    int getSize();
    double getConverage();
    vector<vector<double> > getX();
    vector<vector<double> > getW();
    vector<int> getTarget();
    vector<int> getPredict();

private:
    int size;
    int newton_enable;
    double converage;
    vector<vector<double> > X;
    vector<vector<double> > W;
    vector<int> target;
    vector<int> predict;
};

#endif