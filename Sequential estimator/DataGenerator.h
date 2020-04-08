#include <vector>
/*
 * DataGenerator(mean, varience) or
 * DataGenerator(basis, varience, weight) and assume mean=0
 *
 * 
*/
class DataGenerator{
	public:
		DataGenerator(double, double);
		DataGenerator(int, double, double *);
		void setMean(double);
		void setVarience(double);
		void setBasis(int);
		void setWeight(double *);
		double getMean();
		double getVarience();
		double *getWeight();
		int getBasis();
		std::vector<double> random(int);
		//double *polynomial();

	private:
		double mean;
		double varience;
		int basis;
		double *weight;
		std::vector<double> marsaglia();
};
