#include <random>
#include <ctime>
#include <cmath>
#include "DataGenerator.h"

DataGenerator::DataGenerator(double mean, double varience){
	setMean(mean);
	setVarience(varience);
};

DataGenerator::DataGenerator(int basis, double varience, double *weight){
	setMean(0.0);
	setBasis(basis);
	setVarience(varience);
	setWeight(weight);
};

void DataGenerator::setMean(double mean){
	this->mean = mean;
};

void DataGenerator::setVarience(double varience){
	this->varience = varience;
};

void DataGenerator::setBasis(int basis){
	this->basis = basis;
};

void DataGenerator::setWeight(double *weight){
	this->weight = weight;
};

double DataGenerator::getMean(){
	return this->mean;
};

double DataGenerator::getVarience(){
	return this->varience;
};

int DataGenerator::getBasis(){
	return this->basis;
};

double *DataGenerator::getWeight(){
	return this->weight;
};

std::vector<double> DataGenerator::marsaglia(){
	std::random_device rd;
  	std::mt19937 gen( rd() );
	std::uniform_real_distribution<double> unid(-1.0, 1.0);
	double u, v, s;
	do{
		u = unid(gen);
		v = unid(gen);
		s = pow(u, 2) + pow(v, 2);
	}while(s >= 1);

	s = sqrt(-2*log(s)/s);

	std::vector<double> ret;
	ret.push_back(u*s);
	ret.push_back(v*s);
	return ret;
}

std::vector<double> DataGenerator::random(int num){
	double sigma = sqrt(varience);
	std::vector<double> d;
	std::vector<double> ret;

	for(int i=0; i<num; i+=2){
		d = marsaglia();
		ret.push_back(mean + sigma*d[0]);
		ret.push_back(mean + sigma*d[1]);
	}
	return ret;
}