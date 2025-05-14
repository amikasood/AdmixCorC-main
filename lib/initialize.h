#ifndef INITIALIZE_H
#define INITIALIZE_H

#include <Eigen>
#include <random>
#include <iostream>
#include <fstream>

//using namespace Eigen;

struct admixture {
	Eigen::MatrixXd Q;
	Eigen::MatrixXd phi;
	double obj;
};

std::vector <admixture> random_initial(int pop, int ind, int K, std::vector <admixture> admx);
std::vector <admixture> read_initial(int pop, int ind, int K, std::vector <admixture> admx);
double random_real(double initial, double last);
int random_int(int initial, int last);

#endif /* INITIALIZE_H */
