#include <Eigen>
#include <random>
#include <iostream>
#include "initialize.h"

std::vector <admixture> Mutation(Eigen::MatrixXd theta, std::vector <admixture> admx, int pop, int ind, int K, double gamma);
