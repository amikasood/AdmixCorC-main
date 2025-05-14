#include <Eigen>
#include <random>
#include <iostream>
#include "initialize.h"
#include "objective.h"
#include "step_size.h"

std::vector <admixture> GDQPhi(Eigen::MatrixXd theta, std::vector <admixture> admx, int pop, int ind, int K, double gamma);
std::vector <admixture> GDPhi(Eigen::MatrixXd theta, std::vector <admixture> admx, int pop, int ind, int K, double gamma);
