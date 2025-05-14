#include <Eigen>
#include <random>
#include <iostream>
#include <math.h>
#include "initialize.h"

std::vector <admixture> Align(std::vector <admixture> combined, int pop, int ind, int K);
Eigen::ArrayXi Align_Q(Eigen::MatrixXd Q1, Eigen::MatrixXd Q2, int ind, int K);
Eigen::MatrixXd KL_Q_mat(Eigen::MatrixXd Q1, Eigen::MatrixXd Q2, int ind, int K);
double KL_Q_mat_col(Eigen::ArrayXd Q1, Eigen::ArrayXd Q2, int ind);
