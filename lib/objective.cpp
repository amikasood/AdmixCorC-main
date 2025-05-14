#include "objective.h"

double objective(Eigen::MatrixXd kin, Eigen::MatrixXd Q, Eigen::MatrixXd phi, int ind, int K, double gamma){
	double f = (kin-(Q*phi*Q.transpose().eval())).squaredNorm()/kin.norm() + (2*gamma*phi.trace())/K;
	return f;
}
