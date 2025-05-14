#include "mutation.h"


std::vector <admixture> Mutation(Eigen::MatrixXd theta, std::vector <admixture> admx, int pop, int ind, int K, double gamma)
{
	std::vector<std::pair<double, int> > f0;
	Eigen::MatrixXd Q(ind, K), temp(ind,K);
	Eigen::MatrixXd J = Eigen::MatrixXd::Constant(ind, K, 1.0);

	for (int i = 0; i < pop; ++i) {
		f0.push_back(std::make_pair(admx[i].obj, i));
	}
	
	std::sort(f0.begin(), f0.end());
	
	int x = random_int(0,pop-1);

	for (int i = pop-1; i > x; i--)
	{
		int y = f0[i].second;
		Q = admx[y].Q;
		temp = (Eigen::MatrixXd::Random(ind,K)+J)/2;
		
		for(int j=0; j<ind; j++){
			double sum=temp.row(j).sum();
			temp.row(j)=temp.row(j)/sum;
		}
		Q = (Q + temp)/2;
		admx[y].Q=Q;

	}

	return admx;
}
