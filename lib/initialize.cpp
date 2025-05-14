#include "initialize.h"

static std::random_device rd;  
static std::mt19937_64 gen(rd());

double random_real(double initial, double last) {
	std::uniform_real_distribution<double> distribution(initial, last);
	return distribution(gen); 
}

int random_int(int initial, int last) {
	std::uniform_int_distribution<int> distribution(initial, last);
	return distribution(gen);  // Use rng as a generator
}

std::vector <admixture> random_initial(int pop, int ind, int K, std::vector <admixture> admx){

	//admixture admx[pop], *admxp;
	Eigen::MatrixXd temp(ind,K), phi(K,K), TM(ind,ind);
	double sum(ind);
	int maxRow,maxCol;
	std::uniform_int_distribution<int> dis(0, 1);

	for(int i=0; i<pop; i++){
		for(int j=0; j<K ; j++){
			for(int k=0; k<K; k++){
				phi(j,k)=random_real(0,1);
			}
		}
		phi=( phi + phi.transpose().eval() )/2;
		double max = phi.maxCoeff(&maxRow, &maxCol);
		if(max>1) phi=phi/max;
		admx[i].phi=phi;
	
		temp=Eigen::MatrixXd::Constant(ind, K, 0);
		for(int j=0; j<ind ; j++){
			//sum=temp.row(j).sum();
			temp(j,random_int(0,K-1))=1;
		}
		admx[i].Q=temp;
		admx[i].obj=0.0;
	}

	return admx;
}

std::vector <admixture> read_initial(int pop, int ind, int K, std::vector <admixture> admx){

	std::ifstream myfile;
	Eigen::MatrixXd temp(ind,K), phi(K,K);
	for(int i=0; i<pop; i++){
		std::string file = "Q_"+std::to_string(i+1)+".txt";
		myfile.open(file);
		for (int x = 0; x < ind; x++) {
			for (int y = 0; y < K; y++) {
				myfile >> temp(x,y);
  			}
		}
		myfile.close();

		file = "P_"+std::to_string(i+1)+".txt";
		myfile.open(file);
		for (int x = 0; x < K; x++) {
			for (int y = 0; y < K; y++) {
				myfile >> phi(x,y);
			}
		}
		myfile.close();

		admx[i].Q=temp;
		admx[i].obj=0.0;
		admx[i].phi=phi;

	}
	return admx;
}
