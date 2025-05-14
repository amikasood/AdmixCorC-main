#include <iostream>
#include <unistd.h>
#include <cstdio>
#include <stdlib.h>
#include <fstream>
#include <map>
#include <Eigen>
 
#include "../lib/read_file.h"
#include "../lib/gradient_descent.h"
#include "../lib/recombine.h"
#include "../lib/mutation.h"
#include "../lib/compete.h"
#include "../lib/align.h"

//using namespace Eigen;

int main(int argc, char **argv) {

	char *conf = NULL;
	int index, parse;
	int K, pop = 3, gstep = 2, mstep = 2, ind;
	std::string kin;
	input kinship;
	Eigen::MatrixXd theta;
	double gamma, normz;

	gamma=0.025;

	//Parse command line arguments
	while ((parse = getopt (argc, argv, "hc:")) != -1){
		switch (parse)
		{
			case 'c':
			        conf = optarg;
			        break;
			case 'h':
				std::cout << "help needed \n";
				exit(0);
			default:/* '?' */
				fprintf(stderr, "Usage: ./main -c conf \n");
				exit(EXIT_FAILURE);
		}
	}
	
	if(argc==1){
		fprintf(stderr, "Usage: ./main -c conf \n");
		fprintf(stderr, "Usage: ./main -h \n");
		exit(EXIT_FAILURE);
	}
	for (index = optind; index < argc; index++){
	      printf ("Non-option argument %s\n", argv[index]);
	      exit(EXIT_FAILURE);
	}

	//Read config file
	std::cout<<"Reading config file...\n";
	kvpair config_value;
	config_value=read_config(conf);
	for(kvpair::iterator it = config_value.begin(); it != config_value.end(); ++it) {
		if(it->first=="Kinship"){
			kin = it->second;
		}
		if(it->first=="K"){
			K = atoi(it->second.c_str());
		}
		if(it->first=="pop"){
                        pop = atoi(it->second.c_str());
		}
		if(it->first=="gamma"){
			gamma = atof(it->second.c_str());
		}
		if(it->first=="gstep"){
                        gstep = atoi(it->second.c_str());
		}
		if(it->first=="mstep"){
                        mstep = atoi(it->second.c_str());
		}
	}
	pop=pop*K;
	std::vector<admixture> admx(pop);
	std::vector<admixture> admxc(pop);
	std::vector<admixture> admxcombined(pop*2);
	admixture best;

	//Read Kinship file
	std::cout <<"Reading Kinship file...\n";
	kinship=read_kinship(kin);
	theta=kinship.theta;
	ind=kinship.ind;

	std::cout <<"Initializing Ancestry and Ancestry Kinship matrices...\n";	
	//admx=random_initial(pop, ind, K, admx);
	admx= read_initial(pop, ind, K, admx);

	std::cout <<"Optimizing Ancestry and Ancestry Kinship matrices...\n";

	admx=GDQPhi(theta, admx, pop, ind, K, gamma);
	std::vector<std::pair<double, int> > f0;
	for (int i = 0; i < pop; ++i) {
		f0.push_back(std::make_pair(admx[i].obj, i));
	}
       	std::sort(f0.begin(), f0.end());
	best.Q=admx[f0[0].second].Q;
	best.phi=admx[f0[0].second].phi;
	best.obj=admx[f0[0].second].obj;
	f0.clear();

	std::cout <<"Begin Memetic Optimization...\n";
	int nstep = 1;
	int minstep = 0;
	
	while(minstep<mstep)
	{
		//---------Align Q matrices--------//
		admx=Align(admx, pop, ind, K);

		//Recombination
		admxc=Recombine(admx, ind, K);

		//Optimize P matrix corresponding to the mutated Q matrix
		admxc=GDPhi(theta, admxc, pop, ind, K, gamma);

		//Mutation
		admxc=Mutation(theta, admxc, pop, ind, K, gamma);

		//GD after gstep
		//Optimize the Q and P matrices for childern after gstep
		if(nstep%gstep==0){
			admxc=GDQPhi(theta, admxc, pop, ind, K, gamma);
		}

		//admxcombined.insert( admxcombined.end(), admx.begin(), admx.end() );
		admxcombined=admx;
		admxcombined.insert( admxcombined.end(), std::make_move_iterator(admxc.begin()), std::make_move_iterator(admxc.end()) );

		for (int i = 0; i < 2*pop; ++i) {
			f0.push_back(std::make_pair(admxcombined[i].obj, i));
		}
		std::sort(f0.begin(), f0.end());
		if(best.obj>f0[0].first)
		{
			best.Q=admxcombined[f0[0].second].Q;
			best.phi=admxcombined[f0[0].second].phi;
			best.obj=admxcombined[f0[0].second].obj;
			minstep=0;
		}
		else {minstep++; }
		f0.clear();

		//Compete
		admx=Compete(admxcombined,pop,ind,K);
		nstep++;

	}

	std::cout<<"\nbest obj: \n"<<objective(theta, best.Q, best.phi, ind, K, gamma)/(ind*ind)<<"\n";
	std::cout<<"best phi: \n"<<best.phi<<"\n";
	//std::cout<<"best Q: \n"<<best.Q<<"\n";
	admx.clear();
        admxc.clear();
        admxcombined.clear();


	return 0;
}
