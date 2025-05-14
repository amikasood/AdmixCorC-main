#include "compete.h"

std::vector <admixture> Compete(std::vector <admixture> combined, int pop, int ind, int K)
{
	//std::cout<<"inside compete....\n";
	int pop2=2*pop;
	std::vector<admixture> admx(pop);
	Eigen::VectorXd fcombined(pop2), p(pop2), sum(pop2), f0(pop);

	for(int i =0; i<pop2; i++)
	{
		fcombined[i]=combined[i].obj;
	}
	//std::cout<<fcombined<<"\n";
	//std::cout<<pop<<"\n\n";
	//double s = *max_element(fcombined.begin(), fcombined.end()) + *min_element(fcombined.begin(), fcombined.end());
	double s = fcombined.maxCoeff();
	fcombined = s-fcombined.array();
	p = fcombined/fcombined.sum();
	sum[0]=p[0];
	for(int i = 1; i<pop2; i++) sum[i]=sum[i-1]+p[i];

	int track=0, length=0, j, t=0;

	while(length<pop)
	{
		double r = random_real(0,1);
		//std::cout<<"random number..."<<length<<" "<<r<<" "<<sum[0]<<" "<<track<<"\n";
		if(r<sum[0])
		{
			j=0;
			if((f0.array() == (s-fcombined[0])).any()==0)
			{
				f0[track]=s-fcombined[j];
				admx[track]=combined[j];
			//	std::cout<<"objectine 1 "<<combined[j].obj<<"\n";
				track++;
				t=1;
			}
		}
		else	
		{
			for(j=1; j<pop2; j++)
			{
				if((r>sum[(j-1)]) && (r<sum[j])) break;
			}
			if((f0.array() == (s-fcombined[j])).any()==0)
			{
				f0[track]=s-fcombined[j];
				admx[track]=combined[j];
			//	std::cout<<"objectine2 "<<combined[j].obj<<" "<<j<<" "<<track<<"\n";
				track++;
				t=1;
			}
		}


		if(t==1) length++;
		t=0;
	}

	//for(int i=0; i<pop;i++) admx[i]=combined[i];

	return admx;
}
