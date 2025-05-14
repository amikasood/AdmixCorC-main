#include "gradient_descent.h"

std::vector <admixture> GDQPhi(Eigen::MatrixXd theta, std::vector <admixture> admx, int pop, int ind, int K, double gamma){
	
	double normz=theta.norm(), f, f1, ffinal;
	Eigen::MatrixXd Q(ind, K), phi(K, K), QT(K, ind), dQ(ind, K), Q1(ind, K), QT1(K, ind), dphi(K, K), diag(K, K), phi1(K, K), Qfinal(ind, K), phifinal(K, K);
	Eigen::VectorXd f0(ind), diagv = Eigen::VectorXd::Constant(K, 1.0);
	Eigen::ArrayXd rowmean(ind);
	int nstep, ndQ, ndphi;
	diag = diagv.asDiagonal();

	for(int i=0; i<pop; i++){

		f=100;
		nstep=1;
		ndQ=100;
		ndphi=100;
		Q=admx[i].Q;

                QT=Q.transpose().eval();
                phi=admx[i].phi;

		while(ndQ>5e-5 || ndphi>5e-5){
			//optimize Q
			dQ = 4*( (Q*phi*QT-theta)*Q*phi ) / (normz);
			//rowmean=dQ.rowwise().mean();
			//dQ = dQ.array()-rowmean;
			//dQ = dQ.array() - dQ.rowwise().mean(); //------ to be included later

			//to get step size (backtracking line search)
			double lrq = get_max_step_size_admix(Q, dQ, ind, K);
			double T0 = objective(theta, Q, phi, ind, K, gamma);
			double dQ0 = dQ.squaredNorm()/2;
			
			while( objective(theta,Q-lrq*dQ,phi,ind,K,gamma) > (T0-(lrq*dQ0)) )
			{
				lrq=lrq/2;
			}
			Q1 = Q-(dQ*lrq);

			for(int j=0; j<ind; j++){
				Q1.row(j) = (Q1.row(j).array() <0.0 ).select(Q1.row(j).array()-Q1.row(j).minCoeff()+0.0001, Q1.row(j).array());
				double sum=Q1.row(j).sum();
				Q1.row(j)=Q1.row(j)/sum;
			}
			QT1 = Q1.transpose().eval();

			//optimize phi
			dphi = 2* (QT1*((Q1*phi*QT1)-theta)*Q1)/normz + (2*gamma*diag)/K;
			//to get the step size
			double lrp = get_max_step_size_admix(phi, dphi, K, K);
			T0 = objective(theta, Q1, phi, ind, K, gamma);
			double dphi0 = dphi.squaredNorm()/2;
			
			while(objective(theta, Q1, phi-lrp*dphi, ind, K, gamma) > (T0-(lrp*dphi0)))
			{
				lrp=lrp*0.5;
			}

			phi1 = phi-(lrp*dphi);
			phi1 = (phi1.array() < 0.0).select(phi1.array() - phi1.minCoeff(),phi1.array());
			double max = phi1.maxCoeff();
			if(max > 1.0) {phi1 = phi1.array()/max;}

			f1 = objective(theta, Q1, phi1, ind, K, gamma);
			ndQ<-(Q-Q1).norm()/sqrt(ind*K);
 			ndphi<-(phi-phi1).norm()/K;

			if(f>f1){
				ffinal=f1;
				Qfinal=Q1;
				phifinal=phi1;
			}
			Q=Q1;
			QT=Q.transpose().eval();
			phi=phi1;
			//f=f1;
			nstep=nstep+1;
	
			if(nstep>250) {break;}
		}
		
		admx[i].Q=Qfinal;
		admx[i].phi=phifinal;
		admx[i].obj=ffinal;

	}

	return admx;
	
}

std::vector <admixture> GDPhi(Eigen::MatrixXd theta, std::vector <admixture> admx, int pop, int ind, int K, double gamma)
{
	double normz=theta.norm(), f, f1, ffinal;
        Eigen::MatrixXd Q(ind, K), phi(K, K), QT(K, ind), dphi(K, K), diag(K, K), phi1(K, K), phifinal(K, K);
        Eigen::VectorXd f0(ind), diagv = Eigen::VectorXd::Constant(K, 1.0);
        int nstep, ndphi;
        diag = diagv.asDiagonal();

	for(int i=0; i<pop; i++)
	{
                f=100;
                nstep=1;
                ndphi=100;
                Q=admx[i].Q;
                QT=Q.transpose().eval();
                phi=admx[i].phi;

		while(ndphi>5e-5)
		{
			dphi = 2* (QT*((Q*phi*QT)-theta)*Q)/normz + (2*gamma*diag)/K;
                        //to get the step size
			double lrp = get_max_step_size_admix(phi, dphi, K, K);
			double T0 = objective(theta, Q, phi, ind, K, gamma);
			double dphi0 = dphi.squaredNorm()/2;

			while(objective(theta, Q, phi-lrp*dphi, ind, K, gamma) > (T0-(lrp*dphi0)))
                        {
	                        lrp=lrp*0.5;
			}

			phi1 = phi-(lrp*dphi);
                        phi1 = (phi1.array() < 0.0).select(phi1.array() - phi1.minCoeff(),phi1.array());
                        double max = phi1.maxCoeff();
                        if(max > 1.0) {phi1 = phi1/max;}

			f1 = objective(theta, Q, phi1, ind, K, gamma);
                        ndphi<-(phi-phi1).norm()/K;

			if(f>f1){
	                        ffinal=f1;
				phifinal=phi1;
			}

			phi=phi1;
			//f=f1;
			nstep=nstep+1;
			if(nstep>250) {break;}
		}

		admx[i].phi=phifinal;
		admx[i].obj=ffinal;
	}

	return admx;
}
