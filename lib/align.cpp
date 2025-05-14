#include "align.h"

//functor
template<class ArgType, class RowIndexType, class ColIndexType>
class indexing_functor {
	  const ArgType &m_arg;
	    const RowIndexType &m_rowIndices;
	      const ColIndexType &m_colIndices;
	public:
	        typedef Eigen::Matrix<typename ArgType::Scalar,
			                 RowIndexType::SizeAtCompileTime,
					                  ColIndexType::SizeAtCompileTime,
							                   ArgType::Flags&Eigen::RowMajorBit?Eigen::RowMajor:Eigen::ColMajor,
									                    RowIndexType::MaxSizeAtCompileTime,
											                     ColIndexType::MaxSizeAtCompileTime> MatrixType;

		  indexing_functor(const ArgType& arg, const RowIndexType& row_indices, const ColIndexType& col_indices)
			      : m_arg(arg), m_rowIndices(row_indices), m_colIndices(col_indices)
				  {}

		    const typename ArgType::Scalar& operator() (Eigen::Index row, Eigen::Index col) const {
			        return m_arg(m_rowIndices[row], m_colIndices[col]);
				  }
};

//function
template <class ArgType, class RowIndexType, class ColIndexType>
Eigen::CwiseNullaryOp<indexing_functor<ArgType,RowIndexType,ColIndexType>, typename indexing_functor<ArgType,RowIndexType,ColIndexType>::MatrixType>
indexing(const Eigen::MatrixBase<ArgType>& arg, const RowIndexType& row_indices, const ColIndexType& col_indices)
{
	  typedef indexing_functor<ArgType,RowIndexType,ColIndexType> Func;
	    typedef typename Func::MatrixType MatrixType;
	      return MatrixType::NullaryExpr(row_indices.size(), col_indices.size(), Func(arg.derived(), row_indices, col_indices));
}

std::vector <admixture> Align(std::vector <admixture> admx, int pop, int ind, int K)
{
	std::vector<std::pair<double, int> > f0;
	Eigen::ArrayXi indices(K);
	Eigen::ArrayXi rows(ind);
	for(int i = 0; i<ind; i++)
		rows(i)=i;
	for (int i = 0; i < pop; ++i) {
		f0.push_back(std::make_pair(admx[i].obj, i));
	}

	std::sort(f0.begin(), f0.end());

	int i_best = f0[0].second;
	Eigen::MatrixXd Q1(ind,K), Q2(ind,K), Q3(ind,K), P2(K,K);

	Q1= admx[f0[0].second].Q;

	for(int i=1; i<pop; i++)
	{
		int x = f0[i].second;
		Q2 = admx[x].Q;
		P2 = admx[x].Q;

		indices = Align_Q(Q1, Q2, ind, K);

		Q2 = indexing(Q2, rows, indices);
		P2 = indexing(P2, indices, indices);

		//std::cout<<"aligned\n"<<indices<<"\n\n"<<Q2<<"\n\n\nphi:\n"<<P2<<"\n\n----";

		admx[x].Q = Q2;
		admx[x].phi = P2;

	}

	return admx;
}

Eigen::ArrayXi Align_Q(Eigen::MatrixXd Q1, Eigen::MatrixXd Q2, int ind, int K)
{
	Eigen::MatrixXd KL(K,K);
	double err = 100;
	int permut[K];
        Eigen::ArrayXi indices(K); 
	for(int i=0; i<K; i++)
	{
		permut[i] = i;
	}

	std::sort (permut,permut+K);

	KL = KL_Q_mat(Q1, Q2, ind, K);
	do{
		//std::cout << permut[0] << ' ' << permut[1] << ' ' << permut[2] << '\n';
		double e = 0;
		for(int j=0; j<K; j++)
		{
			e = e + KL(j, permut[j]);
		}
		if(err>e)
		{
			err = e;
			//std::copy(std::begin(permut), std::end(permut), std::begin(indices));
			for(int j = 0; j<K; j++)
				indices(j)=permut[j];

		}
	}
	while( std::next_permutation(permut,permut+3) );
	
	return indices;
}

Eigen::MatrixXd KL_Q_mat(Eigen::MatrixXd Q1, Eigen::MatrixXd Q2, int ind, int K)
{
	Eigen::MatrixXd KL(K,K);
	Eigen::ArrayXd qi(ind), q2j(ind);

	for(int i=0; i<K; i++)
	{
		qi = Q1.col(i);
		for(int j =0; j<K; j++)
		{
			q2j = Q2.col(j);
			KL(i,j) = KL_Q_mat_col(qi, q2j, ind);
		}

	}

	return KL;

}

double KL_Q_mat_col(Eigen::ArrayXd Q1, Eigen::ArrayXd Q2, int ind)
{
	double k1 = 0;
	for(int j = 0; j<ind; j++)
	{
		if(Q1[j] > 0 && Q2[j] > 0 )
		{
			k1 = k1 + Q1[j] * log(Q1[j]/ Q2[j]);
		}
	}

	k1 = k1/ind;
	return k1;

}
