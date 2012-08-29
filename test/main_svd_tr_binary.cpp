/* run_svd_tr.cpp */

/* Usage: mpic++ -I/home/wuhong_intern/local/include -L/home/wuhong_intern/local/lib -std=c++0x -std=gnu++0x -lmpi -fopenmp -I/usr/include/mysql      -Wall -O3 main_svd_tr.cpp -llapack -lgfortran -lblas -lcblas -o main_svd_tr
*/

#include <map>
#include <mpi.h>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <memory.h>
#include <douban/matvec.hpp>
//#include <douban/linalg/svd.hpp>
#include <douban/linalg/gemv_ax.hpp>
#include <douban/matvec/mat_container.hpp>
#include <douban/matvec/vec_kahan_gemv.hpp>
#include <douban/mpi/linalg/mpi_svd_tr.hpp>
#include <douban/option_parser.hpp>

int main(int argc, char *argv[]) {

	int rank, nprocs;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	int m, n, m_At, n_At, nnz, k = 200, p = 200, display = 10;
	int len, begin, end, s_indx, t_s_indx;
	std::map<int, int> word_kind;
	FILE *fp, *fp2;
	
	douban::option_parser cmd_parser;
	cmd_parser.parse_all(argc, argv);
	
	fp = fopen("../data/pwtk.bin", "rb");
	fread(&m, sizeof(int), 1, fp);
	fread(&n, sizeof(int), 1, fp);
	fread(&nnz, sizeof(int), 1, fp);
	
	fp2 = fopen("../data/pwtk_trans.bin", "rb");
	fread(&m_At, sizeof(int), 1, fp2);
	fread(&n_At, sizeof(int), 1, fp2);
	fread(&nnz, sizeof(int), 1, fp2);
	
	douban::mat_container<double> U(m, k);
	douban::mat_container<double> V(n, k);
	douban::vec_container<double> S(k);

	std::vector<size_t> ar, ac, Ap, Ai, atr, atc, Atp, Ati;
	std::vector<double> av, atv, Av, Atv;
	len = nnz / nprocs;	
	begin = rank * len;
	end = (rank + 1) * len;
	if(rank == nprocs - 1) {
		end = nnz;
		len = nnz - begin;
	}

	std::cout << len << std::endl;

	ar.resize(len);
	ac.resize(len);
	av.resize(len);
	atr.resize(len);
	atc.resize(len);
	atv.resize(len);

	fseek(fp, (begin * (sizeof(int)*2 + sizeof(double))), SEEK_CUR);
	for(size_t i = 0; i < (size_t)len; ++i) {
		fread(&(ar[i]), sizeof(int), 1, fp);
		fread(&(ac[i]), sizeof(int), 1, fp);
		fread(&(av[i]), sizeof(double), 1, fp);
	}

	fseek(fp2, (begin * (sizeof(int)*2 + sizeof(double))), SEEK_CUR);
	for(size_t i = 0; i < (size_t)len; ++i) {
		fread(&(atr[i]), sizeof(int), 1, fp2);
		fread(&(atc[i]), sizeof(int), 1, fp2);
		fread(&(atv[i]), sizeof(double), 1, fp2);
	}

	m = 0;
	m_At = 0;
	
	for(size_t i = 0; i < (size_t)len; ++i) { 
		if(ar[i] != ar[i+1])
			m++;
		if(atr[i] != atr[i+1])
			m_At++;
	}
	
	/*
	m = ar[len - 1] - ar[0] + 1;
	m_At = atr[len - 1] - atr[0] + 1;
	*/

	std::cout << "local m is " << m << std::endl;
	std::cout << "local m_At is " << m_At << std::endl;
	
	s_indx = ar[0];
	t_s_indx = atr[0];
	if(rank != 0) {
		for(size_t i = 0; i < (size_t)len; ++i) ar[i] -= s_indx;
		for(size_t i = 0; i < (size_t)len; ++i) atr[i] -= t_s_indx;	
	}
	
	Ap.resize(m + 1);
	Av.resize(len);
	Ai.resize(len);

	Atp.resize(m_At + 1);
	Atv.resize(len);
	Ati.resize(len);
        
	douban::coo_to_csr((size_t)m, (size_t)n, (size_t)len, av, ar, ac, Av, Ap, Ai);	
	douban::coo_to_csr((size_t)m_At, (size_t)n_At, (size_t)len, atv, atr, atc, Atv, Atp, Ati);	

	auto A = douban::make_mat_csr((size_t)m, (size_t)n, &Av, &Ap, &Ai);
	auto At = douban::make_mat_csr((size_t)m_At, (size_t)n_At, &Atv, &Atp, &Ati);
	
	douban::linalg::svd_tr(A, At, U, S, V, p, 1.e-7, display, s_indx, t_s_indx);
        //douban::linalg::svd_tr(douban::linalg::make_gemv_ax(&A), douban::linalg::make_gemv_ax(&At), U, S, V, p, 1.e-7, display, s_indx, t_s_indx);
	//std::cout << "svd_tr finished!" << std::endl;
	//cost = douban::linalg::svd_cost(A, U, S, V);
	//std::cout << "cost of SVD is " << cost << std::endl; 
	for(size_t i = 0; i < (size_t)k; i++)
	  std::cout << "S(i) is" << S.get(i) << std::endl;
        	
	fclose(fp);
	fclose(fp2);
	MPI_Finalize();

	return 0;
}
