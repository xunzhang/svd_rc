/**
 * @file mpi_bidiag.hpp
 * @author wu hong<xunzhangthu@gmail.com>
 * @brief mpi version of bidiagonalization, One-Sided Lanczos Bidiag. (restarted, with enhancements)
 *
 */

#ifndef FILE_8180cdd3_0cd2_4606_8640_9c0792474dde_H
#define FILE_8180cdd3_0cd2_4606_8640_9c0792474dde_H
#include "douban/linalg/householder.hpp"
#include "douban/linalg/gemv_ax.hpp"
#include "douban/matvec/vec_kahan.hpp"
#include "douban/utility/kahan_sum.hpp"
#include "douban/matvec/io.hpp"
#include "douban/mpi.hpp"

#include <sys/time.h>

static double currenttime(void) {
double timestamp;
  struct timeval tv;
  gettimeofday(&tv, 0);
  timestamp = (double)((double)(tv.tv_sec*1e6) + (double)tv.tv_usec);
  return timestamp;
}

namespace douban {
namespace linalg {

template <class CM, class CRecv>
void local_union(CM && mat, CRecv && recv_tmp, int indx, int nprocs) {
  for(size_t i = 0; i < mat.dim0(); ++i) {
    mat.col(indx)[i] = 0;
    for(size_t j = 0; j < (size_t)nprocs; ++j)
      mat.col(indx)[i] += recv_tmp[j * mat.dim0() + i];
      //std::cout << "mat.col value is " << mat.col(indx)[i] << std::endl;
  }
}


template<class CM, class CV, class CR>
void parallel_gemv_task(CM && mat, CV && vec, CR && res) {
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   
  int m = (int)mat.dim0();// n = (int)mat.dim1();
  
  if(m < nprocs) {
    vec_container<double> tmp_y(m);
    tmp_y = 0;
    tmp_y = gemv(mat, vec); 
    res = 0;
    MPI_Allreduce(&tmp_y[0], &res[0], m, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return ;
  } 
  
  vec_container<double> ttmp(vec.size(), 0);
  ttmp = vec;
  MPI_Allreduce(&ttmp[0], &vec[0], vec.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
  int load = m / nprocs;
   
  int remainder = m % nprocs;
  int max_load = load + remainder;
  int *rcounts = new int [nprocs];
  int *displs = new int [nprocs];
  int offset;
  offset = rank * load;
  
  douban::vec_container<double> y_tmp(max_load, 0);
   
  // General matrix vector multiplication
  
  y_tmp = douban::gemv(mat_rows(mat, offset, offset + max_load), vec);
  
  // Prepare for MPI_Gatherv
  for(int i = 0; i < nprocs; ++i) {
    rcounts[i] = load;
    displs[i] = i * load;
  }
  if(remainder != 0)
    rcounts[nprocs - 1] = max_load;
  if(rank == nprocs - 1)
    load = max_load;
  
  // MPI_Gatherv
  MPI_Gatherv(&y_tmp[0], load, MPI_DOUBLE, &res[0], &rcounts[0], &displs[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  MPI_Bcast(&res[0], res.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  delete [] rcounts;
  delete [] displs;
  return;
}

template<class CAX, class CATX, class CD, class CE, class CRho, class CP, class CQ>
void bidiag_gkl_restart(int locked, int l, int n,
     			CAX && Ax, CATX && Atx, CD && D, CE && E, CRho && rho, CP && P, CQ && Q, int s_indx, int t_s_indx) {
    
    // enhancements version from SLEPC
    int rank, nprocs;
    const double eta = 1.e-10;
    int debug_count = 0; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    auto m_Ax = make_gemv_ax(&Ax);
    auto m_Atx = make_gemv_ax(&Atx);
    
    // step 1
    vec_container<double> temp(Ax.dim0(), 0);
    vec_container<double> send_data(P.dim0(), 0); 
    vec_container<double> recv_tmp((int)P.dim0() * nprocs);
    double time1 = 0.0, time2 = 0.0; 
    time1 = currenttime();
    m_Ax(Q.col(l), temp, P.dim0() > 1000);
    time2 = currenttime();
    std::cout << "matrix vector mul time is " << (time2 - time1) / 1.0e6 << std::endl;

    for(size_t i = s_indx; i < s_indx + Ax.dim0(); ++i)
      send_data[i] = temp[i - s_indx];
    MPI_Gather(&send_data[0], P.dim0(), MPI_DOUBLE, &recv_tmp[0], P.dim0(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //P.col(0) = 0;
    if(rank == 0) {
      local_union(P, recv_tmp, l, nprocs);
      // step 2
      for(int j = locked; j < l; ++j)
        P.col(l) += -rho(j) * P.col(j);
    }
    // to be modified!!
    MPI_Bcast(&(P.col(0)[0]), P.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
     
    vec_container<double> T(n, 0);
    
    for (int j = l; j < n; ++j) {
      debug_count ++;
      vec_container<double> tmp0(Atx.dim0(), 0); // n1, n2 
      vec_container<double> tmp(Q.dim0(), 0); // n
      // step 3
      tmp0 = 0;
      m_Atx(P.col(j), tmp0, Q.dim0() > 1000);
       
      // everything is OK in loop2!!
      tmp = 0;
      for(size_t bc = t_s_indx; bc < t_s_indx + Atx.dim0(); ++bc)
        tmp[bc] = tmp0[bc - t_s_indx];
       
      // step 4
      auto Qj = mat_cols(Q, 0, j + 1);
      auto Tj = make_vec(&T, j + 1);
        
      Tj = 0;
      
      parallel_gemv_task(Qj.trans(), tmp, Tj);
      
      /*
      if(rank == 0) {
        for(size_t bc = 0; bc < Tj.size(); ++bc)
          std::cout << "Tj " << Tj[bc] << std::endl;
        std::cout << "barrier" << std::endl;
      }
      */

      for(size_t bc = t_s_indx; bc < t_s_indx + Atx.dim0(); ++bc)
        tmp0[bc - t_s_indx] = tmp[bc]; 
       
      // step 5 
      // everything is OK in loop2, tmp0!!!
      double r = tmp0.norm2(); 
      //std::cout << "first r is " << r * r << std::endl;
      // everything is OK in loop2, first r is!!!
      double power_r = 0.0;
      power_r = r * r;
      D[j] = vec_unit(P.col(j));
      tmp0.scale(1. / D[j]); 
      //std::cout << "D[j] is " << D[j] << std::endl;
      
      Tj = Tj / D[j];
      r /= D[j];
      power_r /= D[j] * D[j];
      double reduce_r = 0.0;
      MPI_Allreduce(&power_r, &reduce_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      // everything is OK in loop2, reduce r is!!!

      for(size_t bc = t_s_indx; bc < t_s_indx + Atx.dim0(); ++bc)
        tmp[bc] = tmp0[bc - t_s_indx];
      tmp.plus_assign(- gemv(Qj, Tj), Q.dim0() > 1000);
      for(size_t bc = t_s_indx; bc < t_s_indx + Atx.dim0(); ++bc)
        tmp0[bc - t_s_indx] = tmp[bc]; 
      
      double local_r, local_square_sum;
      double reduce_square_sum;
      local_square_sum = Tj.square_sum();
      //std::cout << "reduce_r is " << reduce_r << std::endl;
      //std::cout << "rank " << rank << " local_square_sum is " << local_square_sum << std::endl;
      reduce_square_sum = 0.0;
      double beta;
      beta = reduce_r - local_square_sum;
      //std::cout << "first beta is " << beta << std::endl;
      
      // step 6
      /*
      {
      if(beta < eta * reduce_r) {
        for(size_t bc = t_s_indx; bc < t_s_indx + Atx.dim0(); ++bc)
          tmp[bc] = tmp0[bc - t_s_indx];
        parallel_gemv_task(Qj.trans(), tmp, Tj);
        for(size_t bc = t_s_indx; bc < t_s_indx + Atx.dim0(); ++bc)
          tmp0[bc - t_s_indx] = tmp[bc];
        r = tmp0.square_sum();
        for(size_t bc = t_s_indx; bc < t_s_indx + Atx.dim0(); ++bc)
          tmp[bc] = tmp0[bc - t_s_indx];
        tmp.plus_assign(-gemv(Qj, Tj), Q.dim0() > 1000);
        for(size_t bc = t_s_indx; bc < t_s_indx + Atx.dim0(); ++bc)
          tmp0[bc - t_s_indx] = tmp[bc];
        local_r = r * r;
        local_square_sum = Tj.square_sum();
        MPI_Allreduce(&local_r, &reduce_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        beta = reduce_r - local_square_sum;
        std::cout << "in step 6 reduce_r is " << reduce_r << std::endl;
        std::cout << "in step 6 local_square_sum is " << local_square_sum << std::endl;
      } // end of if
      }
      */

      //std::cout << "last beta is " << beta << std::endl;
      beta = std::sqrt(beta);
      //std::cout << "last beta is " << beta << std::endl;
      E[j] = beta; 
      tmp0.scale(1. / E[j]); // end of step 6	
      //std::cout << "2check beta is " << beta << "and reduce )_r is " << reduce_r << std::endl;
      
      //std::cout << "debug here 1" << std::endl; 
      
      { // local scope, allreduce Q.col(j+1)
      vec_container<double> recv_t((int)Q.dim0() * nprocs, 0);
	
      tmp = 0;
      for(size_t bc = t_s_indx; bc < t_s_indx + Atx.dim0(); ++bc) 
	tmp[bc] = tmp0[bc - t_s_indx];
      
      MPI_Gather(&tmp[0], Q.dim0(), MPI_DOUBLE, &recv_t[0], Q.dim0(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
      if(rank == 0) local_union(Q, recv_t, j + 1, nprocs);
        
      vec_container<double> bcast_data(Q.dim0());
      if(rank == 0)
        for(size_t bc = 0; bc < Q.dim0(); ++bc)
          bcast_data[bc] = Q.col(j + 1)[bc];
      MPI_Bcast(&bcast_data[0], Q.dim0(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      if(rank != 0)
        for(size_t bc = 0; bc < Q.dim0(); ++bc)
          Q.col(j + 1)[bc] = bcast_data[bc];
      } // end of local scope
        
      // step 7
      if(j + 1 < n) {

	// every thing is OK above Q(j+1)!!!!!
	vec_container<double> tmp1(P.dim0(), 0);
        vec_container<double> tmp2(Ax.dim0(), 0);
	m_Ax(Q.col(j + 1), tmp2, P.dim0() > 1000);

	// every thing is OK above tmp2!!!!!
	for(size_t bc = s_indx; bc < s_indx + Ax.dim0(); ++bc) 
	  tmp1[bc] = tmp2[bc - s_indx];
	tmp1.plus_assign(-E[j] * P.col(j), P.dim0() > 1000);
        
	for(size_t bc = 0; bc < tmp1.size(); ++bc)
	  if((int)bc < s_indx || (int)bc >= (s_indx + (int)Ax.dim0()))
	    tmp1[bc] = 0;
        
        // every thing is OK above tmp1!!!	
        { // local scope, allreduce P.col(j+1)
        vec_container<double> recv_d((int)P.dim0() * nprocs);
	MPI_Gather(&tmp1[0], P.dim0(), MPI_DOUBLE, &recv_d[0], P.dim0(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if(rank == 0) local_union(P, recv_d, j + 1, nprocs);
        
        // every thing is OK above P(j+1)!!!
	vec_container<double> bbcast_data(P.dim0(), 0);
        if(rank == 0)
          for(size_t bc = 0; bc < P.dim0(); ++bc)
            bbcast_data[bc] = P.col(j + 1)[bc];
        MPI_Bcast(&bbcast_data[0], P.dim0(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if(rank != 0)
          for(size_t bc = 0; bc < P.dim0(); ++bc)
            P.col(j + 1)[bc] = bbcast_data[bc];
      
        } // end of local scope
         
      } // end of if

    } // end of main loop 
}

template <class CA, class CD, class CE, class CRho, class CP, class CQ>
void bidiag_gkl_restart(int locked, int l, int n,
                        CA && A, CD && D, CE && E, CRho && rho, CP && P, CQ && Q) {
  bidiag_gkl_restart(
      locked, l, n,
      make_gemv_ax(&A), make_gemv_atx(&A), D, E, rho, P, Q);
}

} // namespace linalg
} // namespace douban
#endif
