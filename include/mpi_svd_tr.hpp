/**
 * @file   svd_tr.hpp
 * @author Changsheng Jiang <jiangzuoyan@gmail.com>
 * @date   Mon Feb 27 12:00:18 2012
 *
 * @brief mpi version of thick restart svd method
 *
 *
 */
#ifndef FILE_5381048f_3032_49de_9166_64d32706b8ea_H
#define FILE_5381048f_3032_49de_9166_64d32706b8ea_H
#include "douban/linalg/svd.hpp"
#include "mpi_bidiag.hpp"
#include "douban/linalg/gemv_ax.hpp"
#include "douban/matvec/io.hpp"
#include "douban/gemm.hpp"
#include "douban/clog.hpp"

#include <iostream>
#include <sys/time.h>

static double current_time(void){
  double timestamp;
  struct timeval tv;
  gettimeofday(&tv, 0);		
  timestamp = (double)((double)(tv.tv_sec*1e6) +(double)tv.tv_usec);
  return timestamp;
}

namespace douban {
namespace linalg {

template <class CAX, class CATX, class CU, class CS, class CV>
  typename std::enable_if<!std::is_integral<typename std::remove_reference<CV>::type>::value, int>::type
  svd_tr(CAX && Ax, CATX && Atx, 
	 CU && U, CS && S, CV && V, 
         int over_sample=0, double eps=1.e-7, int display=0, int s_indx=0, int t_s_indx=0) {
                   
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    double t_start, t_end, t_start_tmp, t_end_tmp;
    int m = U.dim0(), n = V.dim0();
    int nev = std::max(U.dim1(), std::max(V.dim1(), S.size()));
    if (nev <= 0) return 0;
    if (nev > std::min(m, n)) return -1;
    if (over_sample == 0) over_sample = nev;
    int k = nev + over_sample;
    k = std::min(k, std::min(m, n));
    std::vector<double> work(std::max(m, n) * k + 3 * k * k + 4 * k, 0);
    double *mat_buf = &work[0];
    double *b_buf = mat_buf + std::max(m, n) * k;
    double *bu_buf = b_buf + k * k;
    double *bv_buf = bu_buf + k * k;
    double *alpha_buf = bv_buf + k * k;
    double *beta_buf = alpha_buf + k;
    double *rho_buf = beta_buf + k;
    double *s_buf = rho_buf + k;
    //CLOG_IF((sint)work.size() != s_buf + k - &work[0], FATAL, "should never touch here ...");
    auto alpha = make_vec(alpha_buf, k);
    auto beta = make_vec(beta_buf, k);
    auto rho = make_vec(rho_buf, k);
    auto B = make_mat_col_major(b_buf, k, k);
				
    mat_container<double, true> P(m, k);
    mat_container<double, true> Q(n, k + 1);
    P = 0;

    if(rank == 0) {
      Q = 0;
      Q.col(0) = vec_random(Q.col(0).size()) - .5;
      vec_unit(Q.col(0));
    }
				
    MPI_Bcast(&(Q.col(0)[0]), n * (k + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int nconv = 0, l = 0, locked = 0;
    int thick_cnt = 0, cnt = 0, mv_cost = 0;
    double ot = 0;
    t_start = current_time();
				while (nconv < nev) {
					thick_cnt += 1;
					cnt += 1;
					CLOG_IF(display, INFO, "thick " << cnt << "/" << thick_cnt << " locked=" << locked << " l=" << l);
					vec_lim_str(rho, 30);
					std::cout << vec_lim_str(rho, 5) << std::endl;
					CLOG_IF(display >= 10, INFO, "rho=" << vec_lim_str(rho, 30));
					if (display >= 5) ot = get_drtime();
					std::cout << "bidiag_gkl_restart start" << std::endl; 
					t_start_tmp = current_time();
					std::cout << "l is" << l << "and k is " << k << std::endl;
					bidiag_gkl_restart(locked, l, k, Ax, Atx, alpha, beta, rho, P, Q, s_indx, t_s_indx);
					t_end_tmp = current_time();
					std::cout << "bidiag_gkl_restart end" << std::endl; 
					std::cout << "bidiag_glk_restart time is " << (t_end_tmp - t_start_tmp) / 1.0e6 << std::endl;
					CLOG_IF(display >= 5, INFO, "bidiag gkl restart using " << (get_drtime() - ot) << " seconds");
					vec_lim_str(alpha, 30);
					vec_lim_str(beta, 30);
					CLOG_IF(display >= 10, INFO, "alpha=" << vec_lim_str(alpha, 30));
					CLOG_IF(display >= 10, INFO, "beta=" << vec_lim_str(beta, 30));
					mv_cost += k - l;
					B = 0;
					for (int i = locked; i < l; ++i) {
						B(i, i) = s_buf[i];
						B(i, l) = rho[i];
					}
					for (int i = l; i < k; ++i) {
						B(i, i) = alpha[i];
					}
					for (int i = l; i + 1 < k; ++i) {
						B(i, i + 1) = beta[i];
					}
					auto BU = make_mat_col_major(bu_buf, k - locked, k - locked);
					auto BV = make_mat_col_major(bv_buf, k - locked, k - locked);
					auto BS = make_vec(s_buf + locked, k - locked);
					CLOG_IF(display >= 10, INFO, "svd_ge ...");
					std::cout << "svd_ge begin" << std::endl; 
					t_start_tmp = current_time();
					svd_ge(make_mat_sub(&B, k - locked, k - locked, locked, locked),
							BU, BS, BV);
					t_end_tmp = current_time();
					std::cout << "svd_ge end" << std::endl; 
					std::cout << "svd_ge time is " << (t_end_tmp - t_start_tmp) / 1.0e6 << std::endl;
					vec_lim_str(BS, 100);
					CLOG_IF(display >= 6, INFO, "S=" << vec_lim_str(BS, 100));
					CLOG_IF(display >= 10, INFO, "done svd_ge");
					make_vec(rho_buf + locked, k - locked) = beta[k - 1] * BU.row(k - locked - 1);
					int o_nconv = nconv, n_nconv = locked;
					while (n_nconv < k) {
						if (std::abs(rho[n_nconv]) > eps * s_buf[0]) break;
						n_nconv ++;
					}
					if (n_nconv != o_nconv) {
						cnt = 0;
						l = (n_nconv + k) / 2;
						if (l == n_nconv) l = k;
					}
					if (display >= 5) ot = get_drtime();
					if(rank == 0) {
#pragma omp parallel sections
						{
#pragma omp section
							{
								if (l > locked) {
									auto Qk = make_mat_sub(&Q, Q.dim0(), k - locked, 0, locked);
									auto Ql = make_mat_sub(&Q, Q.dim0(), l - locked, 0, locked);
									blas::gemm(1, Qk, make_mat_sub(&BV, BV.dim0(), l - locked, 0, 0), 0, Ql);
								}
							}
#pragma omp section
							{
								if (l > locked) {
									auto Pk = make_mat_sub(&P, P.dim0(), k - locked, 0, locked);
									auto Pl = make_mat_sub(&P, P.dim0(), l - locked, 0, locked);
									blas::gemm(1, Pk, make_mat_sub(&BU, BU.dim0(), l - locked, 0, 0), 0, Pl);
								}
							}
						}
					}
					MPI_Bcast(&l, 1, MPI_INT, 0, MPI_COMM_WORLD);
					MPI_Bcast(&locked, 1, MPI_INT, 0, MPI_COMM_WORLD);
					MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
					Q.col(l).assign(Q.col(k));
					CLOG_IF(display >= 5, INFO, "updated P and Q in " << (get_drtime() - ot) << " seconds");
					nconv = n_nconv;
					//CLOG_IF(display && nconv != o_nconv, INFO, "**** nconv=" << nconv);
					MPI_Bcast(&nconv, 1, MPI_INT, 0, MPI_COMM_WORLD);
					MPI_Bcast(&nev, 1, MPI_INT, 0, MPI_COMM_WORLD);
					if (nconv >= nev) break;  
					if (cnt >= 10000) {
						CLOG_IF(display, ERROR, "thick restart with cnt over limit ...");
						throw std::runtime_error("thick restart with cnt over limit ...");
					}
					locked = nconv;
				}
				t_end = current_time();
				std::cout << "main while loop time is " << (t_end - t_start) / 1.0e6 << std::endl;
				S = make_vec(s_buf, S.size());
				U = make_mat_sub(&P, P.dim0(), U.dim1(), 0, 0);
				V = make_mat_sub(&Q, Q.dim0(), V.dim1(), 0, 0);
				return 0;
			}
} // namespace linalg
} // namespace douban

#endif
