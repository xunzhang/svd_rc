/* run_svd_tr.cpp */
#include "mpi_svd_tr.hpp"

#include <douban/mpi.hpp>
#include <douban/matvec.hpp>
#include <douban/option_parser.hpp>
#include <douban/linalg/gemv_ax.hpp>
#include <douban/matvec/mat_container.hpp>
#include <douban/matvec/vec_kahan_gemv.hpp>
#include <douban/mpi/graph_load.hpp>
#include <douban/utility/string.hpp>
#include <douban/clog_utility.hpp>

#include <boost/foreach.hpp>

#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <memory.h>
#include <numeric>

int main(int argc, char *argv[]) {
  douban::mpi::main_env mpi_main_env;
  douban::mpi::Comm comm;
  int rank = comm.rank(), nprocs = comm.size();

  size_t m = 0;
  size_t n = 0;
  size_t k = 0;
  size_t p = 0;
  int display = 10;

  std::string in_fmt;
  std::vector<std::string> in_files;
  std::string out_u, out_s, out_v;

  douban::option_parser cmd_parser;
  cmd_parser
      .add_help()
      .add_value_option("k", &k, "NUM::# of eigens")
      .add_value_option("p", &p, "NUM::# of over sample")
      .add_value_option("m|row", &m, "NUM::# row of matrix")
      .add_value_option("n|col", &n, "NUM::# col of matrix")
      .add_vfunc_option("in-file", [&](const std::string f,
                                       const std::vector<std::string> fns) {
          in_fmt = f;
          in_files.insert(in_files.end(), fns.begin(), fns.end());
          return 1;
        }, "FMT FILE+::input format and file")
      .add_value_option("out-u", &out_u, "FILE::out u file with %{rank} replaced")
      .add_value_option("out-s", &out_s, "FILE::out s file with %{rank} replaced")
      .add_value_option("out-v", &out_v, "FILE::out v file with %{rank} replaced")
      ;
  cmd_parser.parse_all(argc, argv);
  if (p == 0) p = k;
  CLOG_CHECK(p > 0 && k > 0, "invalid p or k");
  CLOG_CHECK(m > 0 && n > 0, "invalid m or n");
  CLOG_CHECK(in_files.size(), "no input files");

  std::vector<size_t> Ap, Ai, Atp, Ati;
  std::vector<double> Av, Atv;
  size_t row_step = (m + nprocs - 1) / nprocs;
  size_t col_step = (n + nprocs - 1) / nprocs;
  size_t lm = std::min(row_step * (rank + 1), m) - row_step * rank;
  size_t ln = std::min(col_step * (rank + 1), n) - col_step * rank;
  {
    std::vector< std::vector<size_t> > ai_loads(nprocs);
    std::vector< std::vector<size_t> > aj_loads(nprocs);
    std::vector< std::vector<double> > av_loads(nprocs);
    std::vector< std::vector<size_t> > ati_loads(nprocs);
    std::vector< std::vector<size_t> > atj_loads(nprocs);
    std::vector< std::vector<double> > atv_loads(nprocs);
    CLOG_S(INFO, "loading ...");

    size_t c = douban::graph_files_run([&](size_t i, size_t j, double v) {
        int ir = i / row_step;
        int jr = j /col_step;
        ai_loads[ir].push_back(i);
        aj_loads[ir].push_back(j);
        av_loads[ir].push_back(v);
        ati_loads[jr].push_back(j);
        atj_loads[jr].push_back(i);
        atv_loads[jr].push_back(v);
      }, in_files, "fsv", comm, 8);

    CLOG_T(INFO, "loaded #", c);
    {
      std::vector<size_t> ai, aj;
      std::vector<double> av;
      CLOG_S(INFO, "all to all of a");
      douban::mpi::all_to_all(
          &ai_loads[0],
          [&](std::vector<size_t> &r) {
            ai.insert(ai.end(), r.begin(), r.end());
          }, comm);
      CLOG_T(INFO, "done ai");
      douban::mpi::all_to_all(
          &aj_loads[0],
          [&](std::vector<size_t> &r) {
            aj.insert(aj.end(), r.begin(), r.end());
          }, comm);
      CLOG_T(INFO, "done aj");
      douban::mpi::all_to_all(
          &av_loads[0],
          [&](std::vector<double> &r) {
            av.insert(av.end(), r.begin(), r.end());
          }, comm);
      CLOG_T(INFO, "done av");
      ai_loads.clear();
      aj_loads.clear();
      av_loads.clear();
      BOOST_FOREACH (auto &e, ai) {
        e -= row_step * rank;
      }
      Av.resize(ai.size());
      Ai.resize(ai.size());
      Ap.resize(lm + 1);
      CLOG_S(INFO, "A ...");
      std::cout << "lmlmlmlm" << lm << std::endl;
      std::cout << "nnnnnnnn" << n << std::endl;
      std::cout << "aiaiai.size" << ai.size() << std::endl;
      douban::coo_to_csr(lm, n, ai.size(), av, ai, aj, Av, Ap, Ai);
      CLOG_T(INFO, "done");
    }
    {
      std::vector<size_t> ati, atj;
      std::vector<double> atv;
      CLOG_S(INFO, "all to all At ...");
      douban::mpi::all_to_all(
          &ati_loads[0],
          [&](std::vector<size_t> &r) {
            ati.insert(ati.end(), r.begin(), r.end());
          }, comm);
      douban::mpi::all_to_all(
          &atj_loads[0],
          [&](std::vector<size_t> &r) {
            atj.insert(atj.end(), r.begin(), r.end());
          }, comm);
      douban::mpi::all_to_all(
          &atv_loads[0],
          [&](std::vector<double> &r) {
            atv.insert(atv.end(), r.begin(), r.end());
          }, comm);
      CLOG_T(INFO, "done all to all At");
      BOOST_FOREACH (auto &e, ati) {
        e -= col_step * rank;
      }
      Atv.resize(ati.size());
      Ati.resize(ati.size());
      Atp.resize(ln + 1);
      CLOG_S(INFO, "At ...");
      std::cout << "lnlnlnln" << ln << std::endl;
      std::cout << "mmmmmmmm" << m << std::endl;
      std::cout << "atiati.size" << ati.size() << std::endl;
      douban::coo_to_csr(ln, m, ati.size(), atv, ati, atj, Atv, Atp, Ati);
      CLOG_T(INFO, "done");
    }
  }

  // init A and A'
  auto A = douban::make_mat_csr(Ap.size() - 1, n, &Av, &Ap, &Ai);
  auto At = douban::make_mat_csr(Atp.size() - 1, m, &Atv, &Atp, &Ati);
  VCLOG(1, "Av.sum=", std::accumulate(Av.begin(), Av.end(), 0.));
  VCLOG(1, "Atv.sum=", std::accumulate(Atv.begin(), Atv.end(), 0.));

  douban::mat_container<double> U(m, k);
  douban::mat_container<double> V(n, k);
  douban::vec_container<double> S(k);

  CLOG_S(INFO, "svd_tr ...");
  douban::linalg::svd_tr(A, At, U, S, V, p, 1.e-7, display, rank * row_step, rank * col_step);
  CLOG_T(INFO, "done svd_tr");

  for(size_t kk = 0; kk < k; ++kk)
    std::cout << "S(i) is " << S.get(kk) << std::endl;

  if (comm.rank() == 0 && out_u.size()) {
    out_u = douban::str_replace(out_u, "%{rank}", rank);
    std::ofstream ofs(out_u);
    size_t row_start = rank * row_step;
    for (size_t i = 0; i < U.dim0(); ++i) {
      douban::text_voutput(*ofs.rdbuf(), i + row_start, '\t',
                           U.row(i), '\n');
    }
  }

  if (comm.rank() == 0 && out_v.size()) {
    out_v = douban::str_replace(out_v, "%{rank}", rank);
    std::ofstream ofs(out_v);
    size_t col_start = rank * col_step;
    for (size_t i = 0; i < V.dim0(); ++i) {
      douban::text_voutput(*ofs.rdbuf(), i + col_start, '\t',
                           V.row(i), '\n');
    }
  }

  if (comm.rank() == 0 && out_s.size()) {
    std::ofstream ofs(out_s);
    douban::text_output(*ofs.rdbuf(), S);
    ofs.rdbuf()->sputc('\n');
  }

  return 0;
}
