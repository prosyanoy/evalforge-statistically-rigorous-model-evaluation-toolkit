// EvalForge native kernels: multithreaded bootstrap and paired-permutation
// resampling for accuracy-like means, macro-F1, ROC AUC and ECE.
//
// Determinism contract: replicate `b` draws its randomness only from
// (seed, b), so results are identical for any thread count. Bootstrap index
// streams depend only on (seed, b, n), which lets callers evaluate two models
// (or several classes) on exactly the same resamples by reusing the seed.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

inline uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

struct Rng {
    uint64_t s;
    Rng(uint64_t seed, uint64_t stream)
        : s(mix64(seed * 0x9E3779B97F4A7C15ULL + (stream + 1) * 0xD1B54A32D192ED03ULL)) {}
    inline uint64_t next() {
        s += 0x9E3779B97F4A7C15ULL;
        return mix64(s);
    }
    // Unbiased-enough fast range reduction (Lemire) into [0, n).
    inline uint32_t below(uint32_t n) {
        return static_cast<uint32_t>((static_cast<__uint128_t>(next()) * n) >> 64);
    }
};

template <class F>
void parallel_blocks(int64_t total, int n_threads, F fn) {
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (n_threads <= 0) n_threads = hw > 0 ? hw : 1;
    n_threads = static_cast<int>(std::min<int64_t>(n_threads, std::max<int64_t>(total, 1)));
    if (n_threads <= 1) {
        fn(int64_t{0}, total);
        return;
    }
    std::vector<std::thread> pool;
    int64_t chunk = (total + n_threads - 1) / n_threads;
    for (int t = 0; t < n_threads; ++t) {
        int64_t lo = t * chunk, hi = std::min<int64_t>(total, lo + chunk);
        if (lo >= hi) break;
        pool.emplace_back([=] { fn(lo, hi); });
    }
    for (auto &th : pool) th.join();
}

using DArr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IArr = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

void check_1d(const py::buffer_info &b, const char *name) {
    if (b.ndim != 1) throw std::invalid_argument(std::string(name) + " must be 1-D");
}

// Macro-F1 over K classes from per-class counts; classes with no true and no
// predicted samples score 0 (matches sklearn zero_division=0).
inline double macro_f1_counts(const int64_t *tp, const int64_t *pc, const int64_t *tc, int K) {
    double acc = 0.0;
    for (int k = 0; k < K; ++k) {
        int64_t d = pc[k] + tc[k];
        if (d > 0) acc += 2.0 * static_cast<double>(tp[k]) / static_cast<double>(d);
    }
    return acc / K;
}

void check_labels(const int32_t *y, int64_t n, int K, const char *name) {
    for (int64_t i = 0; i < n; ++i)
        if (y[i] < 0 || y[i] >= K) throw std::invalid_argument(std::string(name) + " has labels outside [0, n_classes)");
}

// ---------------------------------------------------------------- bootstrap

py::array_t<double> boot_mean(DArr values, int64_t n_boot, uint64_t seed, int n_threads) {
    auto vb = values.request();
    check_1d(vb, "values");
    const int64_t n = vb.shape[0];
    if (n == 0) throw std::invalid_argument("values is empty");
    const double *v = static_cast<const double *>(vb.ptr);
    py::array_t<double> out(n_boot);
    double *o = out.mutable_data();
    {
        py::gil_scoped_release nogil;
        parallel_blocks(n_boot, n_threads, [&](int64_t lo, int64_t hi) {
            for (int64_t b = lo; b < hi; ++b) {
                Rng r(seed, b);
                double s = 0;
                for (int64_t i = 0; i < n; ++i) s += v[r.below(static_cast<uint32_t>(n))];
                o[b] = s / static_cast<double>(n);
            }
        });
    }
    return out;
}

py::array_t<double> boot_macro_f1(IArr y, IArr pred, int K, int64_t n_boot, uint64_t seed, int n_threads) {
    auto yb = y.request(), pb = pred.request();
    check_1d(yb, "y"); check_1d(pb, "pred");
    const int64_t n = yb.shape[0];
    if (pb.shape[0] != n || n == 0) throw std::invalid_argument("y/pred length mismatch or empty");
    const int32_t *yy = static_cast<const int32_t *>(yb.ptr);
    const int32_t *pp = static_cast<const int32_t *>(pb.ptr);
    check_labels(yy, n, K, "y"); check_labels(pp, n, K, "pred");
    py::array_t<double> out(n_boot);
    double *o = out.mutable_data();
    {
        py::gil_scoped_release nogil;
        parallel_blocks(n_boot, n_threads, [&](int64_t lo, int64_t hi) {
            std::vector<int64_t> tp(K), pc(K), tc(K);
            for (int64_t b = lo; b < hi; ++b) {
                Rng r(seed, b);
                std::fill(tp.begin(), tp.end(), 0);
                std::fill(pc.begin(), pc.end(), 0);
                std::fill(tc.begin(), tc.end(), 0);
                for (int64_t i = 0; i < n; ++i) {
                    uint32_t j = r.below(static_cast<uint32_t>(n));
                    ++tc[yy[j]]; ++pc[pp[j]];
                    if (yy[j] == pp[j]) ++tp[yy[j]];
                }
                o[b] = macro_f1_counts(tp.data(), pc.data(), tc.data(), K);
            }
        });
    }
    return out;
}

py::array_t<double> boot_auc(IArr y, DArr score, int64_t n_boot, uint64_t seed, int n_threads) {
    auto yb = y.request(), sb = score.request();
    check_1d(yb, "y"); check_1d(sb, "score");
    const int64_t n = yb.shape[0];
    if (sb.shape[0] != n || n == 0) throw std::invalid_argument("y/score length mismatch or empty");
    const int32_t *yy = static_cast<const int32_t *>(yb.ptr);
    const double *ss = static_cast<const double *>(sb.ptr);
    check_labels(yy, n, 2, "y");
    // One-off sort; each replicate then only needs O(n) work on resample counts.
    std::vector<int32_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return ss[a] < ss[b]; });
    std::vector<int32_t> pos(n), lab(n);
    std::vector<char> last(n, 0);  // last element of a tie group (in sorted order)
    for (int64_t k = 0; k < n; ++k) {
        pos[order[k]] = static_cast<int32_t>(k);
        lab[k] = yy[order[k]];
        last[k] = (k + 1 == n || ss[order[k + 1]] != ss[order[k]]) ? 1 : 0;
    }
    py::array_t<double> out(n_boot);
    double *o = out.mutable_data();
    {
        py::gil_scoped_release nogil;
        parallel_blocks(n_boot, n_threads, [&](int64_t lo, int64_t hi) {
            std::vector<int32_t> w(n);
            for (int64_t b = lo; b < hi; ++b) {
                Rng r(seed, b);
                std::fill(w.begin(), w.end(), 0);
                for (int64_t i = 0; i < n; ++i) ++w[pos[r.below(static_cast<uint32_t>(n))]];
                double cum_neg = 0, num = 0, gp = 0, gn = 0, tot_p = 0;
                for (int64_t k = 0; k < n; ++k) {
                    if (lab[k] == 1) gp += w[k]; else gn += w[k];
                    if (last[k]) {
                        num += gp * (cum_neg + 0.5 * gn);
                        cum_neg += gn; tot_p += gp; gp = 0; gn = 0;
                    }
                }
                o[b] = (tot_p == 0 || cum_neg == 0) ? kNaN : num / (tot_p * cum_neg);
            }
        });
    }
    return out;
}

// ECE = sum_bins |sum_{i in bin} (correct_i - conf_i)| / n, resampled.
py::array_t<double> boot_ece(IArr bins, DArr gap, int n_bins, int64_t n_boot, uint64_t seed, int n_threads) {
    auto bb = bins.request(), gb = gap.request();
    check_1d(bb, "bins"); check_1d(gb, "gap");
    const int64_t n = bb.shape[0];
    if (gb.shape[0] != n || n == 0) throw std::invalid_argument("bins/gap length mismatch or empty");
    const int32_t *bi = static_cast<const int32_t *>(bb.ptr);
    const double *g = static_cast<const double *>(gb.ptr);
    check_labels(bi, n, n_bins, "bins");
    py::array_t<double> out(n_boot);
    double *o = out.mutable_data();
    {
        py::gil_scoped_release nogil;
        parallel_blocks(n_boot, n_threads, [&](int64_t lo, int64_t hi) {
            std::vector<double> acc(n_bins);
            for (int64_t b = lo; b < hi; ++b) {
                Rng r(seed, b);
                std::fill(acc.begin(), acc.end(), 0.0);
                for (int64_t i = 0; i < n; ++i) {
                    uint32_t j = r.below(static_cast<uint32_t>(n));
                    acc[bi[j]] += g[j];
                }
                double e = 0;
                for (int k = 0; k < n_bins; ++k) e += std::fabs(acc[k]);
                o[b] = e / static_cast<double>(n);
            }
        });
    }
    return out;
}

// -------------------------------------------------------------- permutation
// Paired permutation: for every sample, model A and B outputs are swapped with
// probability 1/2 (exchangeable under H0). Returns the null distribution of
// metric(A') - metric(B').

inline void fill_swap(Rng &r, int64_t n, std::vector<char> &swap) {
    uint64_t word = 0;
    for (int64_t i = 0; i < n; ++i) {
        if ((i & 63) == 0) word = r.next();
        swap[i] = static_cast<char>((word >> (i & 63)) & 1ULL);
    }
}

py::array_t<double> perm_mean(DArr a, DArr b, int64_t n_perm, uint64_t seed, int n_threads) {
    auto ab = a.request(), bb = b.request();
    check_1d(ab, "a"); check_1d(bb, "b");
    const int64_t n = ab.shape[0];
    if (bb.shape[0] != n || n == 0) throw std::invalid_argument("a/b length mismatch or empty");
    const double *pa = static_cast<const double *>(ab.ptr);
    const double *pb = static_cast<const double *>(bb.ptr);
    py::array_t<double> out(n_perm);
    double *o = out.mutable_data();
    {
        py::gil_scoped_release nogil;
        parallel_blocks(n_perm, n_threads, [&](int64_t lo, int64_t hi) {
            std::vector<char> sw(n);
            for (int64_t p = lo; p < hi; ++p) {
                Rng r(seed, p);
                fill_swap(r, n, sw);
                double d = 0;
                for (int64_t i = 0; i < n; ++i) d += sw[i] ? (pb[i] - pa[i]) : (pa[i] - pb[i]);
                o[p] = d / static_cast<double>(n);
            }
        });
    }
    return out;
}

py::array_t<double> perm_macro_f1(IArr y, IArr pa, IArr pb, int K, int64_t n_perm, uint64_t seed, int n_threads) {
    auto yb = y.request(), ab = pa.request(), bb = pb.request();
    check_1d(yb, "y"); check_1d(ab, "pred_a"); check_1d(bb, "pred_b");
    const int64_t n = yb.shape[0];
    if (ab.shape[0] != n || bb.shape[0] != n || n == 0) throw std::invalid_argument("length mismatch or empty");
    const int32_t *yy = static_cast<const int32_t *>(yb.ptr);
    const int32_t *A = static_cast<const int32_t *>(ab.ptr);
    const int32_t *B = static_cast<const int32_t *>(bb.ptr);
    check_labels(yy, n, K, "y"); check_labels(A, n, K, "pred_a"); check_labels(B, n, K, "pred_b");
    py::array_t<double> out(n_perm);
    double *o = out.mutable_data();
    {
        py::gil_scoped_release nogil;
        parallel_blocks(n_perm, n_threads, [&](int64_t lo, int64_t hi) {
            std::vector<char> sw(n);
            std::vector<int64_t> tpa(K), pca(K), tca(K), tpb(K), pcb(K), tcb(K);
            for (int64_t p = lo; p < hi; ++p) {
                Rng r(seed, p);
                fill_swap(r, n, sw);
                std::fill(tpa.begin(), tpa.end(), 0); std::fill(pca.begin(), pca.end(), 0); std::fill(tca.begin(), tca.end(), 0);
                std::fill(tpb.begin(), tpb.end(), 0); std::fill(pcb.begin(), pcb.end(), 0); std::fill(tcb.begin(), tcb.end(), 0);
                for (int64_t i = 0; i < n; ++i) {
                    int32_t x = sw[i] ? B[i] : A[i], z = sw[i] ? A[i] : B[i];
                    ++tca[yy[i]]; ++pca[x]; if (x == yy[i]) ++tpa[x];
                    ++tcb[yy[i]]; ++pcb[z]; if (z == yy[i]) ++tpb[z];
                }
                o[p] = macro_f1_counts(tpa.data(), pca.data(), tca.data(), K) -
                       macro_f1_counts(tpb.data(), pcb.data(), tcb.data(), K);
            }
        });
    }
    return out;
}

// AUC of the score vector x where x[i] = A[i] if sw[i] == s else B[i], computed in O(n) by
// merging the pre-sorted orders of A and B (no per-permutation sort).
double auc_merge(const double *A, const double *B, const int32_t *oa, const int32_t *ob, const char *sw,
                 char s, const int32_t *y, int64_t n) {
    int64_t ia = 0, ib = 0;
    double cum_neg = 0, num = 0, gp = 0, gn = 0, tot_p = 0, cur = 0;
    bool have = false;
    auto flush = [&] {
        num += gp * (cum_neg + 0.5 * gn);
        cum_neg += gn; tot_p += gp; gp = 0; gn = 0;
    };
    while (true) {
        while (ia < n && sw[oa[ia]] != s) ++ia;
        while (ib < n && sw[ob[ib]] == s) ++ib;
        if (ia >= n && ib >= n) break;
        double v;
        int32_t lab;
        if (ib >= n || (ia < n && A[oa[ia]] <= B[ob[ib]])) { v = A[oa[ia]]; lab = y[oa[ia]]; ++ia; }
        else { v = B[ob[ib]]; lab = y[ob[ib]]; ++ib; }
        if (have && v != cur) flush();
        cur = v; have = true;
        if (lab == 1) gp += 1; else gn += 1;
    }
    if (have) flush();
    return (tot_p == 0 || cum_neg == 0) ? kNaN : num / (tot_p * cum_neg);
}

py::array_t<double> perm_auc(IArr y, DArr sa, DArr sb, int64_t n_perm, uint64_t seed, int n_threads) {
    auto yb = y.request(), ab = sa.request(), bb = sb.request();
    check_1d(yb, "y"); check_1d(ab, "score_a"); check_1d(bb, "score_b");
    const int64_t n = yb.shape[0];
    if (ab.shape[0] != n || bb.shape[0] != n || n == 0) throw std::invalid_argument("length mismatch or empty");
    const int32_t *yy = static_cast<const int32_t *>(yb.ptr);
    const double *A = static_cast<const double *>(ab.ptr);
    const double *B = static_cast<const double *>(bb.ptr);
    check_labels(yy, n, 2, "y");
    std::vector<int32_t> oa(n), ob(n);
    std::iota(oa.begin(), oa.end(), 0);
    std::iota(ob.begin(), ob.end(), 0);
    std::sort(oa.begin(), oa.end(), [&](int32_t i, int32_t j) { return A[i] < A[j]; });
    std::sort(ob.begin(), ob.end(), [&](int32_t i, int32_t j) { return B[i] < B[j]; });
    py::array_t<double> out(n_perm);
    double *o = out.mutable_data();
    {
        py::gil_scoped_release nogil;
        parallel_blocks(n_perm, n_threads, [&](int64_t lo, int64_t hi) {
            std::vector<char> sw(n);
            for (int64_t p = lo; p < hi; ++p) {
                Rng r(seed, p);
                fill_swap(r, n, sw);
                // A' takes A where not swapped; B' takes A where swapped.
                o[p] = auc_merge(A, B, oa.data(), ob.data(), sw.data(), 0, yy, n) -
                       auc_merge(A, B, oa.data(), ob.data(), sw.data(), 1, yy, n);
            }
        });
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "EvalForge native resampling kernels (multithreaded, thread-count independent)";
    m.def("boot_mean", &boot_mean, py::arg("values"), py::arg("n_boot"), py::arg("seed"), py::arg("n_threads") = 0);
    m.def("boot_macro_f1", &boot_macro_f1, py::arg("y"), py::arg("pred"), py::arg("n_classes"), py::arg("n_boot"),
          py::arg("seed"), py::arg("n_threads") = 0);
    m.def("boot_auc", &boot_auc, py::arg("y"), py::arg("score"), py::arg("n_boot"), py::arg("seed"),
          py::arg("n_threads") = 0);
    m.def("boot_ece", &boot_ece, py::arg("bins"), py::arg("gap"), py::arg("n_bins"), py::arg("n_boot"),
          py::arg("seed"), py::arg("n_threads") = 0);
    m.def("perm_mean", &perm_mean, py::arg("a"), py::arg("b"), py::arg("n_perm"), py::arg("seed"),
          py::arg("n_threads") = 0);
    m.def("perm_macro_f1", &perm_macro_f1, py::arg("y"), py::arg("pred_a"), py::arg("pred_b"), py::arg("n_classes"),
          py::arg("n_perm"), py::arg("seed"), py::arg("n_threads") = 0);
    m.def("perm_auc", &perm_auc, py::arg("y"), py::arg("score_a"), py::arg("score_b"), py::arg("n_perm"),
          py::arg("seed"), py::arg("n_threads") = 0);
}
