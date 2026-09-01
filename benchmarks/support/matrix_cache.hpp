#pragma once

// Reading the SpMV matrix suite out of the cache scripts/fetch_matrices.py
// fills, and building the synthetic generator v1 shipped.
//
// A benchmark row names a matrix, not a pair of dimensions. The name has to
// match a row of experiments/matrix_manifest.csv, which carries the SHA-256 of
// the archive and of the converted file, so a row in a committed sweep points at
// a matrix whose bytes can be checked years later. Nothing here downloads
// anything: a missing matrix is an error that tells the reader to run
// fetch_matrices.py, because a benchmark that quietly pulled 250 MB off the
// network mid sweep would put the download inside the measurement.
//
// The binary format is the one fetch_matrices.py writes, little endian:
//
//     char[8]      "CKLCSR01"
//     int32        m, n, nnz
//     int32[m+1]   row_ptr
//     int32[nnz]   col_idx
//     float32[nnz] values

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace ckl {
namespace bench {

/// One CSR matrix on the host, plus the name a benchmark row will carry.
struct HostCsr {
    std::string name;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<float> values;
    int m = 0;
    int n = 0;
    int nnz = 0;
};

/// The cache directory: $CKL_MATRIX_CACHE, or ~/.cache/ckl/matrices.
inline std::string matrix_cache_dir() {
    const char* env = std::getenv("CKL_MATRIX_CACHE");
    if (env != nullptr && env[0] != '\0') {
        return std::string(env);
    }
    const char* home = std::getenv("HOME");
    if (home != nullptr && home[0] != '\0') {
        return std::string(home) + "/.cache/ckl/matrices";
    }
    return ".cache/ckl/matrices";
}

/// v1's generator: most rows a handful of nonzeros, about two percent of them
/// hundreds. Kept as the eighth case of the suite, now driven by a seed rather
/// than by one hard coded 2024.
inline HostCsr make_synthetic(int m, int n, std::uint64_t seed) {
    HostCsr a;
    a.name = "synthetic_skewed";
    a.m = m;
    a.n = n;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_real_distribution<float> val(-1.0f, 1.0f);
    a.row_ptr.assign(static_cast<std::size_t>(m) + 1, 0);
    std::vector<char> taken(static_cast<std::size_t>(n), 0);
    std::vector<int> cols;
    for (int i = 0; i < m; ++i) {
        int degree = 8 + static_cast<int>(unit(rng) * 16.0);
        if (unit(rng) < 0.02) {
            degree = 400 + static_cast<int>(unit(rng) * 600.0);
        }
        degree = degree < n ? degree : n;
        cols.clear();
        while (static_cast<int>(cols.size()) < degree) {
            const int c = static_cast<int>(rng() % static_cast<std::uint64_t>(n));
            if (taken[static_cast<std::size_t>(c)] == 0) {
                taken[static_cast<std::size_t>(c)] = 1;
                cols.push_back(c);
            }
        }
        std::sort(cols.begin(), cols.end());
        for (int c : cols) {
            taken[static_cast<std::size_t>(c)] = 0;
            a.col_idx.push_back(c);
            a.values.push_back(val(rng));
        }
        a.row_ptr[static_cast<std::size_t>(i) + 1] =
            a.row_ptr[static_cast<std::size_t>(i)] + degree;
    }
    a.nnz = a.row_ptr.back();
    return a;
}

/// Reads one cached matrix. Returns false and fills error on any failure.
inline bool read_cached_matrix(const std::string& name, HostCsr* out, std::string* error) {
    const std::string path = matrix_cache_dir() + "/" + name + ".csr.bin";
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        *error = "no cached matrix at " + path +
                 "; run python3 scripts/fetch_matrices.py to fetch and convert the suite";
        return false;
    }
    char magic[8] = {0};
    std::int32_t header[3] = {0, 0, 0};
    bool ok = std::fread(magic, 1, sizeof(magic), f) == sizeof(magic) &&
              std::fread(header, sizeof(std::int32_t), 3, f) == 3;
    if (ok && std::memcmp(magic, "CKLCSR01", 8) != 0) {
        ok = false;
        *error = path + " does not start with the CKLCSR01 magic; it is not a converted CSR";
    }
    if (!ok) {
        if (error->empty()) {
            *error = path + " is truncated at the header";
        }
        std::fclose(f);
        return false;
    }
    out->name = name;
    out->m = header[0];
    out->n = header[1];
    out->nnz = header[2];
    if (out->m < 0 || out->n < 0 || out->nnz < 0) {
        *error = path + " declares a negative dimension";
        std::fclose(f);
        return false;
    }
    out->row_ptr.resize(static_cast<std::size_t>(out->m) + 1);
    out->col_idx.resize(static_cast<std::size_t>(out->nnz));
    out->values.resize(static_cast<std::size_t>(out->nnz));
    const bool body =
        std::fread(out->row_ptr.data(), sizeof(int), out->row_ptr.size(), f) ==
            out->row_ptr.size() &&
        (out->nnz == 0 || (std::fread(out->col_idx.data(), sizeof(int), out->col_idx.size(), f) ==
                               out->col_idx.size() &&
                           std::fread(out->values.data(), sizeof(float), out->values.size(), f) ==
                               out->values.size()));
    std::fclose(f);
    if (!body) {
        *error = path + " is truncated: the header promises " + std::to_string(out->nnz) +
                 " nonzeros and the file does not hold them";
        return false;
    }
    if (out->row_ptr.back() != out->nnz) {
        *error = path + " has a row_ptr whose last entry is " +
                 std::to_string(out->row_ptr.back()) + " against a declared nnz of " +
                 std::to_string(out->nnz);
        return false;
    }
    return true;
}

/**
 * Resolves a matrix specification.
 *
 * A bare name loads that matrix from the cache. "synthetic" builds v1's
 * generator at its default 131072 rows, and "synthetic:N" at N rows, which is
 * what keeps the ordinary sweep rows working without a populated cache.
 */
inline bool load_matrix(const std::string& spec, std::uint64_t seed, HostCsr* out,
                        std::string* error) {
    error->clear();
    if (spec.rfind("synthetic", 0) == 0) {
        int rows = 131072;
        const std::size_t colon = spec.find(':');
        if (colon != std::string::npos) {
            rows = std::atoi(spec.c_str() + colon + 1);
        }
        if (rows <= 0) {
            *error = "synthetic needs a positive row count, got " + spec;
            return false;
        }
        *out = make_synthetic(rows, rows, seed);
        // The seed is part of the matrix's identity: seeds 1 to 5 are five
        // different matrices, and two rows that name the same generator at
        // different seeds must not collapse onto each other in a summary.
        out->name = "synthetic:" + std::to_string(rows) + ":seed" + std::to_string(seed);
        return true;
    }
    // A plain integer is the v1 form: spmv <m> <n> with a generated matrix.
    if (!spec.empty() && spec.find_first_not_of("0123456789") == std::string::npos) {
        const int rows = std::atoi(spec.c_str());
        if (rows <= 0) {
            *error = "a numeric matrix specification is a synthetic row count and must be positive";
            return false;
        }
        *out = make_synthetic(rows, rows, seed);
        out->name = "synthetic:" + spec + ":seed" + std::to_string(seed);
        return true;
    }
    return read_cached_matrix(spec, out, error);
}

}  // namespace bench
}  // namespace ckl
