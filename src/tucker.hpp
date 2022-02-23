/*
 * Copyright (c) 2016-2017, Rafael Ballester-Ripoll
 *                          (Visualization and MultiMedia Lab, University of Zurich),
 *                          rballester@ifi.uzh.ch
 *
 * Licensed under the LGPLv3.0 (https://github.com/rballester/tthresh/blob/master/LICENSE)
 */

#ifndef __TUCKER_HPP__
#define __TUCKER_HPP__

#include "Slice.hpp"
#include <Eigen/Dense>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;
using namespace Eigen;

// Projects an unfolded core M into M_proj using the transformation matrix U.
// U is an output parameter and is computed as the HOSVD of the tensor (left singular
// vectors of M) and M is compressed using U.transpose().
void project(MatrixXd& M, MatrixXd& U, MatrixXd& M_proj)
{
    SelfAdjointEigenSolver < MatrixXd > es(M * M.transpose()); // M*M^T is symmetric -> faster eigenvalue computation
    VectorXd eigenvalues = es.eigenvalues().real();
    MatrixXd U_unsorted = es.eigenvectors().real();
    uint32_t s = M.rows();
    U = MatrixXd(s, s);
    // We sort the (eigenvalue, eigenvector) pairs in descending order
    vector < pair < double, uint32_t >>eigenvalues_sorted(s);
    for (uint32_t i = 0; i < s; ++i)
        eigenvalues_sorted[i] = pair < double, uint32_t >(-eigenvalues(i), i);
    sort(eigenvalues_sorted.begin(), eigenvalues_sorted.end());
    for (uint32_t i = 0; i < s; ++i)
        U.col(i) = U_unsorted.col(eigenvalues_sorted[i].second);
    M_proj = U.transpose() * M;
}

// U is an input parameter and M is decompressed using U (sliced as appropriate)
void unproject(MatrixXd& M, MatrixXd& U, MatrixXd& M_proj, Slice slice) {
    if (not slice.is_standard()) {
        if (slice.points[0] < 0 or slice.points[1] > U.rows()) { // TODO put in decompress.hpp
            cout << "Error: the slicing falls out of the tensor size range" << endl;
            exit(1);
        }
        int8_t sign = (0 < slice.points[2]) - (slice.points[2] < 0);
        if ((sign < 0 and slice.points[0] < slice.points[1]) or (sign > 0 and slice.points[0] > slice.points[1])) {
            cout << "Error: unfeasible slicing" << endl;
            exit(1);
        }
        MatrixXd convolution = MatrixXd::Zero(slice.get_size(), U.rows()); // convolution*U convolves U along the columns
        #pragma omp parallel for
        for (int32_t i = 0; i < slice.get_size(); ++i) {
            switch (slice.reduction) {
                case Downsampling: {
                    convolution(i, slice.points[0]+i*slice.points[2]) = 1; // Delta kernel
                    break;
                }
                case Box: {
                    int32_t start = slice.points[0] + i*slice.points[2] - slice.points[2]/2;
                    int32_t end = max(min(slice.points[0] + i*slice.points[2] + (slice.points[2] - slice.points[2]/2), U.rows()), 0);
                    double kernel_sum = 1./abs(end-start);
                    for (int32_t j = start; sign*j < sign*end; j += sign)
                        convolution(i, j) = kernel_sum; // Box kernel
                    break;
                }
                case Lanczos: {
                    double a = 2*slice.points[2]; // Upscaled Lanczos window
                    int32_t start = max(min(slice.points[0] + i*slice.points[2] - a, U.rows()-1), 0); // Kernel support: [-a, a], clamped
                    int32_t end = max(min(slice.points[0] + i*slice.points[2] + a + 1, U.rows()), -1);
                    double center = slice.points[0] + i*slice.points[2];
                    double kernel_sum = 0;
                    for (int32_t j = start; sign*j < sign*end; j += sign) {
                        double x = (j-center)/abs(slice.points[2]); // Upscaled x
                        if (x == 0)
                            convolution(i, j) = 1;
                        else
                            convolution(i, j) = a*sin(M_PI*x)*sin(M_PI*x/a)/(M_PI*M_PI*x*x); // Lanczos 2 kernel
                        kernel_sum += convolution(i, j);
                    }
                    for (int32_t j = start; sign*j < sign*end; j += sign)
                        convolution(i, j) /= kernel_sum; // Normalize the kernel so that it adds up to 1
                    break;
                }
            }
        }
        M_proj = (convolution * U) * M;
    }
    else
        M_proj = U * M;
}

// Reads a tensor in the buffer data of size s, and compresses it.
// The factor matrices are output parameters
void hosvd_compress(double *data, vector<MatrixXd>& Us, bool verbose)
{
    char n = s.size();

    // First unfolding: special case (elements are already arranged as we want)
    if (verbose) cout << "\tUnfold (1)... " << flush;
    MatrixXd M = MatrixXd::Map(data, s[0], sprod[n]/s[0]);
    MatrixXd M_proj;
    if (verbose) cout << "Project (1)..." << flush;
    project(M, Us[0], M_proj);
    if (verbose) cout << endl;

    // Remaining unfoldings: all of them go matrix -> matrix
    // Input: matrix of size s[dim-1] x (s[0] * ... * s[dim-2] * s[dim] * ... * s[N])
    // Output: matrix of size s[dim] x (s[0] * ... * s[dim-1] * s[dim+1] * ... * s[N])
    for (uint8_t dim = 1; dim < n; ++dim) {
        if (verbose) cout << "\tUnfold (" << dim+1 << ")... " << flush;
        M = MatrixXd(s[dim], sprod[n]/s[dim]); // dim-th factor matrix
        #pragma omp parallel for
        for (int64_t j = 0; j < M_proj.cols(); ++j) {
            uint32_t write_i = (j/sprod[dim-1]) % s[dim];
            size_t base_write_j = j%sprod[dim-1] + j/(sprod[dim-1]*s[dim])*sprod[dim];
            for (int32_t i = 0; i < M_proj.rows(); ++i)
                M(write_i, base_write_j + i*sprod[dim-1]) = M_proj(i, j);
        }
        if (verbose) cout << "\tProject (" << dim+1 << ")... " << flush;
        project(M, Us[dim], M_proj);
        if (verbose) cout << endl;
    }

    // We fold back from matrix into ND tensor
    if (verbose) cout << "\tFold... " << flush << endl;
    #pragma omp parallel for
    for (int32_t i = 0; i < int32_t(s[n-1]); i++)
        for (size_t j = 0; j < sprod[n-1]; j++)
            data[i*sprod[n-1] + j] = M_proj(i, j);
}

// Reads a tensor in the buffer data of size s, and decompresses it in-place
void hosvd_decompress(vector<double>& data, vector<MatrixXd>& Us, bool verbose, vector<Slice>& cutout)
{
    if (rprod[n] == 0) { // Extreme case: 0 ranks
        data = vector<double> (snewprod[n], 0); // Produce a 0 reconstruction of the expected size, and leave
        return;
    }

    // First unfolding: special case (elements are already arranged as we want)
    if (verbose) cout << "\tUnfold (1)... " << flush;
    MatrixXd M = MatrixXd::Map(data.data(), r[0], rprod[n]/r[0]);
    MatrixXd M_proj;
    if (verbose) {
        cout << "\tUnproject (" << 1 << ")";
        if (not cutout[0].is_standard())
            cout << " with cutout " << cutout[0];
        cout << "... " << flush;
    }
    unproject(M, Us[0], M_proj, cutout[0]);
    if (verbose) cout << endl;

    // Remaining unfoldings: all of them go matrix -> matrix
    // Input: matrix of size s[dim-1] x (s[0] * ... * s[dim-2] * s[dim] * ... * s[N])
    // Output: matrix of size s[dim] x (s[0] * ... * s[dim-1] * s[dim+1] * ... * s[N])
    for (uint8_t dim = 1; dim < n; ++dim) {
        if (verbose) cout << "\tUnfold (" << dim+1 << ")... " << flush;
        M = MatrixXd(r[dim], snewprod[dim]*rprod[n]/rprod[dim+1]); // dim-th factor matrix
        #pragma omp parallel for
        for (int64_t j = 0; j < M_proj.cols(); ++j) {
            uint32_t write_i = (j/snewprod[dim-1]) % r[dim];
            size_t base_write_j = j%snewprod[dim-1] + j/(snewprod[dim-1]*r[dim])*snewprod[dim];
            for (int32_t i = 0; i < M_proj.rows(); ++i)
                M(write_i, base_write_j + i*snewprod[dim-1]) = M_proj(i, j);
        }
        if (verbose) {
            cout << "\tUnproject (" << dim+1 << ")";
            if (not cutout[dim].is_standard())
                cout << " with cutout " << cutout[dim];
            cout << "... " << flush;
        }
        unproject(M, Us[dim], M_proj, cutout[dim]);
        if (verbose) cout << endl;
    }

    // We fold back from matrix into ND tensor
    if (verbose) cout << "\tFold... " << flush << endl;
    data.resize(snewprod[n]);
    data.shrink_to_fit();
    #pragma omp parallel for
    for (ptrdiff_t i = 0; i < ptrdiff_t(snewprod[n]); i++)
        data[i] = M_proj(i/snewprod[n-1], i%snewprod[n-1]);
}

#endif // TUCKER_HPP
