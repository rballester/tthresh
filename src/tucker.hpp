#ifndef __TUCKER_HPP__
#define __TUCKER_HPP__

#include <chrono>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// Projects an unfolded core M into M_proj using the transformation matrix U.
// If compress == true, U is an output parameter, computed as the HOSVD of
// the tensor (left singular vectors of M) and M is compressed using U.transpose().
// If compress == false, U is an input parameter and M is decompressed using U.
void project(MatrixXd& M, MatrixXd& U, MatrixXd& M_proj, bool compress)
{
    if (compress) {
        SelfAdjointEigenSolver < MatrixXd > es(M * M.transpose()); // M*M^T is symmetric -> faster eigenvalue computation
        VectorXd eigenvalues = es.eigenvalues().real();
        MatrixXd U_unsorted = es.eigenvectors().real();
        int s = M.rows();
        U = MatrixXd(s, s);
        // We sort the (eigenvalue, eigenvector) pairs in descending order
        vector < pair < double, int >>eigenvalues_sorted(s);
        for (int i = 0; i < s; ++i)
            eigenvalues_sorted[i] = pair < double, int >(-eigenvalues(i), i);
        sort(eigenvalues_sorted.begin(), eigenvalues_sorted.end());
        for (int i = 0; i < s; ++i)
            U.col(i) = U_unsorted.col(eigenvalues_sorted[i].second);
        M_proj = U.transpose() * M;
    } else
        M_proj = U * M;
}

// Reads a tensor in the buffer data of size s, and compresses or decompresses it in-place.
// If compress == true, then the factor matrices are output parameters
void hosvd(double *data, vector<int>& s, vector<MatrixXd>& Us, bool compress, bool verbose)
{
    char n = s.size();

    // First unfolding: special case (elements are already arranged as we want)
    if (verbose) cout << "\tUnfold (1)... " << flush;
    MatrixXd M = MatrixXd::Map(data, s[0], sprod[n]/s[0]);
    MatrixXd M_proj;
    if (verbose) cout << "Project (1)..." << flush;
    project(M, Us[0], M_proj, compress);
    if (verbose) cout << endl;

    // Remaining unfoldings: all of them go matrix -> matrix
    // Input: matrix of size s[dim-1] x (s[0] * ... * s[dim-2] * s[dim] * ... * s[N])
    // Output: matrix of size s[dim] x (s[0] * ... * s[dim-1] * s[dim+1] * ... * s[N])
    for (int dim = 1; dim < n; ++dim) {
        if (verbose) cout << "\tUnfold (" << dim+1 << ")... " << flush;
        M = MatrixXd(s[dim], sprod[n]/s[dim]); // dim-th factor matrix
        #pragma omp parallel for
        for (long int j = 0; j < sprod[n]/s[dim-1]; ++j) {
            int write_i = (j/sprod[dim-1]) % s[dim];
            long int base_write_j = j%sprod[dim-1] + j/(sprod[dim-1]*s[dim])*sprod[dim];
            for (int i = 0; i < s[dim-1]; ++i)
                M(write_i, base_write_j + i*sprod[dim-1]) = M_proj(i, j);
        }
        if (verbose) cout << "\tProject (" << dim+1 << ")... " << flush;
        project(M, Us[dim], M_proj, compress);
        if (verbose) cout << endl;
    }

    // We fold back from matrix into ND tensor
    if (verbose) cout << "\tFold... " << flush << endl;
    #pragma omp parallel for
    for (int i = 0; i < s[n-1]; i++)
        for (long int j = 0; j < sprod[n-1]; j++)
            data[i*sprod[n-1] + j] = M_proj(i, j);
}

#endif // TUCKER_HPP
