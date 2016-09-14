#ifndef __TUCKER_HPP__
#define __TUCKER_HPP__

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// Projects a core M into M_proj using the transformation matrix U.
// If compress == true, U is an output parameter, computed as the HOSVD of
// the tensor (left singular vectors of M) and M is compressed using U.transpose().
// If compress == false, U is an input parameter and M is decompressed using U.
void project(MatrixXd& M, MatrixXd& U, MatrixXd& M_proj, bool compress) {

    if (compress) {
        SelfAdjointEigenSolver<MatrixXd> es(M*M.transpose());
        VectorXd eigenvalues = es.eigenvalues().real();
        MatrixXd U_unsorted = es.eigenvectors().real();

        int s = M.rows();
        U = MatrixXd(s,s);
        vector< pair<double,int> > eigenvalues_sorted(s);
        for (int i = 0; i < s; ++i)
            eigenvalues_sorted[i] = pair<double,int>(-eigenvalues(i),i);
        sort(eigenvalues_sorted.begin(),eigenvalues_sorted.end());
        for (int i = 0; i < s; ++i)
            U.col(i) = U_unsorted.col(eigenvalues_sorted[i].second);
        M_proj = U.transpose() * M;
    }
    else
        M_proj = U * M;
}

// Reads a tensor in the buffer data of size s, compresses or decompresses it
// and puts the result in the same buffer.
// If compress == true, then U1, U2 and U3 are output parameters.
void tucker(double* data, int* s, MatrixXd& U1, MatrixXd& U2, MatrixXd& U3, bool compress) {

    MatrixXd M, M_proj;
    M = MatrixXd(s[0],s[1]*s[2]);
    for (int i = 0; i < s[0]*s[1]*s[2]; ++i) {
        int write_i = i % s[0];
        int write_j = i / s[0];
        M(write_i,write_j) = data[i];
    }
    project(M,U1,M_proj,compress);

    M = MatrixXd(s[1],s[0]*s[2]);
    for (int j = 0; j < s[1]*s[2]; ++j) {
        int write_i = j % s[1];
        for (int i = 0; i < s[0]; ++i) {
            int write_j = i + s[0]*(j/s[1]);
            M(write_i,write_j) = M_proj(i,j);
        }
    }
    project(M,U2,M_proj,compress);

    M = MatrixXd(s[2],s[0]*s[1]);
    for (int j = 0; j < s[0]*s[2]; j++) {
        int write_i = j / s[0];
        for (int i = 0; i < s[1]; i++) {
            int write_j = (j%s[0]) + i*s[0];
            M(write_i,write_j) = M_proj(i,j);
        }
    }
    project(M,U3,M_proj,compress);

    for (int j = 0; j < s[0]*s[1]; j++) {
        for (int i = 0; i < s[2]; i++) {
            int write_i = j + i*s[0]*s[1];
            data[write_i] = M_proj(i,j);
        }
    }

}

#endif
