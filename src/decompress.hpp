#ifndef __DECOMPRESS_HPP__
#define __DECOMPRESS_HPP__

#include "tthresh.hpp"
#include "tucker.hpp"
#include "zlib_io.hpp"
#include "decode.hpp"
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

void decode_factor(MatrixXd & U, int n_columns) {

    U = MatrixXd(n_columns, n_columns);

    // First, the matrix's maximum, used for quantization
    double maximum;
    read_zlib_stream(reinterpret_cast < unsigned char *>(&maximum), sizeof(maximum));

    // Next, the q for each column
    vector < char >U_q(n_columns);
    for (int i = 0; i < n_columns; ++i)
        read_zlib_stream(reinterpret_cast < unsigned char *>(&U_q[i]), sizeof(U_q[i]));

    // Finally we can dequantize the matrix
    char matrix_rbyte;
    char matrix_rbit = -1;
    for (int j = 0; j < n_columns; ++j) {
        for (int i = 0; i < n_columns; ++i) {
            char q = U_q[j];
            if (q > 0) {
                q = min(63, q + 2);
                unsigned long int quant = 0;
                for (char j = q; j >= 0; --j) {
                    if (matrix_rbit < 0) {
                        matrix_rbit = 7;
                        read_zlib_stream(reinterpret_cast < unsigned char *>(&matrix_rbyte), sizeof(matrix_rbyte));
                    }
                    quant |= ((matrix_rbyte >> matrix_rbit) & 1UL) << j;
                    matrix_rbit--;
                }
                if (q == 63) // The matrix value is read verbatim as a double and we get 0 error
                    U(i, j) = *reinterpret_cast<double*>(&quant);
                else { // We dequantize this matrix value
                    char sign = (quant >> q) & 1UL; // Read the sign bit
                    quant &= ~(1UL << q); // Put the sign bit to zero
                    U(i, j) = -(2 * sign - 1) / ((1UL << q) - double (1)) *maximum * double (quant);
                }
            }
            else
                U(i, j) = 0;
        }
    }
}

void decompress(string compressed_file, string output_file, double *data, bool verbose, bool debug) {

    if (verbose)
        cout << endl << "/***** Decompression *****/" << endl << endl << flush;
    
    // Read output tensor dimensionality, sizes and type
    open_zlib_read_stream(compressed_file);
    char n;
    read_zlib_stream(reinterpret_cast < unsigned char *>(&n), sizeof(n));
    vector < int >s(n);
    read_zlib_stream(reinterpret_cast < unsigned char *>(&s[0]), n * sizeof(s[0]));
    ind_t size = 1;
    for (char i = 0; i < n; ++i)
        size *= s[i];
    if (debug) {
        cout << "Decompressing a tensor of size " << s[0];
        for (char i = 1; i < n; ++i)
            cout << " x " << s[i];
        cout << "..." << endl;
    }
    char io_type_code;
    read_zlib_stream(reinterpret_cast < unsigned char *>(&io_type_code), sizeof(io_type_code));
    char io_type_size;
    if (io_type_code == 0)
        io_type_size = sizeof(unsigned char);
    else if (io_type_code == 1)
        io_type_size = sizeof(unsigned short);
    else if (io_type_code == 2)
        io_type_size = sizeof(int);
    else if (io_type_code == 3)
        io_type_size = sizeof(float);
    else
        io_type_size = sizeof(double);
    vector<double> minimums;
    vector<double> maximums;
    vector<char>chunk_ids(size, 0);
    char chunk_num = 1;
    ind_t assigned = 0;
    while(assigned < size) {
        chunk_info ci;
        read_zlib_stream(reinterpret_cast < unsigned char *>(&ci), sizeof(chunk_info));
        minimums.push_back(ci.minimum);
        maximums.push_back(ci.maximum);
        vector < char >mask;

        decode(ci.compressed_size, mask);

        unsigned long int ind = 0;
        for (unsigned int i = 0; i < mask.size() and ind < size; ++i) {	// TODO: sum directly
            for (char chunk_rbit = 7; chunk_rbit >= 0 and ind < size; --chunk_rbit) {
                while (ind < size and chunk_ids[ind] > 0) // Skip core elements already assigned to a previous chunk
                    ind++;
                if ((mask[i] >> chunk_rbit) & 1) { // If element "ind" belongs to the current chunk
                    chunk_ids[ind] = chunk_num;
                    ++assigned;
                }
                ind++;
            }
        }
        if (verbose)
            cout << "Decoded chunk " << int(chunk_num) << ", compressed_size is " << ci.compressed_size << ", mask has " << mask.size() * 8 << " bits, q=" << int (chunk_num - 1) << endl << flush;
        ++chunk_num;
    }
    
    // Read factor matrices
    if (verbose)
        cout << "Decoding factor matrices... " << flush;
    vector < MatrixXd > Us(n);
    for (char i = 0; i < n; ++i)
        decode_factor(Us[i], s[i]);
    if (verbose)
        cout << "Done" << endl << flush;
    
    // Recover the quantized core and dequantize it
    double *c = new double[size];
    char core_quant_rbyte = 0;
    char core_quant_rbit = -1;

    for (int i = 0; i < size; ++i) {
        int chunk_num = chunk_ids[i];
        char q = chunk_num - 1;
        double chunk_min = minimums[q];
        double chunk_max = maximums[q];

        if (q > 0) {
            unsigned long int quant = 0;
            for (long int j = q; j >= 0; --j) { // Read q bits
                if (core_quant_rbit < 0) {
                    read_zlib_stream(reinterpret_cast < unsigned char *>(&core_quant_rbyte), sizeof(core_quant_rbyte));
                    core_quant_rbit = 7;
                }
                quant |= ((core_quant_rbyte >> core_quant_rbit) & 1UL) << j;
                core_quant_rbit--;
            }
            if (q == 63) { // The core value is read verbatim as a double and we get 0 error
                c[i] = *reinterpret_cast < double* >(&quant);
            }
            else { // We dequantize this core value
                char sign = (quant >> q) & 1UL; // Read the sign bit
                quant &= ~(1UL << q); // Put the sign bit to zero
                double dequant;
                dequant = quant / ((1UL << q) - 1.) * (chunk_max - chunk_min) + chunk_min;
                c[i] = -(sign * 2 - 1) * dequant;
            }
        } else
            c[i] = 0;
    }
    close_zlib_read_stream();

    if (verbose)
        cout << "Reconstructing tensor... " << flush;
    hosvd(c, s, Us, false, verbose);
    if (verbose)
        cout << "Done" << endl << flush;

    ofstream output_stream(output_file.c_str(), ios::out | ios::binary);
    unsigned long int buf_elems = CHUNK;
    char *buffer = new char[io_type_size * buf_elems];
    unsigned long int buffer_wpos = 0;
    double sse = 0;
    double datanorm = 0;
    double datamin = std::numeric_limits < double >::max();
    double datamax = std::numeric_limits < double >::min();
    for (long int i = 0; i < size; ++i) {
        if (io_type_code == 0) {
            reinterpret_cast < unsigned char *>(buffer)[buffer_wpos] = abs(c[i]);
        } else if (io_type_code == 1) {
            reinterpret_cast < unsigned short *>(buffer)[buffer_wpos] = abs(c[i]);
        } else if (io_type_code == 2) {
            reinterpret_cast < int *>(buffer)[buffer_wpos] = c[i];
        } else if (io_type_code == 3) {
            reinterpret_cast < float *>(buffer)[buffer_wpos] = c[i];
        } else {
            reinterpret_cast < double *>(buffer)[buffer_wpos] = c[i];
        }
        buffer_wpos++;
        if (buffer_wpos == buf_elems) {
            buffer_wpos = 0;
            output_stream.write(buffer, io_type_size * buf_elems);
        }
        if (data != NULL) {
            datanorm += data[i] * data[i];
            sse += (data[i] - c[i]) * (data[i] - c[i]);
            datamin = min(datamin, data[i]);
            datamax = max(datamax, data[i]);
        }
    }
    if (buffer_wpos > 0)
        output_stream.write(buffer, io_type_size * buffer_wpos);
    delete[] buffer;
    output_stream.close();

    if (data != NULL) {	// If the uncompressed input is available, we compute the error statistics
        datanorm = sqrt(datanorm);
        double eps = sqrt(sse) / datanorm;
        double rmse = sqrt(sse / size);
        double psnr = 20 * log10((datamax - datamin) / (2 * rmse));
        cout << "eps = " << eps << ", rmse = " << rmse << ", psnr = " << psnr << endl;
    }

    delete[] c;
}

#endif // DECOMPRESS_HPP
