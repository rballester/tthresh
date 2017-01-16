#ifndef __COMPRESS_HPP__
#define __COMPRESS_HPP__

#include <fstream>
#include <vector>
#include "encode.hpp"
#include "tthresh.hpp"
#include "tucker.hpp"
#include "zlib_io.hpp"
#include <unistd.h>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// *** Structure of the compressed file ***
// (1) 1 byte: number of dimensions n
// (2) n * 4 bytes: tensor sizes
// (3) 1 byte: tensor type
// (4) 1 byte: number of chunks
// (5) Per-chunk information and masks: n_chunks * (chunk_info + compressed mask)
// (6) Factor matrices
// (7) The quantized core

void encode_factor(MatrixXd & U, int n_columns, vector < char >&columns_q)
{
    // First, the matrix's maximum, used for quantization
    double maximum = U.array().abs().maxCoeff();
//    output_stream.write(reinterpret_cast < char *>(&maximum), sizeof(double));
    write_zlib_stream(reinterpret_cast< unsigned char *> (&maximum), sizeof(double));

    // Next, the q for each column
    for (int i = 0; i < n_columns; ++i) {
//        output_stream.write(reinterpret_cast < char *>(&columns_q[i]), sizeof(char));
        write_zlib_stream(reinterpret_cast< unsigned char *> (&columns_q[i]), sizeof(char));
    }

    // Finally the matrix itself, quantized
    char matrix_wbyte = 0;
    char matrix_wbit = 7;
    for (int j = 0; j < n_columns; ++j) {
        for (int i = 0; i < n_columns; ++i) {
            char q = columns_q[j];
            if (q > 0) {
                q = min(63, q + 2);	// Seems a good compromise
                unsigned long int to_write = min(((1UL << q) - 1),
                                                 (unsigned long int)
                                                 roundl(abs(U(i, j)) / maximum * ((1UL << q) - 1)));
                to_write |= (U(i, j) < 0) * (1UL << q);	// The sign is the first bit to write
                for (int j = q; j >= 0; --j) {
                    matrix_wbyte |= ((to_write >> j) & 1UL) << matrix_wbit;
                    matrix_wbit--;
                    if (matrix_wbit < 0) {
                        matrix_wbit = 7;
//                        output_stream.write(&matrix_wbyte, sizeof(char));
                        write_zlib_stream(reinterpret_cast< unsigned char *> (&matrix_wbyte), sizeof(char));
                        matrix_wbyte = 0;
                    }
                }
            }
        }
    }
    if (matrix_wbit < 7) {
//        output_stream.write(&matrix_wbyte, sizeof(char));
        write_zlib_stream(reinterpret_cast< unsigned char *> (&matrix_wbyte), sizeof(char));
    }
}

double *compress(string input_file, string compressed_file, string io_type, vector < int >&s, Target target, double target_value, unsigned long int skip_bytes, bool verbose, bool debug)
{
    if (verbose)
        cout << endl << "/***** Compression *****/" << endl << endl << flush;

    /**************************/
    // Check input data type
    /**************************/

    unsigned char n = s.size();
    unsigned long int size = 1; // Total number of tensor elements
    for (int i = 0; i < n; ++i)
        size *= s[i];
    char io_type_size, io_type_code;
    if (io_type == "uchar") {
        io_type_size = sizeof(unsigned char);
        io_type_code = 0;
    }
    else if (io_type == "ushort") {
        io_type_size = sizeof(unsigned short);
        io_type_code = 1;
    }
    else if (io_type == "int") {
        io_type_size = sizeof(int);
        io_type_code = 2;
    }
    else if (io_type == "float") {
        io_type_size = sizeof(float);
        io_type_code = 3;
    }
    else if (io_type == "double") {
        io_type_size = sizeof(double);
        io_type_code = 4;
    }
    
    /******************/
    // Check file sizes
    /******************/

    unsigned long int expected_size = skip_bytes + size * io_type_size;
    ifstream input_stream(input_file.c_str(), ios::in | ios::binary);
    if (!input_stream.is_open()) {
        cout << "Could not open \"" << input_file << "\"" << endl;
        exit(1);
    }
    streampos fsize = input_stream.tellg();	// Check that buffer size matches expected size
    input_stream.seekg(0, ios::end);
    fsize = input_stream.tellg() - fsize;
    if (expected_size != fsize) {
        cout << "Invalid file size: expected ";
        if (skip_bytes > 0)
            cout << skip_bytes << " + ";
        cout << "(" << s[0];
        for (int i = 1; i < n; ++i)
            cout << "*" << s[i];
        cout << ")*" << int (io_type_size) << " = " << expected_size << " bytes, but found " << fsize << " bytes";
        if (expected_size > fsize) {
            cout << " (" << expected_size / double (fsize) << " times too small)";
        } else {
            cout << " (" << fsize / double (expected_size) << " times too large)";
        }
        if (skip_bytes == 0 and expected_size < fsize and fsize < expected_size + 1000) {
            cout << ". Perhaps the file has a header (use flag -k)?";
        }
        cout << endl;
        exit(1);
    }

    /********************************************/
    // Save tensor dimensionality, sizes and type
    /********************************************/

    open_zlib_write_stream(compressed_file.c_str());
    write_zlib_stream(reinterpret_cast < unsigned char *> (&n), sizeof(char));
    write_zlib_stream(reinterpret_cast < unsigned char *> (&s[0]), n*sizeof(int));
    write_zlib_stream(reinterpret_cast < unsigned char *> (&io_type_code), sizeof(char));

    /*****************************/
    // Load input file into memory
    /*****************************/

    input_stream.seekg(0, ios::beg);
    char *in = new char[size * io_type_size];
    input_stream.read(in, size * io_type_size);
    input_stream.close();

    // Cast the tensor to doubles
    double *data;
    double dmin = numeric_limits < double >::max(), dmax = numeric_limits < double >::min(), dnorm = 0;	// Tensor statistics
    if (io_type == "double")
        data = (double *) in + skip_bytes;
    else
        data = new double[size];
    for (int i = 0; i < size; ++i) {
        if (io_type_code == 0) {
            data[i] = *reinterpret_cast< unsigned char* >(&in[skip_bytes + i * io_type_size]);
        } else if (io_type_code == 1) {
            data[i] = *reinterpret_cast< unsigned short* >(&in[skip_bytes + i * io_type_size]);
        } else if (io_type_code == 2) {
            data[i] = *reinterpret_cast< int* >(&in[skip_bytes + i * io_type_size]);
        } else if (io_type_code == 3) {
            data[i] = *reinterpret_cast< float* >(&in[skip_bytes + i * io_type_size]);
        }
        dmin = min(dmin, data[i]); // Compute statistics, since we're at it
        dmax = max(dmax, data[i]);
        dnorm += data[i] * data[i];
    }
    dnorm = sqrt(dnorm);
    if (io_type_code != 4)
        delete[]in;
    if (debug) cout << "Tensor statistics: min = " << dmin << ", max = " << dmax << ", norm = " << dnorm << endl;

    // Compute the cumulative products (useful later on for index computations)
    vector<unsigned long int> sprod(n+1); // Cumulative size products. The i-th element contains s[0]*...*s[i-1]
    sprod[0] = 1;
    for (int dim = 0; dim < n; ++dim)
        sprod[dim+1] = sprod[dim]*s[dim];

    /**********************************************************************/
    // Compute the target SSE (sum of squared errors) from the given metric
    /**********************************************************************/

    double sse;
    if (target == eps)
        sse = pow(target_value * dnorm, 2);
    else if (target == rmse)
        sse = pow(target_value, 2) * size;
    else
        sse = pow((dmax - dmin) / (2 * (pow(10, target_value / 20))), 2) * size;
    double lim = sse / size;
    if (verbose)
        cout << "We target MSE = " << lim << endl;

    /*********************************/
    // Create and decompose the tensor
    /*********************************/

    if (verbose)
        cout << "Decomposing the " << int (n) << "D tensor... " << flush;
    double *c = new double[size];	// Tucker core
    memcpy(c, data, size * sizeof(double));
    vector < MatrixXd > Us(n);	// Tucker factor matrices
    hosvd(c, s, Us, true, verbose);
    if (verbose)
        cout << "Done" << endl << flush;

    /***********************************/
    // Sort abs(core) in ascending order
    /***********************************/

    if (verbose)
        cout << "Sorting core's absolute values... " << flush;
    vector < pair < double, int >>sorting(size);
    for (int i = 0; i < size; ++i)
        sorting[i] = pair < double, int >(abs(c[i]), i);
    sort(sorting.begin(), sorting.end());
    if (verbose)
        cout << "Done" << endl << flush;

    /************************************************/
    // Generate adaptive chunks from the sorted curve
    /************************************************/

    unsigned long int adder = 1;
    char q = 0;
    unsigned long int left = 0;
    unsigned long int old_right = left;	// Inclusive bound
    unsigned long int right = left;	// Exclusive bound
    vector < int >chunk_ids(size, 0);
    int chunk_num = 1;
    vector < vector < char >>Us_q(n);
    for (int i = 0; i < n; ++i)
        Us_q[i] = vector < char >(s[i], 0);
                                    int assigned = 0;
    while (left < size) {
        while (left < size and q < 63) {
            right = min(size, old_right + adder);
            double chunk_min = sorting[left].first;
            double chunk_max = sorting[right - 1].first;
            double sse = 0;
            if (right > left + 1) {
                if (q > 0) {
                    for (int i = left; i < right; ++i) {	// TODO Can we approximate the error computation?
                        long int quant = roundl((sorting[i].first - chunk_min) * ((1UL << q) - 1.) / (chunk_max - chunk_min));
                        double dequant = quant * (chunk_max - chunk_min) / ((1UL << q) - 1.) + chunk_min;
                        sse += (sorting[i].first - dequant) * (sorting[i].first - dequant);
                    }
                } else {
                    for (int i = left; i < right; ++i)
                        sse += (sorting[i].first - chunk_min) * (sorting[i].first - chunk_min);
                }
            }
            double mse = sse / (right - left);
            if (debug)
                cout << "We try [" << left << "," << right << "), adder = " << adder << ", mse = " << mse << endl;
            if (mse >= 0.9 * lim or right == size) {
                if (mse >= lim) {
                    if (adder > 1) {
                        adder = ceil(adder / 4.);
                        continue;
                    } else {
                        right = old_right;
                        break;
                    }
                } else
                    break;
            } else {
                old_right = right;
                adder *= 2;
            }
        }

        if (q == 63)
            right = size;

        int chunk_size = (right - left);
        double chunk_min = sorting[left].first;
        double chunk_max = sorting[right - 1].first;

        /********************************************/
        // Quantize (in-place) the core elements
        /********************************************/

        // If q = 0 there's no need to store anything quantized, not even the sign
        // If q = 63, values are kept as they are and we forget about quantization
        if (q > 0 and q < 63) {
            for (int i = left; i < right; ++i) {
                unsigned long int to_write = 0;
                if (chunk_size > 1)
                    // The following min() prevents overflowing the q-bit representation when converting double -> long int
                    to_write = min(((1UL << q) - 1), (unsigned long int)
                                   roundl((sorting[i].first - chunk_min) / (chunk_max - chunk_min) * ((1UL << q) - 1)));
                to_write |= (c[sorting[i].second] < 0) * (1UL << q);
                c[sorting[i].second] = static_cast < double >(to_write);
            }
        }

        /********************************************/
        // Save mask and compute RLE+Huffman out of it
        /********************************************/

        for (int i = left; i < right; ++i) {
            unsigned long int index = sorting[i].second;
            chunk_ids[index] = chunk_num;
                                                                                    assigned++;
            // We use this loop also to update the needed quantization bits per factor column
            for (int dim = 0; dim < n; ++dim) {
                int coord = index % sprod[dim+1] / sprod[dim];
                Us_q[dim][coord] = max(Us_q[dim][coord], q);
            }
        }

        vector < char >mask;
        char mask_wbyte = 0;
        char mask_wbit = 7;
        for (int i = 0; i < size; ++i) {
            if (chunk_ids[i] == 0)
                mask_wbit--;
            else if (chunk_ids[i] == chunk_num) {
                mask_wbyte |= 1 << mask_wbit;
                mask_wbit--;
            }
            if (mask_wbit < 0) {
                mask.push_back(mask_wbyte);
                mask_wbyte = 0;
                mask_wbit = 7;
            }
        }
        if (mask_wbit < 7)
            mask.push_back(mask_wbyte);
        vector < char >compressed_mask;
        encode(mask, compressed_mask); // TODO: make encode() write directly into the zlib stream

        chunk_info ci;
        ci.compressed_size = compressed_mask.size();
        ci.minimum = chunk_min;
        ci.maximum = chunk_max;
        write_zlib_stream(reinterpret_cast < unsigned char *> (&ci), sizeof(chunk_info));
        write_zlib_stream(reinterpret_cast < unsigned char *> (&compressed_mask[0]), compressed_mask.size()*sizeof(char));

        if (verbose) {
            int quant_bits = 0;
            if (q > 0)
                quant_bits = (q + 1) * (right - left);	// The "+1" is for the sign
            cout << "Encoded chunk " << chunk_num << ", compressed_size=" << ci.compressed_size << ", min=" << chunk_min << ", max=" << chunk_max << ", quant_bits=" << quant_bits << ", q=" << int (q) << ", bits=[" << left << "," << right << "), size=" << right - left << endl << flush;
        }

        // Update control variables
        q++;
        left = right;
        old_right = left;
        chunk_num++;
    }

    /*********************************/
    // Encode and save factor matrices
    /*********************************/

    if (debug) {
        cout << "q's for the factor columns: " << endl;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < s[i]; ++j)
                cout << " " << int (Us_q[i][j]);
            cout << endl;
        }
    }

    if (verbose)
        cout << "Encoding factor matrices... " << flush;
    for (int i = 0; i < n; ++i)
        encode_factor(Us[i], s[i], Us_q[i]);
    if (verbose)
        cout << "Done" << endl << flush;

    /********************************************/
    // Save the core encoding
    /********************************************/

    if (verbose)
        cout << "Saving core quantization... " << flush;
    unsigned char core_quant_wbyte = 0;
    char core_quant_wbit = 7;
//    vector<char> buf;
    for (int i = 0; i < size; ++i) {
        chunk_num = chunk_ids[i];
        char q = chunk_num - 1;
        if (q > 0) {
            for (long int j = q; j >= 0; --j) {
                core_quant_wbyte |= ((static_cast < unsigned long int >(c[i]) >> j) &1UL) << core_quant_wbit;
                core_quant_wbit--;
                if (core_quant_wbit < 0) {
//                    output_stream.write(&core_quant_wbyte, sizeof(char));
//                    buf.push_back(core_quant_wbyte);
                    write_zlib_stream(reinterpret_cast < unsigned char *> (&core_quant_wbyte), sizeof(char));
                    core_quant_wbyte = 0;
                    core_quant_wbit = 7;
                }
            }
        }
    }
    if (core_quant_wbit < 7) {
//        output_stream.write(&core_quant_wbyte, sizeof(char));
//        buf.push_back(core_quant_wbyte);
        int ret = write_zlib_stream(reinterpret_cast < unsigned char *> (&core_quant_wbyte), sizeof(char));
        assert(ret == Z_OK);
    }
//    write_zlib_stream(reinterpret_cast < unsigned char *> (&buf[0]), buf.size()*sizeof(char));
    if (verbose)
        cout << "Done" << endl << flush;
    delete[]c;
//    output_stream.close();
    int ret = close_zlib_write_stream();
    assert(ret == Z_OK);

    /***********************************************/
    // Compute statistics of the resulting compression rate
    /***********************************************/
    
    ifstream bpv_stream(compressed_file.c_str(), ios::in | ios::binary);
    streampos beginning = bpv_stream.tellg();
    bpv_stream.seekg(0, ios::end);
    long int newbits = (bpv_stream.tellg() - beginning) * 8;
    cout << "oldbits = " << size * io_type_size * 8L << ", newbits = " << newbits << ", compressionrate = " << size * io_type_size * 8L / double (newbits)
         << ", bpv = " << newbits / double (size) << endl << flush;
    bpv_stream.close();
    return data;
}

#endif // COMPRESS_HPP
