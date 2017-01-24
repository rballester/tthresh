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

// *** Variable types ***
// Number of dimensions: char
// Chunk counting: char
// Size of each dimension: unsigned int
// Total size: size_t

// *** Structure of the compressed file ***
// (1) 1 byte: number of dimensions n
// (2) n * 4 bytes: tensor sizes
// (3) 1 byte: tensor type
// (4) 1 byte: number of chunks
// (5) Per-chunk information and masks: n_chunks * (chunk_info + compressed mask)
// (6) Factor matrices
// (7) The quantized core

void encode_factor(MatrixXd & U, unsigned int n_columns, vector < char >&columns_q) {

    // First, the matrix's maximum, used for quantization
    double maximum = U.array().abs().maxCoeff();
    zlib_write_stream(reinterpret_cast< unsigned char *> (&maximum), sizeof(maximum));

    // Next, the q for each column
    for (unsigned int i = 0; i < n_columns; ++i)
        zlib_write_stream(reinterpret_cast< unsigned char *> (&columns_q[i]), sizeof(columns_q[i]));

    // Finally the matrix itself, quantized
    zlib_open_wbit();
    for (unsigned int j = 0; j < n_columns; ++j) {
        for (unsigned int i = 0; i < n_columns; ++i) {
            char q = columns_q[j];
            if (q > 0) {
                q = min(63, q + 2);	// Seems a good compromise
                unsigned long int to_write;
                if (q == 63)
                    to_write = *reinterpret_cast<unsigned long int*>(&U(i,j));
                else {
                    to_write = min(((1UL << q) - 1), (unsigned long int) roundl(abs(U(i, j)) / maximum * ((1UL << q) - 1)));
                    if (U(i, j) < 0)
                        to_write |=  1UL << q;// The sign is the most significant bit
                }
                for (char j = q; j >= 0; --j)
                    zlib_write_bit((to_write >> j) & 1UL);
            }
        }
    }
    zlib_close_wbit();
}

double *compress(string input_file, string compressed_file, string io_type, vector < int >&s, Target target, double target_value, unsigned long int skip_bytes, bool verbose=false, bool debug=false) {

    if (verbose)
        cout << endl << "/***** Compression *****/" << endl << endl << flush;

    /**************************/
    // Check input data type
    /**************************/

    unsigned char n = s.size();
    ind_t size = 1; // Total number of tensor elements
    for (char i = 0; i < n; ++i)
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
    else {
        io_type_size = sizeof(double);
        io_type_code = 4;
    }
    
    /************************/
    // Check input file sizes
    /************************/

    ind_t expected_size = skip_bytes + size * io_type_size;
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
        for (char i = 1; i < n; ++i)
            cout << "*" << s[i];
        cout << ")*" << int(io_type_size) << " = " << expected_size << " bytes, but found " << fsize << " bytes";
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

    open_zlib_write(compressed_file.c_str());
    zlib_write_stream(reinterpret_cast < unsigned char *> (&n), sizeof(n));
    zlib_write_stream(reinterpret_cast < unsigned char *> (&s[0]), n*sizeof(s[0]));
    zlib_write_stream(reinterpret_cast < unsigned char *> (&io_type_code), sizeof(io_type_code));

    /*****************************/
    // Load input file into memory
    /*****************************/

    input_stream.seekg(0, ios::beg);
    char *in = new char[size * io_type_size];
    input_stream.read(in, size * io_type_size);
    input_stream.close();

    // Cast the data to doubles
    double *data;
    double datamin = numeric_limits < double >::max(); // Tensor statistics
    double datamax = numeric_limits < double >::min();
    double datanorm = 0;
    if (io_type == "double")
        data = (double *) in + skip_bytes;
    else
        data = new double[size];
    for (ind_t i = 0; i < size; ++i) {
        switch (io_type_code) {
            case 0:
                data[i] = *reinterpret_cast< unsigned char* >(&in[skip_bytes + i * io_type_size]);
                break;
            case 1:
                data[i] = *reinterpret_cast< unsigned short* >(&in[skip_bytes + i * io_type_size]);
                break;
            case 2:
                data[i] = *reinterpret_cast< int* >(&in[skip_bytes + i * io_type_size]);
                break;
            case 3:
                data[i] = *reinterpret_cast< float* >(&in[skip_bytes + i * io_type_size]);
                break;
        }
        datamin = min(datamin, data[i]); // Compute statistics, since we're at it
        datamax = max(datamax, data[i]);
        datanorm += data[i] * data[i];
    }
    datanorm = sqrt(datanorm);
    if (io_type_code != 4)
        delete[]in;
    if (debug) cout << "Input statistics: min = " << datamin << ", max = " << datamax << ", norm = " << datanorm << endl;

    cumulative_size_products(s, n);

    /**********************************************************************/
    // Compute the target SSE (sum of squared errors) from the given metric
    /**********************************************************************/

    double sse;
    if (target == eps)
        sse = pow(target_value * datanorm, 2);
    else if (target == rmse)
        sse = pow(target_value, 2) * size;
    else
        sse = pow((datamax - datamin) / (2 * (pow(10, target_value / 20))), 2) * size;
    double lim = sse / size;
    if (verbose) {
        double eps = sqrt(sse) / datanorm;
        double rmse = sqrt(sse / size);
        double psnr = 20 * log10((datamax - datamin) / (2 * rmse));
        cout << "We target eps = " << eps << ", rmse = " << rmse << ", psnr = " << psnr << endl;
    }

    /*********************************/
    // Create and decompose the tensor
    /*********************************/

    if (verbose)
        cout << "Decomposing the " << int(n) << "D tensor... " << flush;
    double *c = new double[size];	// Tucker core
    memcpy(c, data, size * sizeof(double));
    vector<MatrixXd> Us(n); // Tucker factor matrices
    hosvd(c, s, Us, true, verbose);
    if (verbose)
        cout << "Done" << endl << flush;

    /***********************************/
    // Sort abs(core) in ascending order
    /***********************************/

    if (verbose)
        cout << "Sorting core's absolute values... " << flush;
    vector < pair < double, ind_t >>sorting(size);
    for (ind_t i = 0; i < size; ++i)
        sorting[i] = pair < double, ind_t >(abs(c[i]), i);
    sort(sorting.begin(), sorting.end());
    if (verbose)
        cout << "Done" << endl << flush;

    /************************************************/
    // Generate adaptive chunks from the sorted curve
    /************************************************/

    ind_t adder = 1;
    char q = 0;
    ind_t left = 0;
    ind_t old_right = left;	// Inclusive bound
    ind_t right = left;	// Exclusive bound
    vector < char >chunk_ids(size, 0);
    char chunk_num = 1;
    vector < vector < char >>Us_q(n);
    for (char i = 0; i < n; ++i)
        Us_q[i] = vector < char >(s[i], 0);

    while (left < size) {
        while (left < size and q < 63) {
            right = min(size, old_right + adder);
            double chunk_min = sorting[left].first;
            double chunk_max = sorting[right - 1].first;
            double sse = 0;
            if (right > left + 1) {
                if (q > 0) {
                    for (ind_t i = left; i < right; ++i) { // TODO: Can we approximate the error due to quantization?
                        unsigned long int quant = roundl((sorting[i].first - chunk_min) * ((1UL << q) - 1.) / (chunk_max - chunk_min));
                        double dequant = quant * (chunk_max - chunk_min) / ((1UL << q) - 1.) + chunk_min;
                        sse += (sorting[i].first - dequant) * (sorting[i].first - dequant);
                    }
                } else {
                    for (ind_t i = left; i < right; ++i)
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

        ind_t chunk_size = (right - left);
        double chunk_min = sorting[left].first;
        double chunk_max = sorting[right - 1].first;

        /********************************************/
        // Quantize (in-place) the core elements
        /********************************************/

        // If q = 0 there's no need to store anything quantized, not even the sign
        // If q = 63, values are kept as they are (doubles) and we forget about quantization
        if (q > 0 and q < 63) {
            #pragma omp parallel for
            for (int i = left; i < right; ++i) {
                unsigned long int to_write = 0;
                if (chunk_size > 1)
                    // The following min() prevents overflowing the q-bit representation when converting double -> long int
                    to_write = min(((1UL << q) - 1), (unsigned long int)
                                   roundl((sorting[i].first - chunk_min) / (chunk_max - chunk_min) * ((1UL << q) - 1)));
                to_write |= (c[sorting[i].second] < 0) * (1UL << q);
                memcpy(&c[sorting[i].second], (void*)&to_write, sizeof(to_write));
            }
        }

        /********************************************/
        // Save mask and compute RLE+Huffman out of it
        /********************************************/
        #pragma omp parallel for
        for (int i = left; i < right; ++i) {
            unsigned long int index = sorting[i].second;
            chunk_ids[index] = chunk_num;

            // We use this loop also to update the needed quantization bits per factor column
            for (char dim = 0; dim < n; ++dim) {
                int coord = index % sprod[dim+1] / sprod[dim];
                Us_q[dim][coord] = max(Us_q[dim][coord], q);
            }
        }

        vector<unsigned long int> counters;

        // RLE
        bool current_bit = false;
        bool last_bit = false;
        unsigned long int counter = 0;
        for (int i = 0; i < size; ++i) {
            if (chunk_ids[i] == 0)
                current_bit = false;
            else if (chunk_ids[i] == chunk_num)
                current_bit = true;
            else
                continue;
            if (current_bit == last_bit)
                counter++;
            else {
                counters.push_back(counter);
                counter = 1;
                last_bit = current_bit;
            }
        }
        counters.push_back(counter);

        zlib_write_stream(reinterpret_cast<unsigned char*> (&chunk_min), sizeof(chunk_min));
        zlib_write_stream(reinterpret_cast<unsigned char*> (&chunk_max), sizeof(chunk_max));
        encode(counters);

        if (verbose) {
            ind_t quant_bits = 0;
            if (q > 0)
                quant_bits = (q + 1) * (right - left);	// The "+1" is for the sign
            cout << "Encoded chunk " << int(chunk_num) << ", min=" << chunk_min << ", max=" << chunk_max << ", quant_bits=" << quant_bits << ", q=" << int (q) << ", bits=[" << left << "," << right << "), size=" << right - left << endl << flush;
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
        for (char i = 0; i < n; ++i) {
            for (int j = 0; j < s[i]; ++j)
                cout << " " << int(Us_q[i][j]);
            cout << endl;
        }
    }

    if (verbose)
        cout << "Encoding factor matrices... " << flush;
    for (char i = 0; i < n; ++i)
        encode_factor(Us[i], s[i], Us_q[i]);
    if (verbose)
        cout << "Done" << endl << flush;

    /************************/
    // Save the core encoding
    /************************/

    if (verbose)
        cout << "Saving core quantization... " << flush;
    unsigned char core_quant_wbyte = 0;
    char core_quant_wbit = 7;
    for (ind_t i = 0; i < size; ++i) {
        chunk_num = chunk_ids[i];
        char q = chunk_num - 1;
        if (q > 0) {
            for (char j = q; j >= 0; --j) {
                core_quant_wbyte |= ((*reinterpret_cast < unsigned long int* >(&c[i]) >> j) &1UL) << core_quant_wbit;
                core_quant_wbit--;
                if (core_quant_wbit < 0) {
                    zlib_write_stream(reinterpret_cast < unsigned char *> (&core_quant_wbyte), sizeof(char));
                    core_quant_wbyte = 0;
                    core_quant_wbit = 7;
                }
            }
        }
    }
    if (core_quant_wbit < 7)
        zlib_write_stream(reinterpret_cast < unsigned char *> (&core_quant_wbyte), sizeof(char));
    if (verbose)
        cout << "Done" << endl << flush;
    delete[] c;
    close_zlib_write();

    /******************************************************************/
    // Compute and display statistics of the resulting compression rate
    /******************************************************************/

    ind_t newbits = zs.total_written_bytes * 8;
    cout << "oldbits = " << size * io_type_size * 8L << ", newbits = " << newbits << ", compressionrate = " << size * io_type_size * 8L / double (newbits)
         << ", bpv = " << newbits / double (size) << endl << flush;
    return data;
}

#endif // COMPRESS_HPP
