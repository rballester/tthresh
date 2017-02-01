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
// Number of dimensions: uint8_t
// Chunk counting: uint8_t
// Size of each dimension: uint32_t
// Total size: size_t

// *** Structure of the compressed file ***
// (1) 1 byte: number of dimensions n
// (2) n * 4 bytes: tensor sizes
// (3) 1 byte: tensor type
// (4) Per-chunk information and masks: n_chunks * (minimum + maximum + compressed mask)
// (5) n * 4 bytes: tensor ranks
// (6) Factor matrices
// (7) The quantized core

void encode_factor(Block<MatrixXd, -1, -1, true> U, vector < uint8_t >&U_q) {

    // First, the matrix's maximum, used for quantization
    double maximum = U.array().abs().maxCoeff();
    zlib_write_stream(reinterpret_cast<uint8_t*> (&maximum), sizeof(maximum));

    // Next, the q for each column
    zlib_write_stream(reinterpret_cast<uint8_t*> (&U_q[0]), U.cols()*sizeof(uint8_t));

    // Finally the matrix itself, quantized
    zlib_open_wbit();
    for (uint32_t j = 0; j < U.cols(); ++j) {
        for (uint32_t i = 0; i < U.rows(); ++i) {
            uint8_t q = U_q[j];
            if (q > 0) {
                q = min(63, q + 2);	// Seems a good compromise
                uint64_t to_write;
                if (q == 63)
                    to_write = *reinterpret_cast<uint64_t*>(&U(i,j));
                else {
                    to_write = min(((1UL << q) - 1), (uint64_t) roundl(abs(U(i, j)) / maximum * ((1UL << q) - 1)));
                    if (U(i, j) < 0)
                        to_write |=  1UL << q;// The sign is the most significant bit
                }
                zlib_write_bit(to_write, q+1);
            }
        }
    }
    zlib_close_wbit();
}

double *compress(string input_file, string compressed_file, string io_type, Target target, double target_value, size_t skip_bytes, bool verbose=false, bool debug=false) {

    uint8_t n = s.size();
    if (verbose) {
        cout << endl << "/***** Compression: " << to_string(n) << "D tensor of size " << s[0];
        for (uint8_t i = 1; i < n; ++i)
            cout << " x " << s[i];
        cout << " *****/" << endl << endl;
    }

    /***********************/
    // Check input data type
    /***********************/

    size_t size = 1; // Total number of tensor elements
    for (uint8_t i = 0; i < n; ++i)
        size *= s[i];
    uint8_t io_type_size, io_type_code;
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

    size_t expected_size = skip_bytes + size * io_type_size;
    ifstream input_stream(input_file.c_str(), ios::in | ios::binary);
    if (!input_stream.is_open()) {
        cout << "Error: could not open \"" << input_file << "\"" << endl;
        exit(1);
    }
    size_t fsize = input_stream.tellg(); // Check that buffer size matches expected size
    input_stream.seekg(0, ios::end);
    fsize = size_t(input_stream.tellg()) - fsize;
    if (expected_size != fsize) {
        cout << "Invalid file size: expected ";
        if (skip_bytes > 0)
            cout << skip_bytes << " + ";
        cout << "(" << s[0];
        for (uint8_t i = 1; i < n; ++i)
            cout << "*" << s[i];
        cout << ")*" << int(io_type_size) << " = " << expected_size << " bytes, but found " << fsize << " bytes";
        if (expected_size > fsize)
            cout << " (" << expected_size / double (fsize) << " times too small)";
        else
            cout << " (" << fsize / double (expected_size) << " times too large)";
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
    for (size_t i = 0; i < size; ++i) {
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

    sprod = vector<size_t> (n+1); // Cumulative size products. The i-th element contains s[0]*...*s[i-1]
    sprod[0] = 1;
    for (uint8_t i = 0; i < n; ++i)
        sprod[i+1] = sprod[i]*s[i];

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
        start_timer("Tucker decomposition...\n");
    double *c = new double[size];	// Tucker core
    memcpy(c, data, size * sizeof(double));
    vector<MatrixXd> Us(n); // Tucker factor matrices
    hosvd_compress(c, Us, verbose);
    if (verbose)
        stop_timer();

    /***********************************/
    // Sort abs(core) in ascending order
    /***********************************/

    if (verbose)
        start_timer("Sorting core's absolute values... ");
    vector< pair<double,size_t> > sorting(size);
    for (size_t i = 0; i < size; ++i)
        sorting[i] = pair < double, size_t >(abs(c[i]), i);
    sort(sorting.begin(), sorting.end());
    if (verbose)
        stop_timer();

    /************************************************/
    // Generate adaptive chunks from the sorted curve
    /************************************************/

    if (verbose)
        start_timer("Encoding chunks...\n");
    size_t adder = 1;
    uint8_t q = 0;
    size_t left = 0;
    size_t old_right = left;	// Inclusive bound
    size_t right = left;	// Exclusive bound
    vector<uint8_t> chunk_ids(size, 0);
    uint8_t chunk_num = 1;
    vector< vector<uint8_t> > Us_q(n);
    for (uint8_t i = 0; i < n; ++i)
        Us_q[i] = vector<uint8_t> (s[i], 0);

    while (left < size) {
        while (left < size and q < 63) {
            right = min(size, old_right + adder);
            double chunk_min = sorting[left].first;
            double chunk_max = sorting[right - 1].first;
            double sse = 0;
            if (right > left + 1) {
                if (q > 0) {
                    // Compute the quantization error. It could also be approximated  as follows:
                    // double k = (chunk_max - chunk_min)/double(1<<q); // Quantization resolution
                    // mse = k*k/12; // Expected L2 error: $(\int_{-k/2}^{k/2} x^2 dx)/k$
                    // But that tends to underestimate the error
                    for (size_t i = left; i < right; ++i) {
                        uint64_t quant = roundl((sorting[i].first - chunk_min) * ((1UL << q) - 1.) / (chunk_max - chunk_min));
                        double dequant = quant * (chunk_max - chunk_min) / ((1UL << q) - 1.) + chunk_min;
                        sse += (sorting[i].first - dequant) * (sorting[i].first - dequant);
                    }
                } else {
                    for (size_t i = left; i < right; ++i)
                        sse += (sorting[i].first - chunk_min) * (sorting[i].first - chunk_min);
                }
            }
            double mse = sse / (right - left);
            if (debug)
                cout << "\t\tWe try [" << left << "," << right << "), adder = " << adder << ", mse = " << mse << endl;
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

        size_t chunk_size = (right - left);
        double chunk_min = sorting[left].first;
        double chunk_max = sorting[right - 1].first;

        /********************************************/
        // Quantize (in-place) the core elements
        /********************************************/

        // If q = 0 there's no need to store anything quantized, not even the sign
        // If q = 63, values are kept as they are (doubles) and we forget about quantization
        if (q > 0 and q < 63) {
            #pragma omp parallel for
            for (size_t i = left; i < right; ++i) {
                size_t to_write = 0;
                if (chunk_size > 1)
                    // The following min() prevents overflowing the q-bit representation when converting double -> long int
                    to_write = min(((1UL << q) - 1), (uint64_t)
                                   roundl((sorting[i].first - chunk_min) / (chunk_max - chunk_min) * ((1UL << q) - 1)));
                if (c[sorting[i].second] < 0)
                    to_write |= 1UL << q;
                memcpy(&c[sorting[i].second], (void*)&to_write, sizeof(to_write));
            }
        }

        /********************************************/
        // Save mask and compute RLE+Huffman out of it
        /********************************************/

        #pragma omp parallel for
        for (size_t i = left; i < right; ++i) {
            size_t index = sorting[i].second;
            chunk_ids[index] = chunk_num;

            // We use this loop also to update the needed quantization bits per factor column
            for (uint8_t i = 0; i < n; ++i) {
                size_t coord = index % sprod[i+1] / sprod[i];
                Us_q[i][coord] = max(Us_q[i][coord], q);
            }
        }

        vector<size_t> rle;

        // RLE
        bool current_bit = false;
        bool last_bit = false;
        size_t counter = 0;
        for (size_t i = 0; i < size; ++i) {
            if (chunk_ids[i] == 0)
                current_bit = false;
            else if (chunk_ids[i] == chunk_num)
                current_bit = true;
            else
                continue;
            if (current_bit == last_bit)
                counter++;
            else {
                rle.push_back(counter);
                counter = 1;
                last_bit = current_bit;
            }
        }
        rle.push_back(counter);

        zlib_write_stream(reinterpret_cast<unsigned char*> (&chunk_min), sizeof(chunk_min));
        zlib_write_stream(reinterpret_cast<unsigned char*> (&chunk_max), sizeof(chunk_max));
        encode(rle);

        if (verbose) {
            size_t quant_bits = 0;
            if (q > 0)
                quant_bits = (q + 1) * (right - left);	// The "+1" is for the sign
            cout << "\tEncoded chunk " << int(chunk_num) << " (q=" << int(q) << "), min=" << chunk_min << ", max=" << chunk_max << ", quant_bits=" << quant_bits << ", bits=[" << left << "," << right << "), size=" << right - left << endl << flush;
        }

        // Update control variables
        q++;
        left = right;
        old_right = left;
        chunk_num++;
    }
    if (verbose)
        stop_timer();

    /*******************************/
    // Compute and save tensor ranks
    /*******************************/

    r = vector<uint32_t> (n);
    rprod = vector<size_t> (n+1);
    rprod[0] = 1;
    for (uint8_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < s[i]; ++j)
            if (Us_q[i][j])
                r[i] = j+1;
        rprod[i+1] = rprod[i]*r[i];
    }
    if (verbose) {
        cout << "Compressed tensor ranks:";
        for (uint8_t i = 0; i < n; ++i)
            cout << " " << r[i];
        cout << endl;
    }
    zlib_write_stream(reinterpret_cast<unsigned char*> (&r[0]), n*sizeof(r[0]));

    /*********************************/
    // Encode and save factor matrices
    /*********************************/

    if (debug) {
        cout << "q's for the factor columns: " << endl;
        for (uint8_t i = 0; i < n; ++i) {
            for (uint32_t j = 0; j < s[i]; ++j)
                cout << " " << int(Us_q[i][j]);
            cout << endl;
        }
    }
    if (verbose)
        start_timer("Encoding factor matrices... ");
    for (uint8_t i = 0; i < n; ++i)
        encode_factor(Us[i].leftCols(r[i]), Us_q[i]);
    if (verbose)
        stop_timer();

    /************************/
    // Save the core encoding
    /************************/

    if (verbose)
        start_timer("Saving core quantization... ");
    zlib_open_wbit();
    for (size_t i = 0; i < size; ++i) {
        uint8_t q = chunk_ids[i]-1;
        if (q > 0)
            zlib_write_bit(*reinterpret_cast<uint64_t*> (&c[i]), q+1);
    }
    zlib_close_wbit();
    if (verbose)
        stop_timer();
    delete[] c;
    close_zlib_write();

    /*******************************************************************/
    // Compute and display statistics of the resulting compression ratio
    /*******************************************************************/

    size_t newbits = zs.total_written_bytes * 8;
    cout << "oldbits = " << size * io_type_size * 8L << ", newbits = " << newbits << ", compressionratio = " << size * io_type_size * 8L / double (newbits)
         << ", bpv = " << newbits / double (size) << endl << flush;
    return data;
}

#endif // COMPRESS_HPP
