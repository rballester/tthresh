/*
 * Copyright (c) 2016-2022, Rafael Ballester-Ripoll
 *                          Peter Lindstrom
 *
 * Licensed under the LGPLv3.0 (https://github.com/rballester/tthresh/blob/master/LICENSE)
 */

#ifndef __COMPRESS_HPP__
#define __COMPRESS_HPP__

#include <fstream>
#include <vector>
#include "encode.hpp"
#include "tthresh.hpp"
#include "tucker.hpp"
#include "io.hpp"
#include <math.h>
#include <Eigen/Dense>
#include <map>

typedef long double LDOUBLE;


using namespace std;
using namespace Eigen;

int qneeded;

double rle_time = 0;
double raw_time = 0;

double core_price = -1;
int core_nplanes;
size_t total_bits = 0;

vector<uint64_t> encode_array(double* c, size_t size, double sse, int nplanes, bool is_core, bool verbose=false) {

    /******************************************/
    // Find last bit plane to encode losslessly
    /******************************************/

    if (is_core and verbose)
        start_timer("Preliminaries... ");

    double maximum = 0;
    vector<double> g(size);
    double normsq = 0;
    for (size_t i = 0; i < size; ++i) {
        maximum = max(maximum, abs(c[i]));
        g[i] = abs(c[i]);
        normsq += c[i]*c[i];
    }

    int msplane = static_cast<int>(std::floor(std::log2(maximum)));

    int k;
    if (is_core) {
        k = static_cast<int>(std::floor(std::log2(3 * sse / size) / 2)) - 1; // In the end, k will be the last encoded plane
        k = std::max(k, -1023);  // ensure 2^k does not underflow to zero
        k = max(k, msplane-63);
    }
    else
        k = msplane-nplanes+0;
    
    double plane_sse = 0;  // Only for factors: total SSE incurred by truncating the current plane
    vector<double> plane_sses; // Only for factors: SSE incurred by each plane
    long i = 0; // Used to find the breakpoint
    double incurred_sse = 0; // Only for core: total SSE incurred
    

    // loop over bit planes
    double m;
    for (m = std::ldexp(1., k + 1); k < msplane; m *= 2, k++) {
        bool done = false;
        // printf("k=%d m=%g=%a err=%e\n", k, m, m, incurred_sse);
        plane_sse = 0;
        for (i = size-1; i >= 0; i--) {
            // compute signed remainder, ri (truncated bit plane(s) in this step)
            double ri = std::fmod(g[i], m);
            if (ri > 0) {
                // compute error, ei, in current approximation
                double ei = abs(c[i]) - g[i];
                // compute SSE contribution of this truncation step
                double di = ri * (ri + 2 * ei);
                assert(di >= 0);
                plane_sse += di;
                // terminate if target SSE would be exceeded
                if (is_core and incurred_sse + di > sse) {
                    done = true;
                    break;
                }
                // truncate bit plane(s)
                g[i] -= ri;
                // update accumulated error
                incurred_sse += di;
            }
        }
        plane_sses.push_back(plane_sse);
        if (done)
            break;
    }
    size_t last_coef = max(0, i);

    double k_sse = 0;  // SSE avoided by encoding plane k until breakpoint
    for (; i >= 0; i--) {
        double ri = std::fmod(g[i], m);
        if (ri > 0) {
            double ei = abs(c[i]) - g[i];
            double di = ri * (ri + 2 * ei);
            k_sse += di;
        }
    }

    // We want the cumulative plane sse's, from the right (msplane) to the left, to compute the factor price
    vector<double> cumulative_plane_sses(plane_sses);
    for (int i = cumulative_plane_sses.size()-2; i >= 0; --i)
       cumulative_plane_sses[i] += cumulative_plane_sses[i+1];

    int lastq;
    if (is_core) {
        lastq = max(0, 63-(msplane-k));
        core_nplanes = 63-lastq + 1;
    }
    else
        lastq = 0; // For factors, we stop based on price, not k

    /**************/
    // Encode array
    /**************/

    double scale = ldexp(1, 63-ilogb(maximum));

    uint64_t tmp;
    memcpy(&tmp, (void*)&scale, sizeof(scale));
    write_bits(tmp, 64);

    // Vector of quantized core coefficients
    vector<uint64_t> coreq(size);
    for (size_t pos = 0; pos < size; ++pos)
        coreq[pos] = uint64_t(abs(c[pos])*scale);

    vector<uint64_t> current(size, 0);

    if (is_core and verbose)
        stop_timer();
    bool done = false;
    total_bits = 0;
    size_t last_total_bits = total_bits;
    double core_size_delta = 0;
    bool all_raw = false;
    if (verbose)
        start_timer("Encoding core...\n");
    for (int q = 63; q >= lastq; --q) {
        if (verbose and is_core)
            cout << "Encoding core's bit plane p = " << q;
        vector<uint64_t> rle;
        size_t counter = 0;
        vector<bool> raw;
        for (size_t i = 0; i < size; ++i) {
            bool current_bit = (coreq[i]>>q)&1UL;
            if (not all_raw and current[i] == 0) { // Feed to RLE
                if (current_bit) {
                    rle.push_back(counter);
                    counter = 0;
                }
                else
                    counter++;
            }
            else // Feed to raw stream
                raw.push_back(current_bit);

            if (current_bit)
                current[i] |= ((uint64_t)1) << q;
            if (q == lastq and i >= last_coef) {
                done = true;
                if (verbose)
                    cout << " <- breakpoint: coefficient " << i;
                break;
            }
        }
        if (verbose and is_core)
            cout << endl;

        rle.push_back(counter);

        uint64_t rawsize = raw.size();
        write_bits(rawsize, 64);
        total_bits += 64;

        {
            high_resolution_clock::time_point timenow = chrono::high_resolution_clock::now();
            for (size_t i = 0; i < raw.size(); ++i)
                write_bits(raw[i], 1);
            total_bits += raw.size();
            raw_time += std::chrono::duration_cast<std::chrono::microseconds>(chrono::high_resolution_clock::now() - timenow).count()/1000.;
        }
        {
            high_resolution_clock::time_point timenow = chrono::high_resolution_clock::now();
            uint64_t this_part = encode(rle);
            rle_time += std::chrono::duration_cast<std::chrono::microseconds>(chrono::high_resolution_clock::now() - timenow).count()/1000.;
            total_bits += this_part;
        }

        if (last_total_bits > 0) {
            if (is_core)
                core_size_delta = total_bits - last_total_bits;
            // The matrices are done when the price of the current bit plane exceeds the price paid when encoding the core
            else {
                double factor_size_delta = total_bits - last_total_bits;
                double factor_k_sse = plane_sses[plane_sses.size() - 1 - (63-q)];
                double factor_incurred_sse = cumulative_plane_sses[plane_sses.size() - 1 - (63-q)];
                double factor_eps_delta = sqrt(factor_k_sse / (normsq - factor_incurred_sse));
                
                if (factor_size_delta / factor_eps_delta >= core_price/n)
                    done = true;
            }
        }
        last_total_bits = total_bits;

        if (raw.size()/double(size) > 0.8)
            all_raw = true;

        write_bits(all_raw, 1);
        write_bits(done, 1);
        total_bits += 2;

        if (done)
            break;
    }
    if (verbose)
        stop_timer();

    /****************************************/
    // Save signs of significant coefficients
    /****************************************/

    for (size_t i = 0; i < size; ++i) {
        if (current[i] > 0) {
            write_bits((c[i] > 0), 1);
            total_bits++;
        }
    }

    if (is_core) {
        double core_eps_delta = sqrt(k_sse / (normsq - incurred_sse));
        core_price = core_size_delta / core_eps_delta;
    }
    return current;
}

double *compress(string input_file, string compressed_file, string io_type, Target target, double target_value, size_t skip_bytes, bool verbose=false, bool debug=false) {

    n = s.size();
    if (verbose) {
        cout << endl << "/***** Compression: " << to_string(n) << "D tensor of size " << s[0];
        for (uint8_t i = 1; i < n; ++i)
            cout << " x " << s[i];
        cout << " *****/" << endl << endl;
    }
    cumulative_products(s, sprod);

    /***********************/
    // Check input data type
    /***********************/

    size_t size = sprod[n]; // Total number of tensor elements
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

    open_write(compressed_file.c_str());
    write_stream(reinterpret_cast < unsigned char *> (&n), sizeof(n));
    write_stream(reinterpret_cast < unsigned char *> (&s[0]), n*sizeof(s[0]));
    write_stream(reinterpret_cast < unsigned char *> (&io_type_code), sizeof(io_type_code));

    /*****************************/
    // Load input file into memory
    /*****************************/

    if (verbose)
        start_timer("Loading and casting input data... ");
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
    if (verbose)
        stop_timer();
    if (debug) cout << "Input statistics: min = " << datamin << ", max = " << datamax << ", norm = " << datanorm << endl;

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
    double epsilon = sqrt(sse) / datanorm;
    if (verbose) {
        double rmse = sqrt(sse / size);
        double psnr = 20 * log10((datamax - datamin) / (2 * rmse));
        cout << "We target eps = " << epsilon << ", rmse = " << rmse << ", psnr = " << psnr << endl;
    }

    /*********************************/
    // Create and decompose the tensor
    /*********************************/

    if (verbose)
        start_timer("Tucker decomposition...\n");
    double *c = new double[size]; // Tucker core

    memcpy(c, data, size * sizeof(double));

    vector<MatrixXd> Us(n); // Tucker factor matrices
    hosvd_compress(c, Us, verbose);

    if (verbose)
        stop_timer();

    /**************************/
    // Encode and save the core
    /**************************/

    open_wbit();
    vector<uint64_t> current = encode_array(c, size, sse, 0, true, verbose);
    close_wbit();

    /*******************************/
    // Compute and save tensor ranks
    /*******************************/

    if (verbose)
        start_timer("Computing ranks... ");
    r = vector<uint32_t> (n, 0);
    vector<size_t> indices(n, 0);
    vector< RowVectorXd > slicenorms(n);
    for (int dim = 0; dim < n; ++dim) {
        slicenorms[dim] = RowVectorXd(s[dim]);
        slicenorms[dim].setZero();
    }
    for (size_t i = 0; i < size; ++i) {
        if (current[i] > 0)
            for (int dim = 0; dim < n; ++dim)
                slicenorms[dim][indices[dim]] += double(current[i])*current[i];
        indices[0]++;
        int pos = 0;
        while (indices[pos] >= s[pos] and pos < n-1) {
            indices[pos] = 0;
            pos++;
            indices[pos]++;
        }
    }

    for (int dim = 0; dim < n; ++dim) {
        for (size_t i = 0; i < s[dim]; ++i) {
            if (slicenorms[dim][i] > 0)
                r[dim] = i+1;
            slicenorms[dim][i] = sqrt(slicenorms[dim][i]);
        }
    }
    if (verbose)
        stop_timer();

    if (verbose) {
        cout << "Compressed tensor ranks:";
        for (uint8_t i = 0; i < n; ++i)
            cout << " " << r[i];
        cout << endl;
    }
    write_stream(reinterpret_cast<unsigned char*> (&r[0]), n*sizeof(r[0]));

    for (uint8_t i = 0; i < n; ++i)
        write_stream(reinterpret_cast<uint8_t*> (slicenorms[i].data()), r[i]*sizeof(double));

    open_wbit();
    for (int dim = 0; dim < n; ++dim) {
        MatrixXd Uweighted = Us[dim].leftCols(r[dim]);
        for (size_t col = 0; col < r[dim]; ++col)
            Uweighted.col(col) = Uweighted.col(col)*slicenorms[dim][col];
        encode_array(Uweighted.data(), s[dim]*r[dim], 0, core_nplanes, false);  //*(s[i]*s[i]/sprod[n]));
    }
    close_wbit();
    close_write();
    delete[] c;
    size_t newbits = zs.total_written_bytes * 8;
    cout << "oldbits = " << size * io_type_size * 8L << ", newbits = " << newbits << ", compressionratio = " << size * io_type_size * 8L / double (newbits)
<< ", bpv = " << newbits / double (size) << endl << flush;
    return data;
}

#endif // COMPRESS_HPP
