/*
 * Copyright (c) 2016-2017, Rafael Ballester-Ripoll
 *                          (Visualization and MultiMedia Lab, University of Zurich),
 *                          rballester@ifi.uzh.ch
 *
 * Licensed under the LGPLv3.0 (https://github.com/rballester/tthresh/blob/master/LICENSE)
 */

#ifndef __DECOMPRESS_HPP__
#define __DECOMPRESS_HPP__

#include "tthresh.hpp"
#include "tucker.hpp"
#include "io.hpp"
#include "decode.hpp"
#include "Slice.hpp"
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

double maximum;
int q;
size_t pointer;

/////
double decode_rle_time;
double decode_raw_time;
double unscramble_time;
/////

vector<uint64_t> decode_array(size_t size, bool is_core, bool verbose, bool debug) {

    uint64_t tmp = read_bits(64);
    memcpy(&maximum, (void*)&tmp, sizeof(tmp));

    vector<uint64_t> current(size, 0);

    decode_rle_time = 0;
    decode_raw_time = 0;
    unscramble_time = 0;

    int zeros = 0;
    bool all_raw = false;
    if (verbose and is_core)
        start_timer("Decoding core...\n");
    for (q = 63; q >= 0; --q) {
        if (verbose and is_core)
            cout << "Decoding core's bit plane p = " << q << endl;
        uint64_t rawsize = read_bits(64);

        size_t read_from_rle = 0;
        size_t read_from_raw = 0;

        if (all_raw) {
            high_resolution_clock::time_point timenow = chrono::high_resolution_clock::now();
            for (uint64_t pointer = 0; pointer < rawsize; ++pointer) {
                current[pointer] |= read_bits(1) << q;
            }
            unscramble_time += std::chrono::duration_cast<std::chrono::microseconds>(chrono::high_resolution_clock::now() - timenow).count()/1000.;
            vector<size_t> rle;
            decode(rle);
        }
        else {
            vector<bool> raw;
            high_resolution_clock::time_point timenow = chrono::high_resolution_clock::now();
            for (uint64_t i = 0; i < rawsize; ++i)
                raw.push_back(read_bits(1));
            decode_raw_time += std::chrono::duration_cast<std::chrono::microseconds>(chrono::high_resolution_clock::now() - timenow).count()/1000.;

            vector<size_t> rle;
            timenow = chrono::high_resolution_clock::now();
            decode(rle);
            decode_rle_time += std::chrono::duration_cast<std::chrono::microseconds>(chrono::high_resolution_clock::now() - timenow).count()/1000.;

            int64_t raw_index = 0;
            int64_t rle_value = -1;
            int64_t rle_index = -1;

            timenow = chrono::high_resolution_clock::now();
            for (pointer = 0; pointer < size; ++pointer) {
                uint64_t this_bit = 0;
                if (not all_raw and current[pointer] == 0) { // Consume bit from RLE
                    if (rle_value == -1) {
                        rle_index++;
                        if (rle_index == int64_t(rle.size()))
                            break;
                        rle_value = rle[rle_index];
                    }
                    if (rle_value >= 1) {
                        read_from_rle++;
                        this_bit = 0;
                        rle_value--;
                    }
                    else if (rle_value == 0) {
                        read_from_rle++;
                        this_bit = 1;
                        rle_index++;
                        if (rle_index == int64_t(rle.size()))
                            break;
                        rle_value = rle[rle_index];
                    }
                }
                else { // Consume bit from raw
                    if (raw_index == int64_t(raw.size()))
                        break;
                    this_bit = raw[raw_index];
                    read_from_raw++;
                    raw_index++;
                }
                if (this_bit)
                    current[pointer] |= this_bit << q;
            }
            unscramble_time += std::chrono::duration_cast<std::chrono::microseconds>(chrono::high_resolution_clock::now() - timenow).count()/1000.;
        }

        all_raw = read_bits(1);

        bool done = read_bits(1);
        if (done)
            break;
        else
            zeros++;
    }
    if (debug)
        cout << "decode_rle_time=" << decode_rle_time << ", decode_raw_time=" << decode_raw_time << ", unscramble_time=" << unscramble_time << endl;
    if (verbose and is_core)
        stop_timer();
    return current;
}

vector<double> dequantize(vector<uint64_t>& current) { // TODO after resize
    size_t size = current.size();
    vector<double> c(size, 0);
    for (size_t i = 0; i < size; ++i) {
        if (current[i] > 0) {
            if (i < pointer) {
                if (q >= 1)
                    current[i] += 1UL<<(q-1);
            }
            else
                current[i] += 1UL<<q;
            char sign = read_bits(1);
            c[i] = double(current[i]) / maximum * (sign*2-1);
        }
    }
    return c;
}

void decompress(string compressed_file, string output_file, double *data, vector<Slice>& cutout, bool autocrop, bool verbose, bool debug) {

    /***************************************************/
    // Read output tensor dimensionality, sizes and type
    /***************************************************/

    open_read(compressed_file);
    read_stream(reinterpret_cast<uint8_t*> (&n), sizeof(n));
    s = vector<uint32_t> (n);
    read_stream(reinterpret_cast<uint8_t*> (&s[0]), n * sizeof(s[0]));

    bool whole_reconstruction = cutout.size() == 0;
    if (cutout.size() < n) // Non-specified slicings are assumed to be the standard (0,1,-1)
        for (uint32_t j = cutout.size(); j < s.size(); ++j)
            cutout.push_back(Slice(0, -1, 1));

    cumulative_products(s, sprod);
    size_t size = sprod[n];
    snew = vector<uint32_t> (n);
    for (uint8_t i = 0; i < n; ++i) {
        cutout[i].update(s[i]);
        snew[i] = cutout[i].get_size();
    }
    cumulative_products(snew, snewprod);

    if (verbose) {
        cout << endl << "/***** Decompression: " << to_string(n) << "D tensor of size ";
        if (not whole_reconstruction) {
            cout << snew[0];
            for (uint8_t i = 1; i < n; ++i)
                cout << " x " << snew[i];
            cout << " (originally ";
        }
        cout << s[0];
        for (uint8_t i = 1; i < n; ++i)
            cout << " x " << s[i];
        if (not whole_reconstruction)
            cout << ")";

        cout << " *****/" << endl << endl;
    }

    uint8_t io_type_code;
    read_stream(reinterpret_cast<uint8_t*> (&io_type_code), sizeof(io_type_code));
    uint8_t io_type_size;
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

    /*************/
    // Decode core
    /*************/

    vector<uint64_t> current = decode_array(sprod[n], true, verbose, debug);
    vector<double> c = dequantize(current);
    close_rbit();

    /*******************/
    // Read tensor ranks
    /*******************/

    r = vector<uint32_t> (n);
    read_stream(reinterpret_cast<uint8_t*> (&r[0]), n*sizeof(r[0]));
    rprod = vector<size_t> (n+1);
    rprod[0] = 1;
    for (uint8_t i = 0; i < n; ++i)
        rprod[i+1] = rprod[i]*r[i];
    if (verbose) {
        cout << "Compressed tensor ranks:";
        for (uint8_t i = 0; i < n; ++i)
            cout << " " << r[i];
        cout << endl;
    }

    vector<RowVectorXd> slicenorms(n);
    for (uint8_t i = 0; i < n; ++i) {
        slicenorms[i] = RowVectorXd(r[i]);
        for (uint64_t col = 0; col < r[i]; ++col) { // TODO faster
            double norm;
            read_stream(reinterpret_cast<uint8_t*> (&norm), sizeof(double));
            slicenorms[i][col] = norm;
        }
    }

    //**********************/
    // Reshape core in place
    //**********************/

    size_t index = 0; // Where to read from in the original core
    vector<size_t> indices(n, 0);
    uint8_t pos = 0;
    for (size_t i = 0; i < rprod[n]; ++i) { // i marks where to write in the new rank-reduced core
        c[i] = c[index];
        indices[0]++;
        index++;
        pos = 0;
        // We update all necessary indices in cascade, left to right. pos == n-1 => i == rprod[n]-1 => we are done
        while (indices[pos] >= r[pos] and pos < n-1) {
            indices[pos] = 0;
            index += sprod[pos+1] - r[pos]*sprod[pos];
            pos++;
            indices[pos]++;
        }
    }

    //*****************/
    // Reweight factors
    //*****************/

    vector< MatrixXd > Us;
    for (uint8_t i = 0; i < n; ++i) {
        vector<uint64_t> factorq = decode_array(s[i]*r[i], false, verbose, debug);
        vector<double> factor = dequantize(factorq);
        MatrixXd Uweighted(s[i], r[i]);
        memcpy(Uweighted.data(), (void*)factor.data(), sizeof(double)*s[i]*r[i]);
        MatrixXd U(s[i], r[i]);
        for (size_t col = 0; col < r[i]; ++col) {
            if (slicenorms[i][col] > 1e-10)
                U.col(col) = Uweighted.col(col)/slicenorms[i][col];
            else
                U.col(col) *= 0;
        }
        Us.push_back(U);
    }
    close_rbit();

    /*************************/
    // Autocrop (if requested)
    /*************************/

    if (autocrop) {
        cout << "autocrop =";
        for (uint8_t dim = 0; dim < n; ++dim) {
            uint32_t start_row = 0, end_row = 0;
            bool start_set = false;
            for (int i = 0; i < Us[dim].rows(); ++i) {
                double sqnorm = 0;
                for (int j = 0; j < Us[dim].cols(); ++j)
                    sqnorm += Us[dim](i,j)*Us[dim](i,j);
                if (sqnorm > AUTOCROP_THRESHOLD) {
                    if (not start_set) {
                        start_row = i;
                        start_set = true;
                    }
                    end_row = i+1;
                }
            }
            cutout[dim].points[0] = start_row;
            cutout[dim].points[1] = end_row;
            snew[dim] = end_row-start_row;
            cout << " " << start_row << ":" << end_row;
        }
        cout << endl;
        cumulative_products(snew, snewprod);
    }

    /************************/
    // Reconstruct the tensor
    /************************/

    if (verbose)
        start_timer("Reconstructing tensor...\n");
    hosvd_decompress(c, Us, verbose, cutout);
    if (verbose)
        stop_timer();

    /***********************************/
    // Cast and write the result on disk
    /***********************************/

    if (verbose)
        start_timer("Casting and saving final result... ");
    ofstream output_stream(output_file.c_str(), ios::out | ios::binary);
    size_t buf_elems = CHUNK;
    vector<uint8_t> buffer(io_type_size * buf_elems);
    size_t buffer_wpos = 0;
    double sse = 0;
    double datanorm = 0;
    double datamin = std::numeric_limits < double >::max();
    double datamax = std::numeric_limits < double >::min();
    double remapped = 0;
    for (size_t i = 0; i < snewprod[n]; ++i) {
        if (io_type_code == 0) {
            remapped = (unsigned char)(round(max(0.0, min(double(std::numeric_limits<unsigned char>::max()), c[i]))));
            reinterpret_cast < unsigned char *>(&buffer[0])[buffer_wpos] = remapped;
        }
        else if (io_type_code == 1) {
            remapped = (unsigned short)(round(max(0.0, min(double(std::numeric_limits<unsigned short>::max()), c[i]))));
            reinterpret_cast < unsigned short *>(&buffer[0])[buffer_wpos] = remapped;
        }
        else if (io_type_code == 2) {
            remapped = int(round(max(std::numeric_limits<int>::min(), min(double(std::numeric_limits<int>::max()), c[i]))));;
            reinterpret_cast < int *>(&buffer[0])[buffer_wpos] = remapped;
        }
        else if (io_type_code == 3) {
            remapped = float(c[i]);
            reinterpret_cast < float *>(&buffer[0])[buffer_wpos] = remapped;
        }
        else {
           remapped = c[i];
           reinterpret_cast < double *>(&buffer[0])[buffer_wpos] = remapped;
        }
        buffer_wpos++;
        if (buffer_wpos == buf_elems) {
            buffer_wpos = 0;
            output_stream.write(reinterpret_cast<const char*>(&buffer[0]), io_type_size * buf_elems);
        }
        if (whole_reconstruction and not autocrop and data != NULL) { // If needed, we compute the error statistics
            datanorm += data[i] * data[i];
            sse += (data[i] - remapped) * (data[i] - remapped);
            datamin = min(datamin, data[i]);
            datamax = max(datamax, data[i]);
        }
    }
    if (buffer_wpos > 0)
        output_stream.write(reinterpret_cast<const char*>(&buffer[0]), io_type_size * buffer_wpos);
    output_stream.close();
    if (verbose)
        stop_timer();

    if (whole_reconstruction and not autocrop and data != NULL) {
        datanorm = sqrt(datanorm);
        double eps = sqrt(sse) / datanorm;
        double rmse = sqrt(sse / size);
        double psnr = 20 * log10((datamax - datamin) / (2 * rmse));
        cout << "eps = " << eps << ", rmse = " << rmse << ", psnr = " << psnr << endl;
    }
}

#endif // DECOMPRESS_HPP
