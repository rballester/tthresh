#ifndef __DECOMPRESS_HPP__
#define __DECOMPRESS_HPP__

#include "tthresh.hpp"
#include "tucker.hpp"
#include "zlib_io.hpp"
#include "decode.hpp"
#include "Slice.hpp"
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

void decode_factor(MatrixXd & U, uint32_t n_columns) {

    U = MatrixXd(n_columns, n_columns);

    // First, the matrix's maximum, used for quantization
    double maximum;
    zlib_read_stream(reinterpret_cast<uint8_t*> (&maximum), sizeof(maximum));

    // Next, the q for each column
    vector<uint8_t> U_q(n_columns);
    for (uint32_t i = 0; i < n_columns; ++i)
        zlib_read_stream(reinterpret_cast<uint8_t*> (&U_q[i]), sizeof(U_q[i]));

    // Finally we can dequantize the matrix
    zlib_open_rbit();
    for (uint32_t j = 0; j < n_columns; ++j) {
        for (uint32_t i = 0; i < n_columns; ++i) {
            uint8_t q = U_q[j];
            if (q > 0) {
                q = min(63, q + 2);
                uint64_t quant = zlib_read_bits(q+1);
                if (q == 63) // The matrix value is read verbatim as a double and we get 0 error
                    memcpy(&U(i, j), (void*)&quant, sizeof(quant));
                else { // We dequantize this matrix value
                    uint8_t sign = (quant >> q) & 1UL; // Read the sign bit
                    quant &= ~(1UL << q); // Put the sign bit to zero
                    U(i, j) = -(2 * sign - 1) / ((1UL << q) - double (1)) *maximum * double (quant);
                }
            }
            else
                U(i, j) = 0;
        }
    }
}

void decompress(string compressed_file, string output_file, double *data, vector<Slice>& cutout, bool verbose, bool debug) {

    // Read output tensor dimensionality, sizes and type
    open_zlib_read(compressed_file);
    uint8_t n;
    zlib_read_stream(reinterpret_cast<uint8_t*> (&n), sizeof(n));
    s = vector<uint32_t> (n);
    zlib_read_stream(reinterpret_cast<uint8_t*> (&s[0]), n * sizeof(s[0]));

    bool whole_reconstruction = cutout.size() == 0;
    if (cutout.size() < n) // Non-specified slicings are assumed to be the standard (0,1,-1)
        for (uint32_t j = cutout.size(); j < s.size(); ++j)
            cutout.push_back(Slice(0, -1, 1));

    snew = vector<uint32_t> (n);
    sprod = vector<size_t> (n+1); // Cumulative size products. The i-th element contains s[0]*...*s[i-1]
    sprod[0] = 1;
    snewprod = vector<size_t> (n+1); // Cumulative size products. The i-th element contains snew[0]*...*snew[i-1]
    snewprod[0] = 1;
    for (uint8_t i = 0; i < n; ++i) {
        sprod[i+1] = sprod[i]*s[i];
        cutout[i].update(s[i]);
        snew[i] = cutout[i].get_size();
        snewprod[i+1] = snewprod[i]*snew[i];
    }
    size_t size = sprod[n];

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
    zlib_read_stream(reinterpret_cast<uint8_t*> (&io_type_code), sizeof(io_type_code));
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
    vector<double> minimums;
    vector<double> maximums;
    vector<uint8_t> chunk_ids(size, 0);
    uint8_t chunk_num = 1;
    size_t assigned = 0;
    vector<size_t> jumps(size+1, 0);
    jumps[0] = size+1;
    if (verbose)
        start_timer("Decoding chunks...\n");
    while(assigned < size) {
        double chunk_min;
        zlib_read_stream(reinterpret_cast<uint8_t*> (&chunk_min), sizeof(chunk_min));
        minimums.push_back(chunk_min);
        double chunk_max;
        zlib_read_stream(reinterpret_cast<uint8_t*> (&chunk_max), sizeof(chunk_max));
        maximums.push_back(chunk_max);

        // At each step, we integrate a new mask. Each mask indexes only over the "zeros" of the previous mask
        // The new mask always comes in RLE form
        // The current mask is described by the "jumps" array: jumps[0] is the first RLE counter.
        // If jumps[i] = n is an RLE counter, then jumps[i+n] coontains the next counter. The values in between can be safely ignored
        vector<size_t> rle;
        decode(rle);
        size_t cur_pos = 0;
        bool jump_bit = false;
        bool cur_bit = false;
        size_t jump_ahead = jumps[cur_pos]; // How many bits are remaining from the "jumps" RLE
        size_t last_zero_start = 0;
        size_t last_one_start = -1;
        rle[0]++; // We assume the mask always starts with a "ghost" 0, to simplify corner cases when the first counter is 0
        for (size_t i = 0; i < rle.size(); ++i) {
            size_t counter = rle[i];
            while (counter > 0) {
                if (jump_bit == false) {
                    size_t step = min(counter, jump_ahead);
                    if (cur_bit == false) { // This is the only case in which
                        if (last_one_start != size_t(-1))
                            jumps[last_one_start] = cur_pos-last_one_start;
                        last_zero_start = cur_pos;
                        jumps[cur_pos] = step;
                        cur_pos += step;
                        if (cur_pos < size+1)
                            last_one_start = cur_pos;
                    }
                    else {
                        for (size_t pos = cur_pos; pos < cur_pos+step; ++pos)
                            chunk_ids[pos-1] = chunk_num;
                        assigned += step;
                        cur_pos += step;
                    }
                    counter -= step;
                    jump_ahead -= step;
                }
                else {
                    cur_pos += jump_ahead;
                    jump_ahead = 0;
                }
                if (jump_ahead == 0 and cur_pos < size+1) {
                    jump_bit = not jump_bit;
                    jump_ahead = jumps[cur_pos];
                }
            }
            if (cur_pos < size+1) // Not yet at the end of the mask
                cur_bit = not cur_bit;
        }
        if (cur_bit == false and jump_bit == false) // We finished with a sequence of 0's
            jumps[last_zero_start] = size+1-last_zero_start;
        else // We finished with a sequence of 1's
            jumps[last_one_start] = size+1-last_one_start;

        if (verbose)
            cout << "\tDecoded chunk " << int(chunk_num) << " (q=" << int (chunk_num - 1) << "), min=" << minimums[minimums.size()-1] << ", max=" << maximums[maximums.size()-1] << endl << flush;
        ++chunk_num;
    }
    if (verbose)
        stop_timer();

    // Read factor matrices
    if (verbose)
        start_timer("Decoding factor matrices... ");
    vector < MatrixXd > Us(n);
    for (uint8_t i = 0; i < n; ++i)
        decode_factor(Us[i], s[i]);
    if (verbose)
        stop_timer();
    
    // Recover the quantized core and dequantize it
    if (verbose)
        start_timer("Dequantizing core... ");
    vector<double> c(size);
    zlib_open_rbit();
    for (size_t i = 0; i < size; ++i) {
        uint8_t q = chunk_ids[i] - 1;
        if (q > 0) {
	    double chunk_min = minimums[q];
	    double chunk_max = maximums[q];
            uint64_t quant = zlib_read_bits(q+1);
            if (q == 63) // The core value is read verbatim as a double and we get 0 error
                memcpy(&c[i], (void*)&quant, sizeof(quant));
            else { // We dequantize this core value
                uint8_t sign = (quant >> q) & 1UL; // Read the sign bit
                quant &= ~(1UL << q); // Put the sign bit to zero
                double dequant;
                dequant = quant / ((1UL << q) - 1.) * (chunk_max - chunk_min) + chunk_min;
                c[i] = -(sign * 2 - 1) * dequant;
            }
        } else
            c[i] = 0;
    }
    close_zlib_read();
    if (verbose)
        stop_timer();

    if (verbose)
        start_timer("Reconstructing tensor...\n");
    hosvd_decompress(c, Us, verbose, cutout);
    if (verbose)
        stop_timer();

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
    for (size_t i = 0; i < snewprod[n]; ++i) {
        if (io_type_code == 0)
            reinterpret_cast < unsigned char *>(&buffer[0])[buffer_wpos] = max(0.0, min(double(std::numeric_limits<unsigned char>::max()), c[i]));
        else if (io_type_code == 1)
            reinterpret_cast < unsigned short *>(&buffer[0])[buffer_wpos] = max(0.0, min(double(std::numeric_limits<unsigned short>::max()), c[i]));
        else if (io_type_code == 2)
            reinterpret_cast < int *>(&buffer[0])[buffer_wpos] = c[i];
        else if (io_type_code == 3)
            reinterpret_cast < float *>(&buffer[0])[buffer_wpos] = c[i];
        else
           reinterpret_cast < double *>(&buffer[0])[buffer_wpos] = c[i];
        buffer_wpos++;
        if (buffer_wpos == buf_elems) {
            buffer_wpos = 0;
            output_stream.write(reinterpret_cast<const char*>(&buffer[0]), io_type_size * buf_elems);
        }
        if (whole_reconstruction and data != NULL) {
            datanorm += data[i] * data[i];
            sse += (data[i] - c[i]) * (data[i] - c[i]);
            datamin = min(datamin, data[i]);
            datamax = max(datamax, data[i]);
        }
    }
    if (buffer_wpos > 0)
        output_stream.write(reinterpret_cast<const char*>(&buffer[0]), io_type_size * buffer_wpos);
    output_stream.close();
    if (verbose)
        stop_timer();

    if (whole_reconstruction and data != NULL) {	// If the uncompressed input is available, we compute the error statistics
        datanorm = sqrt(datanorm);
        double eps = sqrt(sse) / datanorm;
        double rmse = sqrt(sse / size);
        double psnr = 20 * log10((datamax - datamin) / (2 * rmse));
        cout << "eps = " << eps << ", rmse = " << rmse << ", psnr = " << psnr << endl;
    }
}

#endif // DECOMPRESS_HPP
