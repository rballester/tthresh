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
    zlib_read_stream(reinterpret_cast<unsigned char*> (&maximum), sizeof(maximum));

    // Next, the q for each column
    vector < char >U_q(n_columns);
    for (int i = 0; i < n_columns; ++i)
        zlib_read_stream(reinterpret_cast<unsigned char*> (&U_q[i]), sizeof(U_q[i]));

    // Finally we can dequantize the matrix
    zlib_open_rbit();
    for (int j = 0; j < n_columns; ++j) {
        for (int i = 0; i < n_columns; ++i) {
            char q = U_q[j];
            if (q > 0) {
                q = min(63, q + 2);
                unsigned long int quant = zlib_read_bits(q+1);
                if (q == 63) // The matrix value is read verbatim as a double and we get 0 error
                    memcpy(&U(i, j), (void*)&quant, sizeof(quant));
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

    // Read output tensor dimensionality, sizes and type
    open_zlib_read(compressed_file);
    char n;
    zlib_read_stream(reinterpret_cast < unsigned char *>(&n), sizeof(n));
    vector < int >s(n);
    zlib_read_stream(reinterpret_cast < unsigned char *>(&s[0]), n * sizeof(s[0]));
    ind_t size = 1;
    for (char i = 0; i < n; ++i)
        size *= s[i];
    cumulative_size_products(s, n);

    if (verbose) {
        cout << endl << "/***** Decompression: " << to_string(n) << "D tensor of size " << s[0];
        for (char i = 1; i < n; ++i)
            cout << " x " << s[i];
        cout << " *****/" << endl << endl;
    }

    char io_type_code;
    zlib_read_stream(reinterpret_cast < unsigned char *>(&io_type_code), sizeof(io_type_code));
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
    vector<ind_t> jumps(size+1, 0);
    jumps[0] = size+1;
    if (verbose)
        start_timer("Decoding chunks...\n");
    while(assigned < size) {
        double chunk_min;
        zlib_read_stream(reinterpret_cast < unsigned char *>(&chunk_min), sizeof(chunk_min));
        minimums.push_back(chunk_min);
        double chunk_max;
        zlib_read_stream(reinterpret_cast < unsigned char *>(&chunk_max), sizeof(chunk_max));
        maximums.push_back(chunk_max);

        // At each step, we integrate a new mask. Each mask indexes only over the "zeros" of the previous mask
        // The new mask always comes in RLE form
        // The current mask is described by the "jumps" array: jumps[0] is the first RLE counter.
        // If jumps[i] = n is an RLE counter, then jumps[i+n] coontains the next counter. The values in between can be safely ignored
        vector<ind_t> rle;
        decode(rle);
        ind_t cur_pos = 0;
        bool jump_bit = false;
        bool cur_bit = false;
        ind_t jump_ahead = jumps[cur_pos]; // How many bits are remaining from the "jumps" RLE
        ind_t last_zero_start = 0;
        ind_t last_one_start = -1;
        rle[0]++; // We assume the mask always starts with a "ghost" 0, to simplify corner cases when the first counter is 0
        for (unsigned long int i = 0; i < rle.size(); ++i) {
            ind_t counter = rle[i];
            while (counter > 0) {
                if (jump_bit == false) {
                    if (cur_bit == false) { // This is the only case in which
                        if (last_one_start > -1)
                            jumps[last_one_start] = cur_pos-last_one_start;
                        last_zero_start = cur_pos;
                        ind_t step = min(counter, jump_ahead);
                        counter -= step;
                        jumps[cur_pos] = step;
                        cur_pos += step;
                        jump_ahead -= step;
                        if (cur_pos < size+1)
                            last_one_start = cur_pos;
                    }
                    else {
                        ind_t step = min(counter, jump_ahead);
                        for (ind_t pos = cur_pos; pos < cur_pos+step; ++pos)
                            chunk_ids[pos-1] = chunk_num;
                        assigned += step;
                        counter -= step;
                        cur_pos += step;
                        jump_ahead -= step;
                    }
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
    for (char i = 0; i < n; ++i)
        decode_factor(Us[i], s[i]);
    if (verbose)
        stop_timer();
    
    // Recover the quantized core and dequantize it
    if (verbose)
        start_timer("Dequantizing core... ");
    double *c = new double[size];
    zlib_open_rbit();
    for (int i = 0; i < size; ++i) {
        char q = chunk_ids[i] - 1;
        if (q > 0) {
	    double chunk_min = minimums[q];
	    double chunk_max = maximums[q];
            unsigned long int quant = zlib_read_bits(q+1);
            if (q == 63) // The core value is read verbatim as a double and we get 0 error
                memcpy(&c[i], (void*)&quant, sizeof(quant));
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
    close_zlib_read();
    if (verbose)
        stop_timer();

    if (verbose)
        start_timer("Reconstructing tensor...\n");
    hosvd(c, s, Us, false, verbose);
    if (verbose)
        stop_timer();

    if (verbose)
        start_timer("Casting and saving final result... ");
    ofstream output_stream(output_file.c_str(), ios::out | ios::binary);
    unsigned long int buf_elems = CHUNK;
    char *buffer = new char[io_type_size * buf_elems];
    unsigned long int buffer_wpos = 0;
    double sse = 0;
    double datanorm = 0;
    double datamin = std::numeric_limits < double >::max();
    double datamax = std::numeric_limits < double >::min();
    for (long int i = 0; i < size; ++i) {
        if (io_type_code == 0)
            reinterpret_cast < unsigned char *>(buffer)[buffer_wpos] = max(0.0, min(double(std::numeric_limits<unsigned char>::max()), c[i]));
        else if (io_type_code == 1)
            reinterpret_cast < unsigned short *>(buffer)[buffer_wpos] = max(0.0, min(double(std::numeric_limits<unsigned short>::max()), c[i]));
        else if (io_type_code == 2)
            reinterpret_cast < int *>(buffer)[buffer_wpos] = c[i];
        else if (io_type_code == 3)
            reinterpret_cast < float *>(buffer)[buffer_wpos] = c[i];
        else
           reinterpret_cast < double *>(buffer)[buffer_wpos] = c[i];
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
    if (verbose)
        stop_timer();

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
