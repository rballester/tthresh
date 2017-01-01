#include "tthresh.hpp"
#include "tucker.hpp"
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

void decode_factor(MatrixXd& U, int s, string file1, string file2, string file3) {

    // First, the q for each column
    vector<char> columns_q(s);
    ifstream columns_q_stream(file1.c_str(), ios::in | ios::binary);
    for (int i = 0; i < s; ++i)
        columns_q_stream.read(&columns_q[i],sizeof(char));
    columns_q_stream.close();

    // Next, the matrix's maximum, used for quantization
    ifstream limits_stream(file2.c_str(), ios::in | ios::binary);
    double maximum;
    limits_stream.read(reinterpret_cast<char*>(&maximum),sizeof(double));
    limits_stream.close();

    // Finally we dequantize the matrix itself
    ifstream matrix_stream(file3.c_str(), ios::in | ios::binary);
    char matrix_rbyte;
    matrix_stream.read(&matrix_rbyte,sizeof(char));
    char matrix_rbit = 7;
    for (int j = 0; j < s; ++j) {
        for (int i = 0; i < s; ++i) {
            char q = columns_q[j];
            if (q == 0)
                U(i,j) = 0;
            else {
                q = min(63,q+2);
                unsigned long int to_read = 0;
                for (int j = q; j >= 0; --j) {
                    to_read |= ((matrix_rbyte>>matrix_rbit)&1UL) << j;
                    matrix_rbit--;
                    if (matrix_rbit < 0) {
                        matrix_rbit = 7;
                        matrix_stream.read(&matrix_rbyte,sizeof(char));
                    }
                }
                char sign = (to_read>>q)&1UL;
                to_read &= ~(1UL<<q);
                U(i,j) = -(2*sign-1)/((1UL<<q)-double(1))*maximum*double(to_read);
            }
        }
    }
    matrix_stream.close();
}

void decompress(string compressed_file, string output_file, double* data, bool verbose, bool debug) {

    if (verbose) cout << endl << "/***** Decompression *****/" << endl << endl << flush;

    int _ = system("rm tthresh-tmp/*");

    stringstream ss;
    ss << "tar -xf " << compressed_file;
    string command(ss.str());
    _ = system(command.c_str());

    // Read output tensor type
    ifstream all_stream("tthresh-tmp/all", ios::in | ios::binary );
    int s[3];
    all_stream.read(reinterpret_cast<char*>(s),3*sizeof(int));
    long int size = s[0]*s[1]*s[2];
    
    char io_type_code;
    all_stream.read(reinterpret_cast<char*>(&io_type_code),sizeof(char));
    char io_type_size;
    if (io_type_code == 0) io_type_size = 1;
    else if (io_type_code == 1) io_type_size = 4;
    else if (io_type_code == 2) io_type_size = 4;
    else io_type_size = 8;
    unsigned char n_chunks;
    all_stream.read(reinterpret_cast<char*>(&n_chunks),sizeof(char));
    all_stream.close();

    //cout << "****" << endl;
    //cout << s[0] << " " << s[1] << " " << s[2] << endl;
    //cout << int(io_type_size) << endl;
    // Read tensor sizes
    //ifstream sizes_stream("tthresh-tmp/sizes", ios::in | ios::binary );
    //sizes_stream.close();

    // Read chunk encodings and condense them into one array
    //ifstream chunk_sizes_stream("tthresh-tmp/chunk_sizes", ios::in | ios::binary);
    //streampos fsize = chunk_sizes_stream.tellg();
    //chunk_sizes_stream.seekg( 0, ios::end );
    //fsize = chunk_sizes_stream.tellg() - fsize;
    //chunk_sizes_stream.seekg( 0, ios::beg );
    //int n_chunks = fsize/4;
    
    ifstream chunk_info_stream("tthresh-tmp/chunk_info", ios::in | ios::binary);
    double minimums[n_chunks];
    double maximums[n_chunks];

    vector<int> encoding_mask(size,0);
    for (int chunk_num = 1; chunk_num <= n_chunks; ++chunk_num) {

        chunk_info ci;
        chunk_info_stream.read(reinterpret_cast<char*>(&ci),sizeof(chunk_info));
	minimums[chunk_num-1] = ci.minimum;
	maximums[chunk_num-1] = ci.maximum;

        stringstream ss;
        ss << "tthresh-tmp/mask_" << setw(4) << setfill('0') << chunk_num << ".compressed";
        decode(ss.str(),"tthresh-tmp/mask.decompressed");
        ifstream mask("tthresh-tmp/mask.decompressed", ios::in | ios::binary);
        std::vector<char> buffer((istreambuf_iterator<char>(mask)), istreambuf_iterator<char>());
        long int ind = 0;
        for (unsigned int i = 0; i < buffer.size() and ind < size; ++i) { // This can be more efficient...
            for (char chunk_rbit = 7; chunk_rbit >= 0 and ind < size; --chunk_rbit) {
                while (encoding_mask[ind] > 0)
                    ind++;
                if ((buffer[i] >> chunk_rbit)&1) {
                    encoding_mask[ind] = chunk_num;
                }
                ind++;
            }
        }
        if (verbose) cout << "Decoded chunk " << chunk_num << ", mask has " << buffer.size()*8 << " bits, q="
                          << int(chunk_num-1) << endl << flush;
    }
    //chunk_sizes_stream.close();

    // Read minimums
//    double minimums[n_chunks];
    //ifstream minimums_stream("tthresh-tmp/minimums", ios::in | ios::binary);
    //minimums_stream.read(reinterpret_cast<char*>(minimums),n_chunks*sizeof(double));
    //minimums_stream.close();

    // Read maximums
    //double maximums[n_chunks];
    //ifstream maximums_stream("tthresh-tmp/maximums", ios::in | ios::binary);
    //maximums_stream.read(reinterpret_cast<char*>(maximums),n_chunks*sizeof(double));
    //maximums_stream.close();

    // Recover the core
    double* c = new double[size];
    ifstream core_quant_stream("tthresh-tmp/core_quant", ios::in | ios::binary);
    std::vector<char> core_quant_buffer((istreambuf_iterator<char>(core_quant_stream)), istreambuf_iterator<char>());
    core_quant_stream.close();
    int core_quant_rbyte = 0;
    char core_quant_rbit = 7;
    for (int i = 0; i < size; ++i) {
        int chunk_num = encoding_mask[i];
        double chunk_min = minimums[chunk_num-1];
        double chunk_max = maximums[chunk_num-1];
        char q = chunk_num-1;

        if (q > 0) {
            char sign = 0;
            unsigned long int quant = 0;
            for (long int j = q; j >= 0; --j) {
                if (j == q and q < 63)
                    sign = ((core_quant_buffer[core_quant_rbyte] >> core_quant_rbit)&1UL);
                else
                    quant |= ((core_quant_buffer[core_quant_rbyte] >> core_quant_rbit)&1UL) << j;
                core_quant_rbit--;
                if (core_quant_rbit < 0) {
                    core_quant_rbyte++;
                    core_quant_rbit = 7;
                }
            }
            if (q == 63)
                c[i] = static_cast<double>(quant);
            else {
                double dequant;
                dequant = quant/((1UL<<q)-1.)*(chunk_max-chunk_min) + chunk_min;
                c[i] = -(sign*2-1)*dequant;
            }
        }
        else
            c[i] = 0;
    }

    // Read factor matrices
    if (verbose) cout << "Decoding factor matrices... " << flush;
    MatrixXd U1(s[0],s[0]), U2(s[1],s[1]), U3(s[2],s[2]);
    decode_factor(U1,s[0],"tthresh-tmp/U1_q","tthresh-tmp/U1_limits","tthresh-tmp/U1");
    decode_factor(U2,s[1],"tthresh-tmp/U2_q","tthresh-tmp/U2_limits","tthresh-tmp/U2");
    decode_factor(U3,s[2],"tthresh-tmp/U3_q","tthresh-tmp/U3_limits","tthresh-tmp/U3");
    if (verbose) cout << "Done" << endl << flush;

    if (verbose) cout << "Reconstructing tensor... " << flush;
    double* r = c;
    tucker(r,s,U1,U2,U3,false);
    if (verbose) cout << "Done" << endl << flush;

    ofstream output_stream(output_file.c_str(), ios::out | ios::binary);
    long int buf_elems = 1<<20;
    char* buffer = new char[io_type_size*buf_elems];
    long int buffer_wpos = 0;
    double sse = 0;
    double input_norm = 0;
    double input_min = std::numeric_limits<double>::max();
    double input_max = std::numeric_limits<double>::min();
    for (long int i = 0; i < size; ++i) {
        if (io_type_code == 0)
            reinterpret_cast<unsigned char*>(buffer)[buffer_wpos] = abs(r[i]);
        else if (io_type_code == 1)
            reinterpret_cast<int*>(buffer)[buffer_wpos] = r[i];
        else if (io_type_code == 2)
            reinterpret_cast<float*>(buffer)[buffer_wpos] = r[i];
        else
            reinterpret_cast<double*>(buffer)[buffer_wpos] = r[i];
        buffer_wpos++;
        if (buffer_wpos == buf_elems) {
            buffer_wpos = 0;
            output_stream.write(buffer,io_type_size*buf_elems);
        }
        if (data != NULL) {
            input_norm += data[i]*data[i];
            sse += (data[i]-r[i])*(data[i]-r[i]);
            input_min = min(input_min,data[i]);
            input_max = max(input_max,data[i]);
        }
    }
    if (buffer_wpos > 0)
        output_stream.write(buffer,io_type_size*buffer_wpos);
    delete[] buffer;
    output_stream.close();
//    if (debug) reco.debug();

    if (data != NULL) { // If the uncompressed input is available, we compute the error statistics
        input_norm = sqrt(input_norm);
        double eps = sqrt(sse)/input_norm;
        double rmse = sqrt(sse/size);
        double psnr = 20*log10((input_max-input_min)/(2*rmse));
        cout << "eps = " << eps << ", rmse = " << rmse << ", psnr = " << psnr << endl;
    }

    delete[] r;
}
