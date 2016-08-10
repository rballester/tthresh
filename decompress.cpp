#include "vmmlib-tensor/tensor.hpp"
#include "tthresh.hpp"

using namespace std;

void decode_factor(double* mem, int s, string file1, string file2, string file3) { // TODO double...

    // First, the q for each column
    vector<char> columns_q(s);
    ifstream columns_q_stream(file1.c_str(), ios::in | ios::binary);
    for (int i = 0; i < s; ++i)
        columns_q_stream.read(&columns_q[i],sizeof(char));
    columns_q_stream.close();

    // Next, the matrix's maximum, used for quantization
    ifstream limits_stream(file2.c_str(), ios::in | ios::binary);
    double maximum; // Todo...
    limits_stream.read(reinterpret_cast<char*>(&maximum),sizeof(double));
    limits_stream.close();

    // Finally we dequantize the matrix itself
    ifstream matrix_stream(file3.c_str(), ios::in | ios::binary);
    char matrix_rbyte;
    matrix_stream.read(&matrix_rbyte,sizeof(char));
    char matrix_rbit = 7;
    for (int i = 0; i < s*s; ++i) {
        int column = i/s;
        char q = columns_q[column];
        if (q == 0)
            mem[i] = 0;
        else {
            q += 2;
            unsigned long int to_read = 0;
            for (int j = q; j >= 0; --j) {
                to_read |= ((matrix_rbyte>>matrix_rbit)&1L) << j;
                matrix_rbit--;
                if (matrix_rbit < 0) {
                    matrix_rbit = 7;
                    matrix_stream.read(&matrix_rbyte,sizeof(char));
                }
            }
            char sign = (to_read>>q)&1L;
            to_read &= ~(1L<<q);
            mem[i] = -(2*sign-1)*double(to_read)*maximum/((1L<<q)-double(1));
        }
    }
    matrix_stream.close();
}

void decompress(string compressed_file, string output_file, double* data, bool verbose, bool debug) {

    if (verbose) cout << endl << "/***** Decompression *****/" << endl << endl << flush;

    int __ = system("rm tmp/*");

    stringstream ss;
    ss << "tar -xf " << compressed_file;
    string command(ss.str());
    int _ = system(command.c_str());

    // Read output tensor type
    ifstream io_type_stream("tmp/io_type", ios::in | ios::binary );
    char io_type_code;
    io_type_stream.read(&io_type_code,sizeof(char));
    io_type_stream.close();
    char io_type_size;
    if (io_type_code == 0) io_type_size = sizeof(char);
    else if (io_type_code == 1) io_type_size = sizeof(int);
    else if (io_type_code == 2) io_type_size = sizeof(float);
    else io_type_size = sizeof(double);

    // Read tensor sizes
    ifstream sizes_stream("tmp/sizes", ios::in | ios::binary );
    int s[3];
    sizes_stream.read(reinterpret_cast<char*>(s),3*sizeof(int));
    long int size = s[0]*s[1]*s[2];
    sizes_stream.close();

    // Read chunk encodings and condense them into one array
    ifstream chunk_sizes_stream("tmp/chunk_sizes", ios::in | ios::binary);
    streampos fsize = chunk_sizes_stream.tellg();
    chunk_sizes_stream.seekg( 0, ios::end );
    fsize = chunk_sizes_stream.tellg() - fsize;
    chunk_sizes_stream.seekg( 0, ios::beg );
    int n_chunks = fsize/4;

    // Read q's
    char qs[n_chunks];
    ifstream qs_stream("tmp/qs", ios::in | ios::binary);
    qs_stream.read(reinterpret_cast<char*>(qs),n_chunks*sizeof(char));
    qs_stream.close();

    vector<int> encoding_mask(size,0);
    for (int chunk_num = 1; chunk_num <= n_chunks; ++chunk_num) {

        int chunk_size;
        chunk_sizes_stream.read((char*)(&chunk_size),sizeof(int));

        stringstream ss;
        ss << "cp tmp/mask_" << setw(4) << setfill('0') << chunk_num << ".compressed tmp/mask.compressed";
        string ss_string = ss.str();
        int _ = system(ss_string.c_str());
        decode();
        ifstream mask("tmp/mask.decompressed", ios::in | ios::binary);
        std::vector<char> buffer((istreambuf_iterator<char>(mask)), istreambuf_iterator<char>());
        long int ind = 0;
        for (int i = 0; i < buffer.size() and ind < size; ++i) { // This can be more efficient...
            for (char chunk_rbit = 7; chunk_rbit >= 0 and ind < size; --chunk_rbit) {
                while (encoding_mask[ind] > 0)
                    ind++;
                if ((buffer[i] >> chunk_rbit)&1) {
                    encoding_mask[ind] = chunk_num;
                }
                ind++;
            }
        }
        if (verbose) cout << "chunk " << chunk_num << " had " << buffer.size()*8 << " bits, ind=" << ind
                          << ", q=" << int(qs[chunk_num-1]) << endl << flush;
    }
    chunk_sizes_stream.close();

    // Read minimums
    double minimums[n_chunks];
    ifstream minimums_stream("tmp/minimums", ios::in | ios::binary);
    minimums_stream.read(reinterpret_cast<char*>(minimums),n_chunks*sizeof(double));
    minimums_stream.close();

    // Read maximums
    double maximums[n_chunks];
    ifstream maximums_stream("tmp/maximums", ios::in | ios::binary);
    maximums_stream.read(reinterpret_cast<char*>(maximums),n_chunks*sizeof(double));
    maximums_stream.close();

    // Recover the core
    double* c = new double[size]; // TODO should be double
    ifstream core_quant_stream("tmp/core_quant", ios::in | ios::binary);
    char core_quant_rbyte;
    core_quant_stream.read(&core_quant_rbyte,sizeof(char));
    char core_quant_rbit = 7;
    int shout = 1;
    int bits_read = 0;
    for (int i = 0; i < size; ++i) {
//        if (i < 20) {
//            cout << "Core pos [" << i << "] has encoding " << encoding_mask[i] << ", with q=" << int(qs[encoding_mask[i]-1]) << endl;
//        }
        int chunk_num = encoding_mask[i];
        double chunk_min = minimums[chunk_num-1];
        double chunk_max = maximums[chunk_num-1];
        char q = qs[chunk_num-1];
        char sign = 0;
        if (q > 0) {
            sign = ((core_quant_rbyte >> core_quant_rbit)&1L);
//                if (shout < 30)
//                    cout << int(sign) << " ";
//                shout++;
            core_quant_rbit--;
            bits_read++;
            if (core_quant_rbit < 0) {
                core_quant_stream.read(&core_quant_rbyte,sizeof(char));
                core_quant_rbit = 7;
            }
        }
        long int quant = 0;
        for (long int j = q-1; j >= 0; --j) {
            quant |= ((core_quant_rbyte >> core_quant_rbit)&1L) << j;
//                if (shout < 30)
//                    cout << int(((core_quant_rbyte >> core_quant_rbit)&1)) << " ";
//                shout++;
            core_quant_rbit--;
            bits_read++;
            if (core_quant_rbit < 0) {
                core_quant_stream.read(&core_quant_rbyte,sizeof(char));
                core_quant_rbit = 7;
            }
        }
        double dequant;
        if (q == 0)
            dequant = chunk_min;
        else
            dequant = quant*(chunk_max-chunk_min)/((1L<<q)-1.) + chunk_min;
        c[i] = -(sign*2-1)*dequant;
//            if (chunk_num == 52) {
//                cout << "bits read=" << bits_read << ", chunk_num = " << chunk_num << ", sign = " << int(sign) << ", q = " << int(q) << ", chunk_min = " << chunk_min << ", chunk_max = " << chunk_max << ", quant = " << quant << ", dequant = " << dequant << endl;
//                shout = 0;
//            }
    }

//    for (int i = 0; i < 10; ++i)
//        cout << c[i] << " " << endl;

    // Read factor matrices
    if (verbose) cout << "Decoding factor matrices... " << flush;
    vmml::tensor<double> U1(s[0],s[0]), U2(s[1],s[1]), U3(s[2],s[2]);
    decode_factor(U1.get_array(),s[0],"tmp/U1_q","tmp/U1_limits","tmp/U1");
    decode_factor(U2.get_array(),s[1],"tmp/U2_q","tmp/U2_limits","tmp/U2");
    decode_factor(U3.get_array(),s[2],"tmp/U3_q","tmp/U3_limits","tmp/U3");
    if (verbose) cout << "Done" << endl << flush;

    vmml::tensor<double> core(s[0],s[1],s[2]); // TODO
    core.set_memory(c);
    if (verbose) cout << "Reconstructing tensor... " << flush;
    vmml::tensor<double> reco = core.ttm(U1,U2,U3);
    if (verbose) cout << "Done" << endl << flush;

    ofstream output_stream(output_file.c_str(), ios::out | ios::binary);
    double* r = reco.get_array();
    long int buf_elems = 1<<20;
    char* buffer = new char[io_type_size*buf_elems];
    long int buffer_wpos = 0;
    double sse = 0;
    double input_norm = 0;
    double input_min = std::numeric_limits<double>::max(); // TODO
    double input_max = std::numeric_limits<double>::min(); // TODO
    for (long int i = 0; i < size; ++i) {
        if (io_type_code == 0)
            reinterpret_cast<unsigned char*>(buffer)[buffer_wpos] = abs(r[i]);
        else if (io_type_code == 1)
            reinterpret_cast<int*>(buffer)[buffer_wpos] = r[i];
        else if (io_type_code == 2)
            reinterpret_cast<float*>(buffer)[buffer_wpos] = r[i];
        else
            reinterpret_cast<double*>(buffer)[buffer_wpos] = r[i];
//        if (i < 10)
//            cout << "i=" << i << ", r[i]=" << r[i] << endl;
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
    output_stream.close();
    if (debug) reco.debug();

    if (data != NULL) { // If the uncompressed input is available, we compute the error statistics
        input_norm = sqrt(input_norm);
        double eps = sqrt(sse)/input_norm;
        double rmse = sqrt(sse/size);
        double psnr = 20*log10((input_max-input_min)/(2*rmse));
        cout << "eps = " << eps << ", rmse = " << rmse << ", psnr = " << psnr << endl;
    }
}
