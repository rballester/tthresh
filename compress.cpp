#include <fstream>
#include <vector>
#include "vmmlib-tensor/tensor.hpp"
#include "tthresh.hpp"

using namespace std;

void encode_factor(double* mem, int s, vector<char>& columns_q, string file1, string file2, string file3) { // TODO double...

    // First, the q for each column
    ofstream columns_q_stream(file1.c_str(), ios::out | ios::binary);
    for (int i = 0; i < s; ++i)
        columns_q_stream.write(&columns_q[i],sizeof(char));
    columns_q_stream.close();

    // Next, the matrix's maximum, used for quantization
    ofstream limits_stream(file2.c_str(), ios::out | ios::binary);
    double maximum = std::numeric_limits<double>::min(); // Todo...
    for (int i = 0; i < s*s; ++i)
        maximum = max(maximum,abs(mem[i]));
    limits_stream.write(reinterpret_cast<char*>(&maximum),sizeof(double));
    limits_stream.close();

    // Finally the matrix itself, quantized
    ofstream matrix_stream(file3.c_str(), ios::out | ios::binary);
    char matrix_wbyte = 0;
    char matrix_wbit = 7;
    for (int i = 0; i < s*s; ++i) {
        int column = i/s;
        char q = columns_q[column];
        if (q > 0) {
            q += 2; // Seems a good compromise
            unsigned long int to_write = roundl(abs(mem[i])*((1L<<q)-1)/maximum);
            to_write |= (mem[i]<0)*(1L<<q);
            for (long int j = q; j >= 0; --j) {
                matrix_wbyte |= ((to_write>>j)&1L) << matrix_wbit;
                matrix_wbit--;
                if (matrix_wbit < 0) {
                    matrix_wbit = 7;
                    matrix_stream.write(&matrix_wbyte,sizeof(char));
                    matrix_wbyte = 0;
                }
            }
        }
    }
    if (matrix_wbit < 7)
        matrix_stream.write(&matrix_wbyte,sizeof(char));
    matrix_stream.close();
}

double* compress(string input_file, string compressed_file, string io_type, int s[3], Target target, double target_value, bool verbose, bool debug) {

    if (verbose) cout << endl << "/***** Compression *****/" << endl << endl << flush;

    int __ = system("rm tmp/*");

    /**************************/
    // Read the input data file
    /**************************/

    int size = s[0]*s[1]*s[2];
    char type_size;
    if (io_type == "uchar")
        type_size = sizeof(char);
    else if (io_type == "int")
        type_size = sizeof(int);
    else if (io_type == "float")
        type_size = sizeof(float);
    else if (io_type == "double")
        type_size = sizeof(double);
    else {
        cout << "Unrecognized I/O type" << endl;
        exit(1);
    }
    char* in = new char[size*type_size];
    ifstream in_stream(input_file.c_str(), ios::in | ios::binary);
    if (!in_stream.is_open()) {
        cout << "Could not open \"" << input_file << "\"" << endl;
        exit(1);
    }
    streampos fsize = in_stream.tellg();
    in_stream.seekg( 0, ios::end );
    fsize = in_stream.tellg() - fsize;
    if (size*type_size != fsize) {
        cout << "Invalid file size (expected " << s[0] << "*" << s[1] << "*" << s[2] << "*"
             << int(type_size) << " = " << size*type_size << ", but is " << fsize << ")" << endl;
        exit(1);
    }
    in_stream.seekg(0, ios::beg);
    in_stream.read(in,size*type_size);
    in_stream.close();
    double* data; // TODO change to double
    if (io_type == "double") {
        data = (double *)in;
    }
    else {
        data = new double[size];
        if (io_type == "uchar") {
            for (int i = 0; i < size; ++i)
                data[i] = *reinterpret_cast<unsigned char *>(&in[i*type_size]);
        }
        else if (io_type == "int") {
            for (int i = 0; i < size; ++i)
                data[i] = *reinterpret_cast<int *>(&in[i*type_size]);
        }
        else {
            for (int i = 0; i < size; ++i)
                data[i] = *reinterpret_cast<float *>(&in[i*type_size]);
        }
        delete in;
    }

    /*********************************/
    // Create and decompose the tensor
    /*********************************/

    // TODO: don't copy memory again
    vmml::tensor<double> X(s[0],s[1],s[2]);
    X.set_memory(data);
    if (debug) X.debug();
    if (verbose) cout << "Decomposing the tensor... " << flush;
    vmml::tensor<double> core(s[0],s[1],s[2]);
    vmml::tensor<double> U1(s[0],s[0]), U2(s[1],s[1]), U3(s[2],s[2]);
    U1.set_dct();
    U2.set_dct();
    U3.set_dct();
    X.tucker_decomposition(core,U1,U2,U3,1,0,false);
    if (verbose) cout << "Done" << endl << flush;

    /**********************************************************************/
    // Compute the target SSE (sum of squared errors) from the given metric
    /**********************************************************************/

    double sse;
    if (target == eps)
        sse = pow((target_value*X.frobenius_norm()),2);
    else if (target == rmse)
        sse = pow(target_value,2)*size;
    else
        sse = pow((X.maximum()-X.minimum()) / (2*(pow(10,target_value/20))),2) * size;
    double lim = sse/size;
    if (debug) cout << "We target SSE=" << lim << endl;

    /***********************************/
    // Sort abs(core) in ascending order
    /***********************************/

    if (verbose) cout << "Sorting core's absolute values... " << flush;
    double* c = core.get_array(); // TODO -> double
    vector< pair<double,int> > sorting(size);
    for (int i = 0; i < size; ++i)
        sorting[i] = pair<double,int>(abs(c[i]),i);
    sort(sorting.begin(),sorting.end());
    if (verbose) cout << "Done" << endl << flush;

    /************************************************/
    // Generate adaptive chunks from the sorted curve
    /************************************************/

    int adder = 1;
    char q = 0;
    int left = 0;
    int old_right = left;
    int right;
    ofstream chunk_sizes_stream("tmp/chunk_sizes", ios::out | ios::binary);
    ofstream minimums_stream("tmp/minimums", ios::out | ios::binary);
    ofstream maximums_stream("tmp/maximums", ios::out | ios::binary);
    ofstream qs_stream("tmp/qs", ios::out | ios::binary);
    vector<char> qs_vec;
    vector<int> encoding_mask(size,0);
    int chunk_num = 1;
    unsigned long int* core_to_write = new unsigned long int[size];
    for (int i = 0; i < size; ++i)
        core_to_write[i] = 0; // TODO unneeded
    vector<char> U1_q(s[0],0);
    vector<char> U2_q(s[1],0);
    vector<char> U3_q(s[2],0);

    while (left < size) {
        bool we_just_reduced = false;
        while (left < size) {
            right = min(size,old_right+adder);
            double chunk_min = sorting[left].first;
            double chunk_max = sorting[right-1].first;
            double sse = 0;
            if (right > left+1) {
                if (q > 0) {
                    for (int i = left; i < right; ++i) {
                        long int quant = roundl((sorting[i].first-chunk_min)*((1L<<q)-1.)/(chunk_max-chunk_min));
                        double dequant = quant*(chunk_max-chunk_min)/((1L<<q)-1.) + chunk_min;
                        sse += (sorting[i].first-dequant)*(sorting[i].first-dequant);
                    }
                }
                else {
                    for (int i = left; i < right; ++i)
                        sse += (sorting[i].first-chunk_min)*(sorting[i].first-chunk_min);
                }
            }
            double msse = sse/(right-left);
            if (debug) cout << "We try [" << left << "," << right << "), adder = " << adder << ", msse = " << msse << endl;
            if (msse >= 0.9*lim or right == size) {
                if (msse >= lim) {
                    if (adder > 1) {
                        adder = ceil(adder/4.);
                        we_just_reduced = true;
                        continue;
                    }
                    else {
                        right--;
                        break;
                    }
                }
                else
                    break;
            }
            else {
                old_right = right;
                if (we_just_reduced)
                    we_just_reduced = false;
                else
                    adder *= 2;
            }
        }

        /********************************************/
        // Fill the core buffer with quantized values
        /********************************************/

        double chunk_min = sorting[left].first;
        double chunk_max = sorting[right-1].first;
        int chunk_size = (right-left);
        chunk_sizes_stream.write(reinterpret_cast<char*>( &chunk_size ),sizeof(int));
        minimums_stream.write(reinterpret_cast<char*>( &chunk_min ),sizeof(double));
        maximums_stream.write(reinterpret_cast<char*>( &chunk_max ),sizeof(double)); // TODO not if q = 0
        qs_stream.write(reinterpret_cast<char*>( &q ),sizeof(char));
        qs_vec.push_back(q);
        if (q > 0) { // If q = 0 there's no need to store anything quantized, not even the sign
            for (int i = left; i < right; ++i) {
                unsigned long int to_write;
                if (right-left == 1)
                    to_write = 0; // TODO not needed...
                else
                    to_write = roundl((sorting[i].first-chunk_min)*((1L<<q)-1)/(chunk_max-chunk_min));
                to_write |= (c[sorting[i].second]<0)*(1L<<q);
                core_to_write[sorting[i].second] = to_write; // TODO copy into c[] directly
            }
        }

        /********************************************/
        // Save mask and compute RLE+Huffman out of it
        /********************************************/

        for (int i = left; i < right; ++i) {
            long int index = sorting[i].second;
            encoding_mask[index] = chunk_num;
            // We use this loop also to store the needed quantization bits per factor column
            int x = index%s[0];
            int y = index%(s[0]*s[1])/s[1];
            int z = index/(s[0]*s[1]);
            U1_q[x] = max(U1_q[x],q);
            U2_q[y] = max(U2_q[y],q);
            U3_q[z] = max(U3_q[z],q);
        }
        ofstream mask("tmp/mask.raw", ios::out | ios::binary);
        char mask_wbyte = 0;
        char mask_wbit = 7;
        for (int i = 0; i < size; ++i) {
            if (encoding_mask[i] == 0)
                mask_wbit--;
            else if (encoding_mask[i] == chunk_num) {
                mask_wbyte |= 1 << mask_wbit;
                mask_wbit--;
            }
            if (mask_wbit < 0) {
                mask.write(&mask_wbyte,sizeof(char));
                mask_wbyte = 0;
                mask_wbit = 7;
            }
        }
        if (mask_wbit < 7)
            mask.write(&mask_wbyte,sizeof(char));
        mask.close();
        encode();
        if (debug) {
            decode();
            if (system("diff tmp/mask.raw tmp/mask.decompressed") != 0) {
                cout << "Huffman error" << endl;
                exit(1);
            }
        }
        int coeff_bits = 0;
        if (q > 0)
            coeff_bits = (q+1)*(right-left); // The "+1" is for the sign
        std::ifstream in("tmp/mask.compressed", std::ifstream::ate | std::ifstream::binary);
        int huffman_bits = in.tellg()*8;
        in.close();
        stringstream ss;
        ss << "mv tmp/mask.compressed tmp/mask_" << setw(4) << setfill('0') << chunk_num << ".compressed";
        string ss_string = ss.str();
        int _ = system(ss_string.c_str());
        if (verbose) cout << "Chunk " << chunk_num << " accepted, min=" << chunk_min << ", max=" << chunk_max
                          << ", cbits=" << coeff_bits << ", hbits=" << huffman_bits << ", q=" << int(q) << ", left="
                          << left << ", right=" << right << " (size=" << right-left << ")" << endl << flush;

        // Update control variables
        q++;
        if (q >= 64) {
            cout << "q grew too much" << endl;
            exit(1);
        }
        left = right;
        old_right = left;
        chunk_num++;
    }
    chunk_sizes_stream.close();
    minimums_stream.close();
    maximums_stream.close();
    qs_stream.close();

    /********************************************/
    // Save the core's encoding
    /********************************************/

    if (verbose) cout << "Saving core encoding... " << endl << flush;
    ofstream core_quant_stream("tmp/core_quant", ios::out | ios::binary);
    char core_quant_wbyte = 0;
    char core_quant_wbit = 7;
    for (int i = 0; i < size; ++i) {
        chunk_num = encoding_mask[i];
        char q = qs_vec[chunk_num-1];
        if (q > 0) {
            for (long int j = q; j >= 0; --j) {
                core_quant_wbyte |= ((core_to_write[i] >> j)&1L) << core_quant_wbit;
                core_quant_wbit--;
                if (core_quant_wbit < 0) {
                    core_quant_stream.write(&core_quant_wbyte,sizeof(char));
                    core_quant_wbyte = 0;
                    core_quant_wbit = 7;
                }
            }
        }
    }
    if (core_quant_wbit < 7)
        core_quant_stream.write(&core_quant_wbyte,sizeof(char));
    core_quant_stream.close();
    delete core_to_write;
    if (verbose) cout << "Done" << endl << flush;

    /****************************/
    // Save tensor sizes and type
    /****************************/

    ofstream sizes_stream("tmp/sizes", ios::out | ios::binary);
    sizes_stream.write(reinterpret_cast<char*>( &s[0] ),sizeof(int));
    sizes_stream.write(reinterpret_cast<char*>( &s[1] ),sizeof(int));
    sizes_stream.write(reinterpret_cast<char*>( &s[2] ),sizeof(int));
    sizes_stream.close();

    ofstream io_type_stream("tmp/io_type", ios::out | ios::binary);
    char io_type_code;
    if (io_type == "uchar") io_type_code = 0;
    else if (io_type == "int") io_type_code = 1;
    else if (io_type == "float") io_type_code = 2;
    else io_type_code = 3;
    io_type_stream.write(reinterpret_cast<char*>( &io_type_code ),sizeof(char));
    io_type_stream.close();

    /*********************************/
    // Encode and save factor matrices
    /*********************************/

    if (debug) {
        for (int i = 0; i < s[0]; ++i)
            cout << " " << int(U1_q[i]);
        cout << endl;
        for (int i = 0; i < s[1]; ++i)
            cout << " " << int(U2_q[i]);
        cout << endl;
        for (int i = 0; i < s[2]; ++i)
            cout << " " << int(U3_q[i]);
        cout << endl;
    }

    if (verbose) cout << "Encoding factor matrices... " << flush;
    encode_factor(U1.get_array(),s[0],U1_q,"tmp/U1_q","tmp/U1_limits","tmp/U1");
    encode_factor(U2.get_array(),s[1],U2_q,"tmp/U2_q","tmp/U2_limits","tmp/U2");
    encode_factor(U3.get_array(),s[2],U3_q,"tmp/U3_q","tmp/U3_limits","tmp/U3");
    if (verbose) cout << "Done" << endl << flush;

    /***********************************************/
    // Tar+gzip the final result and compute the bpv
    /***********************************************/

    {
        stringstream ss;
        ss << "tar -czf " << compressed_file << " " << "tmp/";
        string command(ss.str());
        int _ = system(command.c_str());
        ifstream bpv_stream(compressed_file.c_str(), ios::in | ios::binary);
        streampos beginning = bpv_stream.tellg();
        bpv_stream.seekg( 0, ios::end );
        long int newbits = (bpv_stream.tellg() - beginning)*8;
        cout << "oldbits = " << size*type_size*8 << ", newbits = " << newbits << ", compressionrate = " << size*type_size*8/double(newbits)
                << ", bpv = " << newbits/double(size) << endl << flush;
        bpv_stream.close();
    }

    return data;
}