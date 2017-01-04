#include <fstream>
#include <vector>
#include "tthresh.hpp"
#include "tucker.hpp"
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// Structure of the compressed file:
// 3 * 4 bytes: tensor sizes
// 1 byte: tensor type
// 1 byte: number of chunks
// n_chunks * (chunk_info + compressed mask)
// matrix_info
// core_quant

void encode_factor(MatrixXd& U, int s, vector<char>& columns_q, ofstream& matrix_info_stream) {
    
    // First, the matrix's maximum, used for quantization
    double maximum = U.maxCoeff();
    matrix_info_stream.write(reinterpret_cast<char*>(&maximum),sizeof(double));
    
    // Next, the q for each column
    for (int i = 0; i < s; ++i)
        matrix_info_stream.write(reinterpret_cast<char*>(&columns_q[i]),sizeof(char));

    // Finally the matrix itself, quantized
    char matrix_wbyte = 0;
    char matrix_wbit = 7;
    for (int j = 0; j < s; ++j) {
        for (int i = 0; i < s; ++i) {
            char q = columns_q[j];
            if (q > 0) {
                q = min(63,q+2); // Seems a good compromise
                unsigned long int to_write = min(((1UL<<q)-1),(unsigned long int)roundl(abs(U(i,j))/maximum*((1UL<<q)-1)));
                to_write |= (U(i,j)<0)*(1UL<<q); // The sign is the first bit to write
                for (long int j = q; j >= 0; --j) {
                    matrix_wbyte |= ((to_write>>j)&1UL) << matrix_wbit;
                    matrix_wbit--;
                    if (matrix_wbit < 0) {
                        matrix_wbit = 7;
                        matrix_info_stream.write(&matrix_wbyte,sizeof(char));
                        matrix_wbyte = 0;
                    }
                }
            }
        }
    }
    if (matrix_wbit < 7)
        matrix_info_stream.write(&matrix_wbyte,sizeof(char));
    //matrix_info_stream.close();
}

double* compress(string input_file, string compressed_file, string io_type, int s[3], Target target, double target_value, bool verbose, bool debug) {

    if (verbose) cout << endl << "/***** Compression *****/" << endl << endl << flush;

    int _ = system("mkdir -p tthresh-tmp/");
    _ = system("rm -f tthresh-tmp/*");

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
    
    // Load input file into memory
    char* in = new char[size*type_size];
    ifstream in_stream(input_file.c_str(), ios::in | ios::binary);
    if (!in_stream.is_open()) {
        cout << "Could not open \"" << input_file << "\"" << endl;
        exit(1);
    }
    streampos fsize = in_stream.tellg(); // Check that buffer size matches expected size
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
    
    // Cast the tensor to doubles
    double* data;
    double dmin = numeric_limits<double>::max(), dmax = numeric_limits<double>::min(), dnorm = 0; // Tensor statistics
    if (io_type == "double")
        data = (double *)in;
    else
        data = new double[size];
    for (int i = 0; i < size; ++i) {
        if (io_type == "uchar")
            data[i] = static_cast<unsigned char>(in[i*type_size]);
        else if (io_type == "int")
            data[i] = static_cast<int>(in[i*type_size]);
        else if (io_type == "float")
            data[i] = static_cast<float>(in[i*type_size]);
        dmin = min(dmin,data[i]);
        dmax = max(dmax,data[i]);
        dnorm += data[i]*data[i];
    }
    dnorm = sqrt(dnorm);
    if (io_type != "double")
        delete[] in;

    /****************************/
    // Save tensor sizes and type
    /****************************/

    ofstream all_stream("tthresh-tmp/all", ios::out | ios::binary);
    //ofstream sizes_stream("tthresh-tmp/all", ios::out | ios::binary);
    all_stream.write(reinterpret_cast<char*>( &s[0] ),sizeof(int));
    all_stream.write(reinterpret_cast<char*>( &s[1] ),sizeof(int));
    all_stream.write(reinterpret_cast<char*>( &s[2] ),sizeof(int));
    //all_stream.close();

    char io_type_code;
    if (io_type == "uchar") io_type_code = 0;
    else if (io_type == "int") io_type_code = 1;
    else if (io_type == "float") io_type_code = 2;
    else io_type_code = 3;
    all_stream.write(reinterpret_cast<char*>( &io_type_code ),sizeof(char));
    all_stream << char(0); // We don't know the number of chunks yet
    //cout << "**** n_chunks=" << int(n_chunks) << endl;
    //all_stream.close();    

    /*********************************/
    // Create and decompose the tensor
    /*********************************/

    if (verbose) cout << "Decomposing the tensor... " << flush;
    double* c = new double[size];
    memcpy(c,data,size*sizeof(double));
    MatrixXd U1, U2, U3;
    tucker(c,s,U1,U2,U3,true);
    if (verbose) cout << "Done" << endl << flush;

    /**********************************************************************/
    // Compute the target SSE (sum of squared errors) from the given metric
    /**********************************************************************/
    
    double sse;
    if (target == eps)
        sse = pow(target_value*dnorm,2);
    else if (target == rmse)
        sse = pow(target_value,2)*size;
    else
        sse = pow((dmax-dmin) / (2*(pow(10,target_value/20))),2) * size;
    double lim = sse/size;
    if (debug) cout << "We target MSE = " << lim << endl;
    
    /***********************************/
    // Sort abs(core) in ascending order
    /***********************************/

    if (verbose) cout << "Sorting core's absolute values... " << flush;
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
    int old_right = left; // Inclusive bound
    int right = left; // Exclusive bound
    //ofstream chunk_info_stream("tthresh-tmp/chunk_info", ios::out | ios::binary);
    //ofstream chunk_sizes_stream("tthresh-tmp/chunk_sizes", ios::out | ios::binary);
    //ofstream minimums_stream("tthresh-tmp/minimums", ios::out | ios::binary);
    //ofstream maximums_stream("tthresh-tmp/maximums", ios::out | ios::binary);
    vector<int> encoding_mask(size,0);
    int chunk_num = 1;
    vector<char> U1_q(s[0],0);
    vector<char> U2_q(s[1],0);
    vector<char> U3_q(s[2],0);

    while (left < size) {
        while (left < size and q < 63) {
            right = min(size,old_right+adder);
            double chunk_min = sorting[left].first;
            double chunk_max = sorting[right-1].first;
            double sse = 0;
            if (right > left+1) {
                if (q > 0) {
                    for (int i = left; i < right; ++i) { // TODO Can we approximate the error computation?
                        long int quant = roundl((sorting[i].first-chunk_min)*((1UL<<q)-1.)/(chunk_max-chunk_min));
                        double dequant = quant*(chunk_max-chunk_min)/((1UL<<q)-1.) + chunk_min;
                        sse += (sorting[i].first-dequant)*(sorting[i].first-dequant);
                    }
                }
                else {
                    for (int i = left; i < right; ++i)
                        sse += (sorting[i].first-chunk_min)*(sorting[i].first-chunk_min);
                }
            }
            double mse = sse/(right-left);
            if (debug) cout << "We try [" << left << "," << right << "), adder = " << adder << ", mse = " << mse << endl;
            if (mse >= 0.9*lim or right == size) {
                if (mse >= lim) {
                    if (adder > 1) {
                        adder = ceil(adder/4.);
                        continue;
                    }
                    else {
                        right = old_right;
                        break;
                    }
                }
                else
                    break;
            }
            else {
                old_right = right;
                adder *= 2;
            }
        }

        if (q == 63)
            right = size;

	int chunk_size = (right-left);
        double chunk_min = sorting[left].first;
        double chunk_max = sorting[right-1].first;

        /********************************************/
        // Quantized (in-place) the core elements
        /********************************************/

        for (int i = left; i < right; ++i) {
	    // If q = 0 there's no need to store anything quantized, not even the sign
	    // If q = 63, values are kept as they are and we forget about quantization
            if (q > 0 and q < 63) {
                unsigned long int to_write = 0;
                if (chunk_size > 1)
                    // The following min() prevents overflowing the q-bit representation when converting double -> long int
                    to_write = min(((1UL<<q)-1),(unsigned long int)roundl((sorting[i].first-chunk_min)/(chunk_max-chunk_min)*((1UL<<q)-1)));
                to_write |= (c[sorting[i].second]<0)*(1UL<<q);
                c[sorting[i].second] = static_cast<double>(to_write);
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
            int y = index%(s[0]*s[1])/s[0];
            int z = index/(s[0]*s[1]);
            U1_q[x] = max(U1_q[x],q);
            U2_q[y] = max(U2_q[y],q);
            U3_q[z] = max(U3_q[z],q);
        }

        vector<char> mask;
        //ofstream mask("tthresh-tmp/mask.raw", ios::out | ios::binary);
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
                //mask.write(&mask_wbyte,sizeof(char));
		mask.push_back(mask_wbyte);
                mask_wbyte = 0;
                mask_wbit = 7;
            }
        }
        if (mask_wbit < 7)
	    mask.push_back(mask_wbyte);
            //mask.write(&mask_wbyte,sizeof(char));
        //mask.close();
        stringstream ss;
        //ss << "tthresh-tmp/mask_" << setw(4) << setfill('0') << chunk_num << ".compressed";
        //encode("tthresh-tmp/mask.raw",ss.str());
	vector<char> encoding;
	encode(mask,encoding);
	chunk_info ci;
	ci.compressed_size = encoding.size();
	ci.minimum = chunk_min;
	ci.maximum = chunk_max;
	all_stream.write(reinterpret_cast<char*>( &ci ),sizeof(chunk_info));
        std::copy(encoding.begin(), encoding.end(), std::ostreambuf_iterator < char >(all_stream));

        //if (debug) { // Check RLE+Huffman correctness
        //    decode(ss.str(),"tthresh-tmp/mask.decompressed");
        //    if (system("diff tthresh-tmp/mask.raw tthresh-tmp/mask.decompressed") != 0) {
        //        cout << "Huffman error" << endl;
        //        exit(1);
        //    }
        //    remove("tthresh-tmp/mask.decompressed");
        //}
        //remove("tthresh-tmp/mask.raw");
        if (verbose) {
            int coeff_bits = 0;
            if (q > 0)
                coeff_bits = (q+1)*(right-left); // The "+1" is for the sign
            string ss_str = ss.str();
            std::ifstream in(ss_str.c_str(), std::ifstream::ate | std::ifstream::binary);
            //int huffman_bits = in.tellg()*8;
            in.close();
            cout << "Encoded chunk " << chunk_num << ", min=" << chunk_min << ", max=" << chunk_max
                << ", cbits=" << coeff_bits << ", q=" << int(q) << ", bits=["
                << left << "," << right << "), size=" << right-left << endl << flush;
        }

        // Update control variables
        q++;
        left = right;
        old_right = left;
        chunk_num++;
    }
    //chunk_sizes_stream.close();
    //minimums_stream.close();
    //maximums_stream.close();
    //all_stream.close();
    
    /*********************************/
    // Encode and save factor matrices
    /*********************************/

    if (debug) {
        cout << "q's for the factor columns: " << endl;
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
    //ofstream matrix_info_stream("tthresh-tmp/Us_all", ios::out | ios::binary);
    encode_factor(U1,s[0],U1_q,all_stream);
    encode_factor(U2,s[1],U2_q,all_stream);
    encode_factor(U3,s[2],U3_q,all_stream);
    if (verbose) cout << "Done" << endl << flush;
    
    /********************************************/
    // Save the core encoding
    /********************************************/

    if (verbose) cout << "Saving core encoding... " << flush;
    //ofstream core_quant_stream("tthresh-tmp/core_quant", ios::out | ios::binary);
    char core_quant_wbyte = 0;
    char core_quant_wbit = 7;
    for (int i = 0; i < size; ++i) {
        chunk_num = encoding_mask[i];
        char q = chunk_num-1;
        if (q > 0) {
            for (long int j = q; j >= 0; --j) {
                core_quant_wbyte |= ((static_cast<unsigned long int>(c[i]) >> j)&1UL) << core_quant_wbit;
                core_quant_wbit--;
                if (core_quant_wbit < 0) {
                    all_stream.write(&core_quant_wbyte,sizeof(char));
                    core_quant_wbyte = 0;
                    core_quant_wbit = 7;
                }
            }
        }
    }
    if (core_quant_wbit < 7)
        all_stream.write(&core_quant_wbyte,sizeof(char));
    if (verbose) cout << "Done" << endl << flush;
    delete[] c;

    // Finally: write the number of chunks, which we now know
    all_stream.seekp(13);
    unsigned char n_chunks = q;
    all_stream.write(reinterpret_cast<char*>( &n_chunks ),sizeof(char));
    all_stream.close();

    /***********************************************/
    // Tar+gzip the final result and compute the bpv
    /***********************************************/

    {
        stringstream ss;
        ss << "tar -czf " << compressed_file << " " << "tthresh-tmp/";
        string command(ss.str());
        int _ = system(command.c_str());
        ifstream bpv_stream(compressed_file.c_str(), ios::in | ios::binary);
        streampos beginning = bpv_stream.tellg();
        bpv_stream.seekg( 0, ios::end );
        long int newbits = (bpv_stream.tellg() - beginning)*8;
        cout << "oldbits = " << size*type_size*8L << ", newbits = " << newbits << ", compressionrate = " << size*type_size*8L/double(newbits)
                << ", bpv = " << newbits/double(size) << endl << flush;
        bpv_stream.close();
    }

    return data;
}
