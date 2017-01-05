#include <fstream>
#include <vector>
#include "tthresh.hpp"
#include "tucker.hpp"
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// *** Structure of the compressed file ***
// 1 byte: number of dimensions N
// N * 4 bytes: tensor sizes
// 1 byte: tensor type
// 1 byte: number of chunks
// Chunk information and masks: n_chunks * (chunk_info + compressed mask)
// Factor matrices
// The quantized core

void encode_factor(MatrixXd & U, int n_columns, vector < char >&columns_q, ofstream & output_stream)
{
    // First, the matrix's maximum, used for quantization
    double maximum = U.maxCoeff();
    output_stream.write(reinterpret_cast < char *>(&maximum), sizeof(double));

    // Next, the q for each column
    for (int i = 0; i < n_columns; ++i)
	output_stream.write(reinterpret_cast < char *>(&columns_q[i]), sizeof(char));

    // Finally the matrix itself, quantized
    char matrix_wbyte = 0;
    char matrix_wbit = 7;
    for (int j = 0; j < n_columns; ++j) {
	for (int i = 0; i < n_columns; ++i) {
	    char q = columns_q[j];
	    if (q > 0) {
		q = min(63, q + 2);	// Seems a good compromise
		unsigned long int to_write = min(((1UL << q) - 1),
						 (unsigned long int)
						 roundl(abs(U(i, j)) / maximum * ((1UL << q) - 1)));
		to_write |= (U(i, j) < 0) * (1UL << q);	// The sign is the first bit to write
		for (long int j = q; j >= 0; --j) {
		    matrix_wbyte |= ((to_write >> j) & 1UL) << matrix_wbit;
		    matrix_wbit--;
		    if (matrix_wbit < 0) {
			matrix_wbit = 7;
			output_stream.write(&matrix_wbyte, sizeof(char));
			matrix_wbyte = 0;
		    }
		}
	    }
	}
    }
    if (matrix_wbit < 7)
	output_stream.write(&matrix_wbyte, sizeof(char));
}

double *compress(string input_file, string compressed_file, string io_type, vector < int >&s, Target target, double target_value, bool verbose, bool debug)
{
    if (verbose)
	cout << endl << "/***** Compression *****/" << endl << endl << flush;

    int _ = system("mkdir -p tthresh-tmp/");
    _ = system("rm -f tthresh-tmp/*");

    /**************************/
    // Read the input data file
    /**************************/

    char n = s.size();
    long int size = 1;
    for (int i = 0; i < n; ++i)
	size *= s[i];
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

    /****************************/
    // Load input file into memory
    /****************************/
    
    char *in = new char[size * type_size];
    ifstream input_stream(input_file.c_str(), ios::in | ios::binary);
    if (!input_stream.is_open()) {
	cout << "Could not open \"" << input_file << "\"" << endl;
	exit(1);
    }
    streampos fsize = input_stream.tellg();	// Check that buffer size matches expected size
    input_stream.seekg(0, ios::end);
    fsize = input_stream.tellg() - fsize;
    if (size * type_size != fsize) {
	cout << "Invalid file size: expected (" << s[0];
	for (int i = 1; i < n; ++i)
	    cout << "*" << s[i];
	cout << ") * " << int (type_size) << " = " << size * type_size << ", but is " << fsize;
	if (size * type_size > fsize) {
	    cout << " (" << size * type_size / double (fsize) << " times too small)" << endl;
	} else {
	    cout << " (" << fsize / double (size * type_size) << " times too large)" << endl;
	}
	exit(1);
    }
    input_stream.seekg(0, ios::beg);
    input_stream.read(in, size * type_size);
    input_stream.close();

    /****************************/
    // Save tensor dimensionality, sizes and type
    /****************************/

    ofstream output_stream("tthresh-tmp/all", ios::out | ios::binary);
    output_stream << n;		// Number of dimensions
    output_stream.write(reinterpret_cast < char *>(&s[0]), n * sizeof(int));

    char io_type_code;
    if (io_type == "uchar")
	io_type_code = 0;
    else if (io_type == "int")
	io_type_code = 1;
    else if (io_type == "float")
	io_type_code = 2;
    else
	io_type_code = 3;
    output_stream.write(reinterpret_cast < char *>(&io_type_code), sizeof(char));
    output_stream << char (0);	// We don't know the number of chunks yet 
    
    // Cast the tensor to doubles
    double *data;
    double dmin = numeric_limits < double >::max(), dmax = numeric_limits < double >::min(), dnorm = 0;	// Tensor statistics
    if (io_type == "double")
	data = (double *) in;
    else
	data = new double[size];
    for (int i = 0; i < size; ++i) {
	if (io_type == "uchar") {
	    data[i] = static_cast < unsigned char >(in[i * type_size]);
	} else if (io_type == "int") {
	    data[i] = static_cast < int >(in[i * type_size]);
	} else if (io_type == "float") {
	    data[i] = static_cast < float >(in[i * type_size]);
	}
	dmin = min(dmin, data[i]);
	dmax = max(dmax, data[i]);
	dnorm += data[i] * data[i];
    }
    dnorm = sqrt(dnorm);
    if (io_type != "double")
	delete[]in;

    /**********************************************************************/
    // Compute the target SSE (sum of squared errors) from the given metric
    /**********************************************************************/

    double sse;
    if (target == eps)
	sse = pow(target_value * dnorm, 2);
    else if (target == rmse)
	sse = pow(target_value, 2) * size;
    else
	sse = pow((dmax - dmin) / (2 * (pow(10, target_value / 20))), 2) * size;
    double lim = sse / size;
    if (debug)
	cout << "We target MSE = " << lim << endl;

    /*********************************/
    // Create and decompose the tensor
    /*********************************/

    if (verbose)
	cout << "Decomposing the " << int (n) << "D tensor... " << flush;
    double *c = new double[size];	// Tucker core
    memcpy(c, data, size * sizeof(double));
    vector < MatrixXd > Us(n);	// Tucker factor matrices
    tucker(c, s, Us, true);
    if (verbose)
	cout << "Done" << endl << flush;

    /***********************************/
    // Sort abs(core) in ascending order
    /***********************************/

    if (verbose)
	cout << "Sorting core's absolute values... " << flush;
    vector < pair < double, int >>sorting(size);
    for (int i = 0; i < size; ++i)
	sorting[i] = pair < double, int >(abs(c[i]), i);
    sort(sorting.begin(), sorting.end());
    if (verbose)
	cout << "Done" << endl << flush;

    /************************************************/
    // Generate adaptive chunks from the sorted curve
    /************************************************/

    long int adder = 1;
    char q = 0;
    long int left = 0;
    long int old_right = left;	// Inclusive bound
    long int right = left;	// Exclusive bound
    vector < int >encoding_mask(size, 0);
    int chunk_num = 1;
    vector < vector < char >>Us_q(n);
    for (int i = 0; i < n; ++i)
	Us_q[i] = vector < char >(s[i], 0);

    while (left < size) {
	while (left < size and q < 63) {
	    right = min(size, old_right + adder);
	    double chunk_min = sorting[left].first;
	    double chunk_max = sorting[right - 1].first;
	    double sse = 0;
	    if (right > left + 1) {
		if (q > 0) {
		    for (int i = left; i < right; ++i) {	// TODO Can we approximate the error computation?
			long int quant = roundl((sorting[i].first - chunk_min) * ((1UL << q) - 1.) / (chunk_max - chunk_min));
			double dequant = quant * (chunk_max - chunk_min) / ((1UL << q) - 1.) + chunk_min;
			sse += (sorting[i].first - dequant) * (sorting[i].first - dequant);
		    }
		} else {
		    for (int i = left; i < right; ++i)
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

	int chunk_size = (right - left);
	double chunk_min = sorting[left].first;
	double chunk_max = sorting[right - 1].first;

	/********************************************/
	// Quantize (in-place) the core elements
	/********************************************/

	for (int i = left; i < right; ++i) {
	    // If q = 0 there's no need to store anything quantized, not even the sign
	    // If q = 63, values are kept as they are and we forget about quantization
	    if (q > 0 and q < 63) {
		unsigned long int to_write = 0;
		if (chunk_size > 1)
		    // The following min() prevents overflowing the q-bit representation when converting double -> long int
		    to_write = min(((1UL << q) - 1), (unsigned long int)
				   roundl((sorting[i].first - chunk_min) / (chunk_max - chunk_min) * ((1UL << q) - 1)));
		to_write |= (c[sorting[i].second] < 0) * (1UL << q);
		c[sorting[i].second] = static_cast < double >(to_write);
	    }
	}

	/********************************************/
	// Save mask and compute RLE+Huffman out of it
	/********************************************/

	for (int i = left; i < right; ++i) {
	    long int index = sorting[i].second;
	    encoding_mask[index] = chunk_num;
	    // We use this loop also to store the needed quantization bits per factor column
	    int x = index % s[0];
	    int y = index % (s[0] * s[1]) / s[0];
	    Us_q[0][x] = max(Us_q[0][x], q);
	    Us_q[1][y] = max(Us_q[1][y], q);
	    if (n == 3) {
		int z = index / (s[0] * s[1]);
		Us_q[2][z] = max(Us_q[2][z], q);
	    } else if (n == 4) {
		int z = index % (s[0] * s[1] * s[2]) / (s[0] * s[1]);
		int t = index / (s[0] * s[1] * s[2]);
		Us_q[2][z] = max(Us_q[2][z], q);
		Us_q[3][t] = max(Us_q[3][t], q);
	    }
	}

	vector < char >mask;
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
		mask.push_back(mask_wbyte);
		mask_wbyte = 0;
		mask_wbit = 7;
	    }
	}
	if (mask_wbit < 7)
	    mask.push_back(mask_wbyte);
	vector < char >encoding;
	encode(mask, encoding);
	chunk_info ci;
	ci.compressed_size = encoding.size();
	ci.minimum = chunk_min;
	ci.maximum = chunk_max;
	output_stream.write(reinterpret_cast < char *>(&ci), sizeof(chunk_info));
	std::copy(encoding.begin(), encoding.end(), std::ostreambuf_iterator < char >(output_stream));

	if (verbose) {
	    int coeff_bits = 0;
	    if (q > 0)
		coeff_bits = (q + 1) * (right - left);	// The "+1" is for the sign
	    cout << "Encoded chunk " << chunk_num << ", min=" << chunk_min << ", max=" << chunk_max << ", cbits=" << coeff_bits << ", q=" << int (q) << ", bits=[" << left << "," << right << "), size=" << right - left << endl << flush;
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
	for (int i = 0; i < n; ++i) {
	    for (int j = 0; j < s[i]; ++j)
		cout << " " << int (Us_q[i][j]);
	    cout << endl;
	}
    }

    if (verbose)
	cout << "Encoding factor matrices... " << flush;
    for (int i = 0; i < n; ++i)
	encode_factor(Us[i], s[i], Us_q[i], output_stream);
    if (verbose)
	cout << "Done" << endl << flush;

    /********************************************/
    // Save the core encoding
    /********************************************/

    if (verbose)
	cout << "Saving core encoding... " << flush;
    char core_quant_wbyte = 0;
    char core_quant_wbit = 7;
    for (int i = 0; i < size; ++i) {
	chunk_num = encoding_mask[i];
	char q = chunk_num - 1;
	if (q > 0) {
	    for (long int j = q; j >= 0; --j) {
		core_quant_wbyte |= ((static_cast < unsigned long int >(c[i]) >> j) &1UL) << core_quant_wbit;
		core_quant_wbit--;
		if (core_quant_wbit < 0) {
		    output_stream.write(&core_quant_wbyte, sizeof(char));
		    core_quant_wbyte = 0;
		    core_quant_wbit = 7;
		}
	    }
	}
    }
    if (core_quant_wbit < 7)
	output_stream.write(&core_quant_wbyte, sizeof(char));
    if (verbose)
	cout << "Done" << endl << flush;
    delete[]c;

    // Finally: write the number of chunks, which we now know
    output_stream.seekp(n * sizeof(int) + 2);
    unsigned char n_chunks = q;
    output_stream.write(reinterpret_cast < char *>(&n_chunks), sizeof(char));
    output_stream.close();

    /***********************************************/
    // Tar+gzip the final result and compute the bpv
    /***********************************************/

    stringstream ss;
    ss << "tar -czf " << compressed_file << " " << "tthresh-tmp/";
    string command(ss.str());
    _ = system(command.c_str());
    ifstream bpv_stream(compressed_file.c_str(), ios::in | ios::binary);
    streampos beginning = bpv_stream.tellg();
    bpv_stream.seekg(0, ios::end);
    long int newbits = (bpv_stream.tellg() - beginning) * 8;
    cout << "oldbits = " << size * type_size * 8L << ", newbits = " << newbits << ", compressionrate = " << size * type_size * 8L / double (newbits)
    << ", bpv = " << newbits / double (size) << endl << flush;
    bpv_stream.close();

    return data;
}
