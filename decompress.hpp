#include "tthresh.hpp"
#include "tucker.hpp"
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

void decode_factor(MatrixXd & U, int n_columns, ifstream & input_stream)
{

    U = MatrixXd(n_columns, n_columns);

    // First, the matrix's maximum, used for quantization
    double maximum;
    input_stream.read(reinterpret_cast < char *>(&maximum), sizeof(double));

    // Next, the q for each column
    vector < char >U_q(n_columns);
    for (int i = 0; i < n_columns; ++i)
	input_stream.read(&U_q[i], sizeof(char));

    // Finally we dequantize the matrix itself
    char matrix_rbyte;
    char matrix_rbit = -1;
    for (int j = 0; j < n_columns; ++j) {
	for (int i = 0; i < n_columns; ++i) {
	    char q = U_q[j];
	    if (q == 0)
		U(i, j) = 0;
	    else {
		q = min(63, q + 2);
		unsigned long int to_read = 0;
		for (int j = q; j >= 0; --j) {
		    if (matrix_rbit < 0) {
			matrix_rbit = 7;
			input_stream.read(&matrix_rbyte, sizeof(char));
		    }
		    to_read |= ((matrix_rbyte >> matrix_rbit) & 1UL) << j;
		    matrix_rbit--;
		}
		char sign = (to_read >> q) & 1UL;
		to_read &= ~(1UL << q);
		U(i, j) = -(2 * sign - 1) / ((1UL << q) - double (1)) *maximum * double (to_read);
	    }
	}
    }
}

void decompress(string compressed_file, string output_file, double *data, bool verbose, bool debug)
{

    if (verbose)
	cout << endl << "/***** Decompression *****/" << endl << endl << flush;

    int _ = system("rm tthresh-tmp/*");

    stringstream ss;
    ss << "tar -xf " << compressed_file;
    string command(ss.str());
    _ = system(command.c_str());

    // Read output tensor dimensionality, sizes and type
    ifstream input_stream("tthresh-tmp/all", ios::in | ios::binary);
    char n;
    input_stream.read(reinterpret_cast < char *>(&n), sizeof(char));
    vector < int >s(n);
    input_stream.read(reinterpret_cast < char *>(&s[0]), n * sizeof(int));
    long int size = 1;
    for (int i = 0; i < n; ++i)
	size *= s[i];
    if (debug) {
	cout << "Decompressing a tensor of size " << s[0];
	for (int i = 1; i < n; ++i)
	    cout << " x " << s[i];
	cout << "..." << endl;
    }

    char io_type_code;
    input_stream.read(reinterpret_cast < char *>(&io_type_code), sizeof(char));
    char io_type_size;
    if (io_type_code == 0) {
	io_type_size = 1;
    } else if (io_type_code == 1) {
	io_type_size = 4;
    } else if (io_type_code == 2) {
	io_type_size = 4;
    } else {
	io_type_size = 8;
    }
    unsigned char n_chunks;
    input_stream.read(reinterpret_cast < char *>(&n_chunks), sizeof(char));
    double minimums[n_chunks];
    double maximums[n_chunks];

    vector < int >encoding_mask(size, 0);
    for (int chunk_num = 1; chunk_num <= n_chunks; ++chunk_num) {

	chunk_info ci;
	input_stream.read(reinterpret_cast < char *>(&ci), sizeof(chunk_info));
	minimums[chunk_num - 1] = ci.minimum;
	maximums[chunk_num - 1] = ci.maximum;

	vector < char >decoded;
	decode(input_stream, ci.compressed_size, decoded);
	long int ind = 0;
	for (unsigned int i = 0; i < decoded.size()and ind < size; ++i) {	// This can be more efficient...
	    for (char chunk_rbit = 7; chunk_rbit >= 0 and ind < size; --chunk_rbit) {
		while (encoding_mask[ind] > 0)
		    ind++;
		if ((decoded[i] >> chunk_rbit) & 1) {
		    encoding_mask[ind] = chunk_num;
		}
		ind++;
	    }
	}
	if (verbose)
	    cout << "Decoded chunk " << chunk_num << ", mask has " << decoded.size() * 8 << " bits, q=" << int (chunk_num - 1) << endl << flush;
    }

    // Read factor matrices
    if (verbose)
	cout << "Decoding factor matrices... " << flush;
    vector < MatrixXd > Us(n);
    for (int i = 0; i < n; ++i)
	decode_factor(Us[i], s[i], input_stream);
    if (verbose)
	cout << "Done" << endl << flush;

    // Recover the core: read all remaining data
    double *c = new double[size];
    std::vector < char >core_quant_buffer((istreambuf_iterator < char >(input_stream)), istreambuf_iterator < char >());
    input_stream.close();
    int core_quant_rbyte = 0;
    char core_quant_rbit = 7;

    for (int i = 0; i < size; ++i) {
	int chunk_num = encoding_mask[i];
	double chunk_min = minimums[chunk_num - 1];
	double chunk_max = maximums[chunk_num - 1];
	char q = chunk_num - 1;

	if (q > 0) {
	    char sign = 0;
	    unsigned long int quant = 0;
	    for (long int j = q; j >= 0; --j) {
		if (j == q and q < 63)
		    sign = ((core_quant_buffer[core_quant_rbyte] >> core_quant_rbit) & 1UL);
		else
		    quant |= ((core_quant_buffer[core_quant_rbyte] >> core_quant_rbit) & 1UL) << j;
		core_quant_rbit--;
		if (core_quant_rbit < 0) {
		    core_quant_rbyte++;
		    core_quant_rbit = 7;
		}
	    }
	    if (q == 63) {
		c[i] = static_cast < double >(quant);
	    } else {
		double dequant;
		dequant = quant / ((1UL << q) - 1.) * (chunk_max - chunk_min) + chunk_min;
		c[i] = -(sign * 2 - 1) * dequant;
	    }
	} else
	    c[i] = 0;
    }
    input_stream.close();

    if (verbose)
	cout << "Reconstructing tensor... " << flush;
    double *r = c;
    hosvd(r, s, Us, false, verbose);
    if (verbose)
	cout << "Done" << endl << flush;

    ofstream output_stream(output_file.c_str(), ios::out | ios::binary);
    long int buf_elems = 1 << 20;
    char *buffer = new char[io_type_size * buf_elems];
    long int buffer_wpos = 0;
    double sse = 0;
    double input_norm = 0;
    double input_min = std::numeric_limits < double >::max();
    double input_max = std::numeric_limits < double >::min();
    for (long int i = 0; i < size; ++i) {
	if (io_type_code == 0) {
	    reinterpret_cast < unsigned char *>(buffer)[buffer_wpos] = abs(r[i]);
	} else if (io_type_code == 1) {
	    reinterpret_cast < int *>(buffer)[buffer_wpos] = r[i];
	} else if (io_type_code == 2) {
	    reinterpret_cast < float *>(buffer)[buffer_wpos] = r[i];
	} else {
	    reinterpret_cast < double *>(buffer)[buffer_wpos] = r[i];
	}
	buffer_wpos++;
	if (buffer_wpos == buf_elems) {
	    buffer_wpos = 0;
	    output_stream.write(buffer, io_type_size * buf_elems);
	}
	if (data != NULL) {
	    input_norm += data[i] * data[i];
	    sse += (data[i] - r[i]) * (data[i] - r[i]);
	    input_min = min(input_min, data[i]);
	    input_max = max(input_max, data[i]);
	}
    }
    if (buffer_wpos > 0)
	output_stream.write(buffer, io_type_size * buffer_wpos);
    delete[]buffer;
    output_stream.close();

    if (data != NULL) {		// If the uncompressed input is available, we compute the error statistics
	input_norm = sqrt(input_norm);
	double eps = sqrt(sse) / input_norm;
	double rmse = sqrt(sse / size);
	double psnr = 20 * log10((input_max - input_min) / (2 * rmse));
	cout << "eps = " << eps << ", rmse = " << rmse << ", psnr = " << psnr << endl;
    }

    delete[]r;
}
