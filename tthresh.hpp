#ifndef TTHRESH_HPP
#define TTHRESH_HPP

enum Mode { none_mode, input_mode, compressed_mode, output_mode, io_type_mode, sizes_mode, target_mode };
enum Target { eps, rmse, psnr };

struct chunk_info {
    unsigned int compressed_size;
    double minimum;
    double maximum;
};

#endif
// TTHRESH_HPP
