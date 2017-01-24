#ifndef __ZLIB_IO_HPP__
#define __ZLIB_IO_HPP__

// Parts are taken from http://www.zlib.net/zlib_how.html
// (example of proper use of zlib's inflate() and deflate())

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "zlib.h"

// Avoids corruption of the input and output data on Windows/MS-DOS systems
#if defined(MSDOS) || defined(OS2) || defined(WIN32) || defined(__CYGWIN__)
#  include <fcntl.h>
#  include <io.h>
#  define SET_BINARY_MODE(file) setmode(fileno(file), O_BINARY)
#else
#  define SET_BINARY_MODE(file)
#endif

#define CHUNK (1<<18)

ind_t min(ind_t a, ind_t b) {
    return (a < b) ? a : b;
}

struct {
    unsigned char rbyte, wbyte;
    char rbit, wbit;
    FILE *file; // File handle to read/write from/to
    unsigned char inout[CHUNK]; // Buffer to write the results of inflation/deflation
    unsigned char buf[CHUNK]; // Buffer used for the read/write operations
    int bufstart = 0;
    int bufend = 0;
    ind_t total_written_bytes = 0; // Used to compute the final file size
    z_stream strm;
} zs; // Read/write state for zlib interfacing

int deflate_chunk(ind_t bytes_to_write, int flush)
{
    zs.strm.avail_in = bytes_to_write;
    zs.strm.next_in = zs.buf;
    int ret;
    do {
        zs.strm.avail_out = CHUNK;
        zs.strm.next_out = zs.inout;
        ret = deflate(&(zs.strm), flush);    /* no bad return value */
        if (ret == Z_STREAM_ERROR)  /* state not clobbered */
            throw ret;
        unsigned have = CHUNK - zs.strm.avail_out;
        zs.total_written_bytes += have;
        if (fwrite(zs.inout, 1, have, zs.file) != have || ferror(zs.file)) {
            (void)deflateEnd(&(zs.strm));
            throw Z_ERRNO;
        }
    } while (zs.strm.avail_out == 0);
    return ret;
}

void open_zlib_write(string output_file)
{
    SET_BINARY_MODE(output_file.c_str());
    zs.wbyte = 0;
    zs.wbit = 7;
    zs.file = fopen(output_file.c_str(), "w");
    zs.strm.zalloc = Z_NULL;
    zs.strm.zfree = Z_NULL;
    zs.strm.opaque = Z_NULL;
    int ret = deflateInit(&(zs.strm), Z_DEFAULT_COMPRESSION);
    if (ret != Z_OK)
        throw ret;
}

void zlib_write_stream(unsigned char *buf, ind_t bytes_to_write)
{
    assert(zs.wbit == 7); // One shouldn't want to write a stream in the middle of individual bits
    while (bytes_to_write > 0) {
        unsigned long int to_copy = min(bytes_to_write, CHUNK-zs.bufend);
        memcpy(zs.buf + zs.bufend, buf, to_copy);
        zs.bufend += to_copy;
        buf += to_copy;
        bytes_to_write -= to_copy;
        if (zs.bufend == CHUNK) { // The buffer is full
            deflate_chunk(CHUNK, Z_NO_FLUSH); // Compress and write CHUNK bytes
            zs.bufend = 0; // Empty the buffer
        }
    }
}

void close_zlib_write() {
    int ret = deflate_chunk(zs.bufend, Z_FINISH);
    if (ret != Z_STREAM_END) /* stream will be complete */
        throw ret;
    /* clean up and return */
    (void)deflateEnd(&(zs.strm));
    fclose(zs.file);
}

void zlib_open_wbit() {
    zs.wbyte = 0;
    zs.wbit = 7;
}

void inline zlib_write_bit(char bit)
{
    zs.wbyte |= bit << zs.wbit;
    zs.wbit--;
    if (zs.wbit < 0) {
        zlib_write_stream(reinterpret_cast < unsigned char *> (&zs.wbyte), sizeof(zs.wbyte));
        zs.wbyte = 0;
        zs.wbit = 7;
    }
}

void zlib_close_wbit() {
    // Write any reamining bits
    if (zs.wbit < 7)
        zlib_write_stream(reinterpret_cast < unsigned char *> (&zs.wbyte), sizeof(zs.wbyte));
}


// If there are no more bytes to inflate, it reads CHUNK bytes (or as many as there are left)
// Then, it inflates up to CHUNK bytes
// It sets zs.bufstart to 0 and zs.bufend to the number of inflated bytes
void inflate_chunk()
{
        if (zs.strm.avail_out > 0) { // If last time we inflated the input buffer ran out, it's time to refill it
            zs.strm.avail_in = fread(zs.inout, 1, CHUNK, zs.file);
            if (ferror(zs.file)) {
                (void)inflateEnd(&zs.strm);
                throw Z_ERRNO;
            }
            if (zs.strm.avail_in == 0) {
                cout << "Error: zlib input file stream finished unexpectedly soon" << endl;
                exit(1);
            }
            zs.strm.next_in = zs.inout;
        }

        zs.strm.avail_out = CHUNK;
        zs.strm.next_out = zs.buf;
        int ret = inflate(&zs.strm, Z_NO_FLUSH);
        if (ret == Z_NEED_DICT)
            ret = Z_DATA_ERROR;
        if (ret == Z_MEM_ERROR) {
            (void)inflateEnd(&zs.strm);
            throw ret;
        }
        zs.bufstart = 0;
        zs.bufend = CHUNK - zs.strm.avail_out;
}

void open_zlib_read(string input_file)
{
    SET_BINARY_MODE(input_file.c_str());
    zs.file = fopen(input_file.c_str(), "r");
    // allocate inflate state
    zs.strm.zalloc = Z_NULL;
    zs.strm.zfree = Z_NULL;
    zs.strm.opaque = Z_NULL;
    zs.strm.avail_in = 0;
    zs.strm.next_in = Z_NULL;
    zs.strm.avail_out = CHUNK;
    zs.bufstart = 0;
    zs.bufend = 0;
    int ret = inflateInit(&(zs.strm));
    if (ret != Z_OK)
        throw ret;
}

void zlib_read_stream(unsigned char *buf, unsigned long int bytes_to_read)
{
    while (bytes_to_read > 0) {
        if (zs.bufstart == zs.bufend) // The buffer is empty
            inflate_chunk(); // Fill the buffer
        unsigned long int to_copy = min(bytes_to_read, zs.bufend-zs.bufstart);
        memcpy(buf, zs.buf + zs.bufstart, to_copy);
        zs.bufstart += to_copy;
        buf += to_copy;
        bytes_to_read -= to_copy;
    }
}

void close_zlib_read()
{
    /* clean up */
    (void)inflateEnd(&zs.strm);
    fclose(zs.file);
}

void zlib_open_rbit() {
    zs.rbyte = 0;
    zs.rbit = -1;
}

char inline zlib_read_bit() {
    if (zs.rbit < 0) {
        zlib_read_stream(reinterpret_cast < unsigned char *>(&zs.rbyte), sizeof(zs.rbyte));
        zs.rbit = 7;
    }
    return (zs.rbyte >> zs.rbit--)&1;
}

#endif // ZLIB_IO_HPP
