/*
 * Copyright (c) 2016-2017, Rafael Ballester-Ripoll
 *                          (Visualization and MultiMedia Lab, University of Zurich),
 *                          rballester@ifi.uzh.ch
 *
 * Licensed under LGPLv3.0 (https://github.com/rballester/tthresh/LICENSE)
 */

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

struct {
    uint64_t rbytes, wbytes;
    int8_t rbit, wbit;
    FILE *file; // File handle to read/write from/to
    uint8_t inout[CHUNK]; // Buffer to write the results of inflation/deflation
    uint8_t buf[CHUNK]; // Buffer used for the read/write operations
    int32_t bufstart = 0;
    int32_t bufend = 0;
    size_t total_written_bytes = 0; // Used to compute the final file size
    z_stream strm;
} zs; // Read/write state for zlib interfacing

int32_t deflate_chunk(size_t bytes_to_write, int32_t flush)
{
    zs.strm.avail_in = bytes_to_write;
    zs.strm.next_in = zs.buf;
    int32_t ret;
    do {
        zs.strm.avail_out = CHUNK;
        zs.strm.next_out = zs.inout;
        ret = deflate(&(zs.strm), flush);    /* no bad return value */
        if (ret == Z_STREAM_ERROR)  /* state not clobbered */
            throw ret;
        uint32_t have = CHUNK - zs.strm.avail_out;
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
    zs.wbytes = 0;
    zs.wbit = 7;
    zs.file = fopen(output_file.c_str(), "w");
    zs.strm.zalloc = Z_NULL;
    zs.strm.zfree = Z_NULL;
    zs.strm.opaque = Z_NULL;
    int32_t ret = deflateInit(&(zs.strm), Z_DEFAULT_COMPRESSION);
    if (ret != Z_OK)
        throw ret;
}

void zlib_write_stream(unsigned char *buf, size_t bytes_to_write)
{
    while (bytes_to_write > 0) {
        size_t to_copy = min(bytes_to_write, CHUNK-zs.bufend);
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
    int32_t ret = deflate_chunk(zs.bufend, Z_FINISH);
    if (ret != Z_STREAM_END) /* stream will be complete */
        throw ret;
    /* clean up and return */
    (void)deflateEnd(&(zs.strm));
    fclose(zs.file);
}

void zlib_open_wbit() {
    zs.wbytes = 0;
    zs.wbit = 63;
}

// Assumption: to_write <= 64
void zlib_write_bits(uint64_t bits, char to_write) {
    if (to_write <= zs.wbit+1) {
        zs.wbytes |= bits << (zs.wbit+1-to_write);
        zs.wbit -= to_write;
    }
    else {
        if (zs.wbit > -1)
            zs.wbytes |= bits >> (to_write-(zs.wbit+1));
        zlib_write_stream(reinterpret_cast<unsigned char *> (&zs.wbytes), sizeof(zs.wbytes));
        to_write -= zs.wbit+1;
        zs.wbytes = 0;
        zs.wbytes |= bits << (64-to_write);
        zs.wbit = 63-to_write;
    }
}

void zlib_close_wbit() {
    // Write any reamining bits
    if (zs.wbit < 63)
        zlib_write_stream(reinterpret_cast < unsigned char *> (&zs.wbytes), sizeof(zs.wbytes));
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
        int32_t ret = inflate(&zs.strm, Z_NO_FLUSH);
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

void zlib_read_stream(uint8_t *buf, size_t bytes_to_read)
{
    while (bytes_to_read > 0) {
        if (zs.bufstart == zs.bufend) // The buffer is empty
            inflate_chunk(); // Fill the buffer
        size_t to_copy = min(bytes_to_read, zs.bufend-zs.bufstart);
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
    zs.rbytes = 0;
    zs.rbit = -1;
}

// Assumption: to_read <= BITS
uint64_t zlib_read_bits(char to_read) {
    uint64_t result = 0;
    if (to_read <= zs.rbit+1) {
        result = zs.rbytes << (63-zs.rbit) >> (64-to_read);
        zs.rbit -= to_read;
    }
    else {
        if (zs.rbit > -1)
            result = zs.rbytes << (64-zs.rbit-1) >> (64-to_read);
        zlib_read_stream(reinterpret_cast<uint8_t *> (&zs.rbytes), sizeof(zs.rbytes));
        to_read -= zs.rbit+1;
        result |= zs.rbytes >> (64-to_read);
        zs.rbit = 63-to_read;
    }
    return result;
}

#endif // ZLIB_IO_HPP
