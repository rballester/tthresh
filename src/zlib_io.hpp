#ifndef __ZLIB_IO_HPP__
#define __ZLIB_IO_HPP__

/* Parts are taken from zpipe.c (example of proper use of zlib's inflate() and deflate())
   Not copyrighted -- provided to the public domain
   Version 1.4  11 December 2005  Mark Adler */

/* Version history:
   1.0  30 Oct 2004  First version
   1.1   8 Nov 2004  Add void casting for unused return values
                     Use switch statement for inflate() return values
   1.2   9 Nov 2004  Add assertions to document zlib guarantees
   1.3   6 Apr 2005  Remove incorrect assertion in inf()
   1.4  11 Dec 2005  Add hack to avoid MSDOS end-of-line conversions
                     Avoid some compiler warnings for input and output buffers
 */

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

int min(int a, int b) {
    if (a < b) return a;
    return b;
}

struct {
    FILE *source, *dest;
    unsigned char zlib_buf[CHUNK];
    int zlib_bufsize = 0; // How many elements are there in the buffer right now
    int ret;
    z_stream strm;
    unsigned char out[CHUNK];
} zs; // I/O state

int deflate_chunk(unsigned long int bytes_to_write, int flush)
{
    zs.strm.avail_in = bytes_to_write;
    zs.strm.next_in = zs.zlib_buf;

    unsigned have;
    do {
        zs.strm.avail_out = CHUNK;
        zs.strm.next_out = zs.out;
        zs.ret = deflate(&(zs.strm), flush);    /* no bad return value */
        assert(zs.ret != Z_STREAM_ERROR);  /* state not clobbered */
        have = CHUNK - zs.strm.avail_out;
        if (fwrite(zs.out, 1, have, zs.dest) != have || ferror(zs.dest)) {
            (void)deflateEnd(&(zs.strm));
            return Z_ERRNO;
        }
    } while (zs.strm.avail_out == 0);
    assert(zs.strm.avail_in == 0);     /* all input will be used */
}

int open_zlib_write_stream(string output_file)
{
    SET_BINARY_MODE(output_file.c_str());
    zs.dest = fopen(output_file.c_str(), "w");
    zs.strm.zalloc = Z_NULL;
    zs.strm.zfree = Z_NULL;
    zs.strm.opaque = Z_NULL;
    zs.ret = deflateInit(&(zs.strm), Z_DEFAULT_COMPRESSION);
    return zs.ret;
}

int write_zlib_stream(unsigned char *buf, unsigned long int bytes_to_write)
{
    while (bytes_to_write > 0) {
        unsigned long int to_copy = min(bytes_to_write, CHUNK-zs.zlib_bufsize);
        memcpy(zs.zlib_buf + zs.zlib_bufsize, buf, to_copy);
        zs.zlib_bufsize += to_copy;
        buf += to_copy;
        bytes_to_write -= to_copy;
        if (zs.zlib_bufsize == CHUNK) { // The buffer is full
            deflate_chunk(CHUNK, Z_NO_FLUSH);
            zs.zlib_bufsize = 0; // Empty the buffer
        }
    }
}

int close_zlib_write_stream() {
    deflate_chunk(zs.zlib_bufsize, Z_FINISH);
    assert(zs.ret == Z_STREAM_END);        /* stream will be complete */
    /* clean up and return */
    (void)deflateEnd(&(zs.strm));
    fclose(zs.dest);
    return Z_OK;
}

/* Compress from file source to file dest until EOF on source.
   def() returns Z_OK on success, Z_MEM_ERROR if memory could not be
   allocated for processing, Z_STREAM_ERROR if an invalid compression
   level is supplied, Z_VERSION_ERROR if the version of zlib.h and the
   version of the library linked do not match, or Z_ERRNO if there is
   an error reading or writing the files. */
int def(FILE *source, FILE *dest, int level)
{
    int ret, flush;
    unsigned have;
    z_stream strm;
    unsigned char in[CHUNK];
    unsigned char out[CHUNK];

    /* allocate deflate state */
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    ret = deflateInit(&strm, level);
    if (ret != Z_OK)
        return ret;

    /* compress until end of file */
    do {
        strm.avail_in = fread(in, 1, CHUNK, source);
        if (ferror(source)) {
            (void)deflateEnd(&strm);
            return Z_ERRNO;
        }
        flush = feof(source) ? Z_FINISH : Z_NO_FLUSH;
        strm.next_in = in;

        /* run deflate() on input until output buffer not full, finish
           compression if all of source has been read in */
        do {
            strm.avail_out = CHUNK;
            strm.next_out = out;
            ret = deflate(&strm, flush);    /* no bad return value */
            assert(ret != Z_STREAM_ERROR);  /* state not clobbered */
            have = CHUNK - strm.avail_out;
            if (fwrite(out, 1, have, dest) != have || ferror(dest)) {
                (void)deflateEnd(&strm);
                return Z_ERRNO;
            }
        } while (strm.avail_out == 0);
        assert(strm.avail_in == 0);     /* all input will be used */

        /* done when last data in file processed */
    } while (flush != Z_FINISH);
    assert(ret == Z_STREAM_END);        /* stream will be complete */

    /* clean up and return */
    (void)deflateEnd(&strm);
    return Z_OK;
}

int open_zlib_read_stream(string input_file)
{
    zs.dest = fopen(input_file.c_str(), "r");
    /* allocate inflate state */
    zs.strm.zalloc = Z_NULL;
    zs.strm.zfree = Z_NULL;
    zs.strm.opaque = Z_NULL;
    zs.strm.avail_in = 0;
    zs.strm.next_in = Z_NULL;
    zs.ret = inflateInit(&(zs.strm));
    zs.zlib_bufsize = 0;
    return zs.ret;
}

/* Decompress from file source to file dest until stream ends or EOF.
   inf() returns Z_OK on success, Z_MEM_ERROR if memory could not be
   allocated for processing, Z_DATA_ERROR if the deflate data is
   invalid or incomplete, Z_VERSION_ERROR if the version of zlib.h and
   the version of the library linked do not match, or Z_ERRNO if there
   is an error reading or writing the files. */
int inf(FILE *source, FILE *dest)
{
    int ret;
    unsigned have;
    z_stream strm;
    unsigned char in[CHUNK];
    unsigned char out[CHUNK];

    /* allocate inflate state */
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    strm.avail_in = 0;
    strm.next_in = Z_NULL;
    ret = inflateInit(&strm);
    if (ret != Z_OK)
        return ret;

    /* decompress until deflate stream ends or end of file */
    do {
        strm.avail_in = fread(in, 1, CHUNK, source);
        if (ferror(source)) {
            (void)inflateEnd(&strm);
            return Z_ERRNO;
        }
        if (strm.avail_in == 0)
            break;
        strm.next_in = in;

        /* run inflate() on input until output buffer not full */
        do {
            strm.avail_out = CHUNK;
            strm.next_out = out;
            ret = inflate(&strm, Z_NO_FLUSH);
            assert(ret != Z_STREAM_ERROR);  /* state not clobbered */
            switch (ret) {
            case Z_NEED_DICT:
                ret = Z_DATA_ERROR;     /* and fall through */
            case Z_DATA_ERROR:
            case Z_MEM_ERROR:
                (void)inflateEnd(&strm);
                return ret;
            }
            have = CHUNK - strm.avail_out;
            if (fwrite(out, 1, have, dest) != have || ferror(dest)) {
                (void)inflateEnd(&strm);
                return Z_ERRNO;
            }
        } while (strm.avail_out == 0);

        /* done when inflate() says it's done */
    } while (ret != Z_STREAM_END);

    /* clean up and return */
    (void)inflateEnd(&strm);
    return ret == Z_STREAM_END ? Z_OK : Z_DATA_ERROR;
}

#endif // ZLIB_IO_HPP
