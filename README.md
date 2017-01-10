# tthresh

##3D and 4D compression using the Tucker tensor decomposition

This is an improved **C++ implementation** evolved from the thresholding compression method for 3D grid data described in the paper [*Lossy Volume Compression Using Tucker Truncation and Thresholding*](http://www.ifi.uzh.ch/en/vmml/publications/lossycompression.html). For more details on the Tucker transform and tensor-based volume compression, check out our [slides](http://www.ifi.uzh.ch/dam/jcr:00000000-73a0-83b8-ffff-ffffd48b8a42/tensorapproximation.pdf).

### Download

```  
git clone https://github.com/rballester/tthresh.git
```

or as a [zip file](https://github.com/rballester/tthresh/archive/master.zip).

### Compilation

Use CMake to generate an executable ```build/tthresh```:

```
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

### Usage

**Compression**:

```
./tthresh -i <dataset> <options> -c <compressed dataset>
```

**Decompression**:

```
./tthresh -c <compressed dataset> -o <decompressed dataset>
```

**Compression + decompression** (this will print both the compression rate and the achieved accuracy):

```
./tthresh -i dataset <options> -c <compressed dataset> -o <decompressed dataset>
```

The target accuracy can be specified either as **relative error**, **RMSE** or **PSNR**. To get more info the available options, run ```./tthresh -h```.

For example, ```./tthresh -i bonsai.raw -c bonsai.compressed -o bonsai.decompressed -t uchar -s 256 256 256 -r 2``` yields about 1.95 RMSE and 1:25 compression rate on the [bonsai data set](http://www.tc18.org/code_data_set/3D_greyscale/bonsai.raw.gz). Left image is a slice from the original, right one from the reconstructed: 

<img src="https://github.com/rballester/tthresh/tree/master/images/original_vs_reconstructed.png" width="512">

You are free to **use and modify** the code as long as you mention the origin. If you use it for research purposes, **please cite the paper**:

```@article{BP:15, year={2015}, issn={0178-2789}, journal={The Visual Computer}, title={Lossy volume compression using {T}ucker truncation and thresholding}, publisher={Springer Berlin Heidelberg}, keywords={Tensor approximation; Data compression; Higher-order decompositions; Tensor rank reduction; Multidimensional data encoding}, author={Ballester-Ripoll, Rafael and Pajarola, Renato}, pages={1-14}}```

### Acknowledgment

Special thanks to [Peter G. Lindstrom](http://people.llnl.gov/pl), author of the [zip and fpzip compressors](http://computation.llnl.gov/projects/floating-point-compression), for sparking fruitful discussions and ideas on how to improve the Tucker compressor.

### Why Tucker?

Tensor-based compression is **non-local**, in the sense that all compressed coefficients contribute to the transformation of each individual voxel (in contrast to e.g. wavelet transforms or JPEG for images, which uses a localized DCT transform). This can be computationally demanding but decorrelates the data at all spatial scales, thus achieving **very competitive compression rates**.