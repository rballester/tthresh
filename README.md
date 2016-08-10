# tthresh
##C++ compressor for volumes (3D grid data) using the Tucker decomposition##

This is an improved implementation evolved from the thresholding compression method described in the paper [*Lossy Volume Compression Using Tucker Truncation and Thresholding*](http://www.ifi.uzh.ch/en/vmml/publications/lossycompression.html). For more details on the Tucker transform and tensor-based volume compression, check out my [slides](http://www.ifi.uzh.ch/dam/jcr:00000000-73a0-83b8-ffff-ffffd48b8a42/tensorapproximation.pdf). Current version is **tthresh2**.

## Setup and Usage

You will need:
- CBLAS and CLAPACK
- tar + gzip (this is used to pack everything together into one compressed file)

The core function is ```thresholding_compression(X,metric,target)```, where:

- ```X``` is a volume 
- ```metric``` is "relative error", "rmse", or "psnr"
- ```target``` specifies the desired target accuracy (according to ```metric```)

Try out the code with the example script ```run.m```:

1. Download the [bonsai data set](http://www.tc18.org/code_data_set/3D_greyscale/bonsai.raw.gz) (16MB, 8-bit unsigned int) and unpack it as ```bonsai.raw``` into the project folder.
2. Compile the C++ file ```thresholding/rle_huffman.cpp``` into an executable ```thresholding/rle_huffman```, and ```thresholding/zigzag.cpp``` into ```thresholding/zigzag```.
3. In the MATLAB interpreter, go to the project folder and call ```run```.

For example, ```thresholding_compression(X,'rmse',2)``` yields about 2.1 RMSE and 1:17 compression rate (left image is a slice from the original, right one from the reconstructed): 

<img src="https://github.com/rballester/tucker_compression/blob/master/images/original_vs_reconstructed.jpg" width="512">

You are free to **use and modify** the code. If you use it for a publication, **please cite the paper**:

```@article{BP:15, year={2015}, issn={0178-2789}, journal={The Visual Computer}, title={Lossy volume compression using {T}ucker truncation and thresholding}, publisher={Springer Berlin Heidelberg}, keywords={Tensor approximation; Data compression; Higher-order decompositions; Tensor rank reduction; Multidimensional data encoding}, author={Ballester-Ripoll, Rafael and Pajarola, Renato}, pages={1-14}}```

## Acknowledgments

Special thanks to [Peter G. Lindstrom](http://people.llnl.gov/pl), author of the [zip and fpzip compressors](http://computation.llnl.gov/projects/floating-point-compression), for sparking fruitful discussions on how to improve the Tucker compressor.

## Why Tucker?

Tensor-based compression is **non-local**, in the sense that all compressed coefficients contribute to the transformation of each individual voxel (such as Fourier-based transforms, and in contrast to e.g. wavelet transforms or JPEG for images, which uses a localized DCT transform). This can be computationally demanding but decorrelates the data at all spatial scales, thus achieving **very competitive compression rates**.