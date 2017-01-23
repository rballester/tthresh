# tthresh

## Multidimensional Compression Using the Tucker Tensor Decomposition

This is an improved **C++ implementation** evolved from the thresholding compression method for 3D grid data described in the paper [*Lossy Volume Compression Using Tucker Truncation and Thresholding*](http://www.ifi.uzh.ch/en/vmml/publications/lossycompression.html). This compressor works for **3 dimensions and above**. For more details on the Tucker transform and tensor-based volume compression, check out our [slides](http://www.ifi.uzh.ch/dam/jcr:00000000-73a0-83b8-ffff-ffffd48b8a42/tensorapproximation.pdf).

### Download

```  
git clone https://github.com/rballester/tthresh.git
```

(or as a [zip file](https://github.com/rballester/tthresh/archive/master.zip)).

### Compilation

Use CMake to generate an executable ```tthresh```:

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

### Usage

**Compression**:

```
tthresh -i <dataset> <options> -c <compressed dataset>
```

**Decompression**:

```
tthresh -c <compressed dataset> -o <decompressed dataset>
```

**Compression + decompression** (this will print both the compression rate and the achieved accuracy):

```
tthresh -i dataset <options> -c <compressed dataset> -o <decompressed dataset>
```

The target accuracy can be specified either as **relative error**, **RMSE** or **PSNR**. To get more info on the available options, run ```tthresh -h```.

You are free to **use and modify** the code as long as you mention the origin. If you use it for research purposes, **please cite the paper**:

```@article{BP:15, year={2015}, issn={0178-2789}, journal={The Visual Computer}, title={Lossy volume compression using {T}ucker truncation and thresholding}, publisher={Springer Berlin Heidelberg}, keywords={Tensor approximation; Data compression; Higher-order decompositions; Tensor rank reduction; Multidimensional data encoding}, author={Ballester-Ripoll, Rafael and Pajarola, Renato}, pages={1-14}}```

### Visual Results (click to enlarge)

"Channel" turbulence timestep from the [Johns Hopkins Turbulence Database](http://turbulence.pha.jhu.edu/newcutout.aspx):

[<img src="https://github.com/rballester/tthresh/blob/master/images/channel.png" width="1024">](https://raw.githubusercontent.com/rballester/tthresh/master/images/channel.png?token=AB6uqZlifcbJR2J7cQikLyVwdXVHaWoFks5Yi1FDwA%3D%3D)

"Foot" from the [TC18 Repository](http://www.tc18.org/code_data_set/3D_images.php):

[<img src="https://github.com/rballester/tthresh/blob/master/images/foot.png" width="1024">](https://raw.githubusercontent.com/rballester/tthresh/master/images/foot.png?token=AB6uqQEiEwrYO9kAzUUJJ1VPXm8StfqMks5Yi1HpwA%3D%3D)

"Boston teapot" from the [TC18 Repository](http://www.tc18.org/code_data_set/3D_images.php):

[<img src="https://github.com/rballester/tthresh/blob/master/images/boston_teapot.png" width="1024">](https://raw.githubusercontent.com/rballester/tthresh/master/images/boston_teapot.png?token=AB6uqVD8-REH39F_alYcSuGNUbvA_qrwks5Yi1HuwA%3D%3D)

### Acknowledgment

Special thanks to [Peter G. Lindstrom](http://people.llnl.gov/pl), author of the [zfp and fpzip compressors](http://computation.llnl.gov/projects/floating-point-compression), for sparking fruitful discussions and ideas on how to improve the Tucker compressor.

### Why Tucker?

Tensor-based compression is **non-local**, in the sense that all compressed coefficients contribute to the transformation of each individual voxel (in contrast to e.g. wavelet transforms or JPEG for images, which uses a localized DCT transform). This can be computationally demanding but decorrelates the data at **all spatial scales**, which has **several advantages**:

- Very competitive compression quality
- Fine bit-rate granularity is possible
- Smooth degradation at high compression (in particular, no blocking artifacts or temporal glitches)
- Ability to downsample and subsample in the compressed domain
