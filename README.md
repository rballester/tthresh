# tthresh

## Multidimensional Compression Using the Tucker Tensor Decomposition

This is an **open-source C++ implementation** written by [Rafael Ballester-Ripoll](http://www.ifi.uzh.ch/en/vmml/people/current-staff/ballester.html) (rballester@ifi.uzh.ch) of the compressor developed in [TTHRESH: Tensor Compression for Multidimensional Visual Data (R. Ballester-Ripoll, P. Lindstrom and R. Pajarola)](https://arxiv.org/pdf/1806.05952.pdf). It is intended for Cartesian grid data of **3 or more dimensions**. It leverages the higher-order singular value decomposition (HOSVD), a generalization of the SVD to 3 and more dimensions, together with adaptive quantization, run-length and arithmetic coding to store the HOSVD transform coefficients' relative positions as sorted by their absolute magnitude.

If you use TTHRESH for a scientific publication, please cite one or both of these papers:

- [*TTHRESH: Tensor Compression for Multidimensional Visual Data*](https://arxiv.org/abs/1806.05952): ```@article{BLP:18, author = {{Ballester-Ripoll}, R. and {Lindstrom}, P. and {Pajarola}, R.}, title = "{{TTHRESH}: Tensor Compression for Multidimensional Visual Data}", journal = {ArXiv e-prints}, archivePrefix = "arXiv", eprint = {1806.05952}, primaryClass = "cs.GR", keywords = {Computer Science - Graphics}, year = 2018, month = jun, url = {https://arxiv.org/abs/1806.05952}}```
- [*Lossy Volume Compression Using Tucker Truncation and Thresholding*](http://www.ifi.uzh.ch/en/vmml/publications/lossycompression.html): ```@article{BP:15, year={2015}, issn={0178-2789}, journal={The Visual Computer}, title={Lossy volume compression using {T}ucker truncation and thresholding}, publisher={Springer Berlin Heidelberg}, keywords={Tensor approximation; Data compression; Higher-order decompositions; Tensor rank reduction; Multidimensional data encoding}, author={Ballester-Ripoll, Rafael and Pajarola, Renato}, pages={1-14}}```

For more information on the Tucker transform and tensor-based volume compression, check out our [slides](http://www.ifi.uzh.ch/dam/jcr:00000000-73a0-83b8-ffff-ffffd48b8a42/tensorapproximation.pdf).

### Visual Results (click to enlarge)

"Isotropic fine" turbulence timestep (512x512x512, 32-bit float) from the [Johns Hopkins Turbulence Database](http://turbulence.pha.jhu.edu/newcutout.aspx):

[<img src="https://github.com/rballester/tthresh/blob/master/images/isotropic_fine.png" width="1024" title="Isotropic fine">](https://github.com/rballester/tthresh/raw/master/images/isotropic_fine.png)

"Foot" (256x256x256, 8-bit unsigned int) from the [TC18 Repository](http://www.tc18.org/code_data_set/3D_images.php):

[<img src="https://github.com/rballester/tthresh/blob/master/images/foot.png" width="1024" title="Foot">](https://github.com/rballester/tthresh/raw/master/images/foot.png)

### Download

```  
git clone https://github.com/rballester/tthresh.git
```

(or as a [zip file](https://github.com/rballester/tthresh/archive/master.zip)).

### Compilation

Use [CMake](https://cmake.org/) to generate an executable ```tthresh```:

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

The target accuracy can be specified either as relative error (```-e```), RMSE (```-r```) or PSNR (```-p```).

**Toy example**:

A toy data set (a 3D sphere) is included in the ```data/``` folder. You can test the compressor with it as follows:

```
tthresh -i data/3D_sphere_64_uchar.raw -t uchar -s 64 64 64 -p 30 -c data/comp.raw -o data/decomp.raw
```

### Extra Features

- Use ```-a``` to reconstruct only the data set's bounding box.
- Use ```-k``` when compressing a file to skip its k leading bytes.
- Use [NumPy-like notation](https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#basic-slicing-and-indexing) immediately after ```-o``` to decimate the data while decompression. For example, ```-o :: :: 0``` will reconstruct only the first z-slice of a volume, ```-o ::2 ::2 ::2``` will decompress only every other voxel along all dimensions, and ```-o ll4 ll4 ll4``` will perform [Lanczos downsampling](https://en.wikipedia.org/wiki/Lanczos_resampling) by a factor of 4. Some result examples for x2 decimation:

[<img src="https://github.com/rballester/tthresh/blob/master/images/decimation.png" width="512" title="Foot">](https://github.com/rballester/tthresh/raw/master/images/decimation.png)

To get more info on the available options, run ```tthresh -h```.

### Acknowledgments

This work was partially supported by the [UZH Forschungskredit "Candoc"](http://www.researchers.uzh.ch/en/funding/phd/fkcandoc.html), grant number FK-16-012. I also thank [Enrique G. Paredes](http://www.ifi.uzh.ch/en/vmml/people/current-staff/egparedes.html) for his help with CMake compilation issues.

### Why Tucker?

Tensor-based compression is non-local, in the sense that all compressed coefficients contribute to the reconstruction of each individual voxel (in contrast to e.g. wavelet transforms or JPEG for images, which uses a localized DCT transform). This can be computationally demanding but decorrelates the data at all spatial scales, which has several advantages:

- Very competitive **compression quality**
- Fine bit-rate **granularity**
- **Smooth degradation** at high compression (in particular, no blocking artifacts or temporal glitches)
- Ability to **downsample** in the compressed domain
