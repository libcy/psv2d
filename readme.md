# psv2d

This is the pure finite difference version of [psmfdm](https://github.com/libcy/psmfdm) with a new matlab plotting tool (waveform.m for plotting seismograms and snapshot.m for visualizing wavefield propagation). Eliminating Fourier transform calculations makes the program 5x-10x faster than the hybrid psv/fdm version. Paremeters are stored in column-major matrices which is compatible with CUBLAS Library. A benchmark shows a ~300x speedup compared to its CPU version(2048*1024 grids, Nvidia GTX 970, Intel core i5-4570)

* Seismograms of example model
  ![speedup](https://raw.githubusercontent.com/libcy/psv2d/master/img/seismogram.png)

* Wavefield snapshot of example model
  ![seismogram](https://raw.githubusercontent.com/libcy/psv2d/master/img/snapshot.png)
