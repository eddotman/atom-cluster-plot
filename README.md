Local Atomic Motif Isolator
===========================

Overview
--------

This repo includes the `LAM` and `LAMisolator` classes, which allow for LAM isolation. The `LAMplotter` class contains some methods for computing histograms and plotting the density isosurfaces via the [Mayavi API](http://docs.enthought.com/mayavi/mayavi/).  

Dependencies
------------

You will need to have `numpy`, `scipy` and `mayavi` installed. Personally, I used [pythonxy](https://code.google.com/p/pythonxy/wiki/Downloads) for the aforementioned dependencies on Windows. If you're on Linux, I'll assume that you know how to `apt-get` or `pip`.  

Additionally, you must install [PeriodicCKDTree](https://github.com/eddotman/periodic_kdtree).  

Usage
-----

Build the `local_atomic_motif.py` file, to ensure that nothing crazy is going on. Depending on the permissions with which you run the script, you may want to manually add a `build/` directory to the same location as the LAM Python files.  

Next, run the `local_atomic_motif_isolator.py`, followed by the `local_atomic_motif_plotter.py`. Do not be surprised by ridiculously long runtimes (which I'll hopefully work on later).  

The `LAMisolator` and `LAMplotter` both include functions called `full_compute()`, which should work as-is.  

Hopefully, further documentation will soon be added to explain how to modify this code.