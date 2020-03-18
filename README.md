### Image Clustering

This is a program uses a clustering algorithm to group pixels into a provided number of clusters. A raw image is supplied as a command-line argument, which writes to an output directory which is specified in `main.rs`. The current clustering implementation is primarily based on a color-only k-means algorithm.

```
$ cargo run --release pictures/kelp.jpg
```

A sample of the pre- and post-processed images can be found below, for an overhead shot of a kelp forest.

<p align="right"><img src="/pictures/kelp.jpg" width="400" height="300" align="left"><img src="/pictures/processed/kelp.jpg" width="400" height="300" /></p>

If using this program to process data for publication, please cite the UCSB COAST Lab. Thank you!
