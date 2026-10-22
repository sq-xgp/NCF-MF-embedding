# Dataset setup

This repository does not redistribute MovieLens 100K. Its bundled license
requires separate permission to redistribute the original data.

Download MovieLens 100K directly from the [GroupLens dataset page](https://grouplens.org/datasets/movielens/100k/)
and extract it so that these files exist locally:

```text
data/ml-100k/u.data
data/ml-100k/u.user
data/ml-100k/u.item
```

The experiments reuse `u.data` and the metadata files directly. They do not
download or re-split the data at runtime. The shared `temporal_split()` function
performs a deterministic per-user chronological 8:1:1 split.

Please cite: F. Maxwell Harper and Joseph A. Konstan. 2015. “The MovieLens
Datasets: History and Context.” *ACM Transactions on Interactive Intelligent
Systems* 5(4), Article 19. DOI: 10.1145/2827872.
