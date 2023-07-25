# MLPACK Installation and example
---
Following [page](https://mccormickml.com/2017/02/01/getting-started-with-mlpack/).

For Linux install
```bash
sudo apt-get install libmlpack-dev
```
may also need Boost
```bash
sudo apt-get install libboost-all-dev
```

Compile the code
```bash
g++ knn_example.cpp -o knn_example -std=c++11 -larmadillo -lmlpack -lboost_serialization
```
Execute
```bash
./knn_example
```

