# new_bachelor

# Updated For Others
I have implemented Neural Network, Logistic Regression, SVM and Knn. I have some issues with them, so let me go one by one.

1. Neural Network (NeuralScratch)
   
   Loction: Thesis/NeuralScratch
   
   It has a very high MSE for test data, while the train MSE is not perfect but low.
   
   Compile: g++ main.cpp Neural.cpp -o neural_network -std=c++11
   
   Run: ./neural_network
   
   Output(only the last iterations):
   
i: 110 MSE: 0.245548
i: 111 MSE: 0.0720359
i: 112 MSE: 0.219472
i: 113 MSE: 0.213353
i: 114 MSE: 0.067509
i: 115 MSE: 0.0407686
i: 116 MSE: 0.0980458
i: 117 MSE: 0.10478
Mean Squared Error (MSE) on test data: 0.731126

3. Logistic Regression (LogisticScratch)
   
   Location: Thesis/LogisticScratch
   
   This code seems to work well.
   
   Compile: g++ -o logisticV2 Logistic_V2.cpp
   
   Run: ./logisticV2
   
   Output:
Epoch 997 - Model 2 Error: 0.256904, Accuracy: 0.79661
Epoch 998 - Model 0 Error: 0.0436494, Accuracy: 0.991525
Epoch 998 - Model 1 Error: 0.4049, Accuracy: 0.661017
Epoch 998 - Model 2 Error: 0.256889, Accuracy: 0.79661
Epoch 999 - Model 0 Error: 0.0436293, Accuracy: 0.991525
Epoch 999 - Model 1 Error: 0.404898, Accuracy: 0.661017
Epoch 999 - Model 2 Error: 0.256873, Accuracy: 0.79661
Accuracy on Test Data: 0.8125

5. SVM
   
   Location: Thesis/SVM
   
   This code also seems ok
   
   Compile: g++ -o svm svm.cpp
   
   Run: ./svm
   
   Output:
Iteration 14800, training accuracy: 0.788136, average error: 0.296371
Iteration 14900, training accuracy: 0.788136, average error: 0.296082
Iteration 15000, training accuracy: 0.79661, average error: 0.295795
y = -0.0709235 * x1 + 0.933164 * x2 + -0.934084 * x3 + 0.177107 * x4 + -0.0929017
y = -0.0528905 * x1 + -0.165752 * x2 + -0.0491766 * x3 + 0.167461 * x4 + -0.179483
y = -0.976752 * x1 + -0.355887 * x2 + 1.13776 * x3 + 1.32275 * x4 + -0.552269
Accuracy: 0.78125

7. KNN
   
   Location: Thesis/knn
   
   I have an issue where the maxClass (predicted class) is always 2. Which makes no sense, the neighbors are not updated correctly.
   
   Compile: g++ -o knn knn.cpp
   
   Run: ./knn
   
   Output:
max class: 2.0 test: 1.0
max class: 2.0 test: 1.0
max class: 2.0 test: 1.0
max class: 2.0 test: 1.0
max class: 2.0 test: 2.0
max class: 2.0 test: 2.0
max class: 2.0 test: 2.0
max class: 2.0 test: 2.0
max class: 2.0 test: 2.0
max class: 2.0 test: 2.0
max class: 2.0 test: 2.0
max class: 2.0 test: 2.0
max class: 2.0 test: 2.0
max class: 2.0 test: 2.0
10 of 32 (31.25)




# For others:

Code is in master branch.

Neural Network :
 ```
 make | ./Neural ./Neural /Shark/examples/Supervised/data/mnist_subset.libsvm
```
Logistic Regression: 
```
make | ./Logistic
```
XGBoost:
```
g++ -o boost_code boost_code.cpp -I/path/to/xgboost/include -L/path/to/xgboost/lib -lxgboost (g++ -o boost Boost.cpp -I/home/wiktoria/Desktop/Thesis/xgboost/include -L/home/wiktoria/Desktop/Thesis/xgboost/lib -lxgboost) | ./boost
```
# For me:
Install the required tools (Doxygen, Graphviz, Python, CMake, and Sphinx):


# Install Doxygen and Graphviz
```bash
sudo apt-get update
sudo apt-get install doxygen graphviz
```
# Install Python and Sphinx
```bash
sudo apt-get install python3 python3-pip
pip3 install sphinx
pip3 install sphinxcontrib-doxylink
sudo apt-get install libomp-dev
```
Download the most recent Boost binaries and set up Boost:


# Download Boost and extract it to a desired location, e.g., /path/to/boost_1.59/
# Replace the URL below with the actual download link   
```bash
wget -O boost_1_59.tar.gz https://example.com/boost_1_59.tar.gz
tar -xzf boost_1_59.tar.gz
```
# Add the Boost library directory to the system's environment variable
```echo 'export BOOST_LIBRARYDIR=/path/to/boost_1.59/lib64-msvc-12.0' >> ~/.bashrc
source ~/.bashrc
```
Clone the Shark repository and prepare the build:


# Clone the Shark repository
```
git clone https://github.com/Shark-ML/Shark.git
cd Shark
```
# Install CMake
```
sudo apt-get install cmake
```
# Create a build directory
```bash
mkdir build
cd build
```
Configure and generate the documentation:

```bash
cmake -DBUILD_DOCUMENTATION=ON ..
```
# Build the documentation
```
make doc
```
Troubleshooting:

If Shark cannot find Boost automatically or if you installed Boost in a custom location:

bash

# Configure and generate using explicit Boost paths
cmake -DBUILD_DOCUMENTATION=ON -DBoost_NO_SYSTEM_PATHS=TRUE -DBOOST_INCLUDEDIR=/path/to/boost/include/ -DBOOST_LIBRARYDIR=/path/to/boost/lib/ ..
make doc

Note: Replace "/path/to/boost/include/" and "/path/to/boost/lib/" with the actual paths to the Boost include and library directories.

# XGBoost
```
git clone --recursive https://github.com/dmlc/xgboost
```
```bash
cd xgboost
mkdir build
cd build
cmake ..
make -j$(nproc)
```

