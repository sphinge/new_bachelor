# new_bachelor
# For others:
Neural Network : make | ./Neural ./Neural /Shark/examples/Supervised/data/mnist_subset.libsvm

Logistic Regression: make | ./Logistic

XGBoost: g++ -o boost_code boost_code.cpp -I/path/to/xgboost/include -L/path/to/xgboost/lib -lxgboost (g++ -o boost Boost.cpp -I/home/wiktoria/Desktop/Thesis/xgboost/include -L/home/wiktoria/Desktop/Thesis/xgboost/lib -lxgboost) | ./boost

# For me:
Install the required tools (Doxygen, Graphviz, Python, CMake, and Sphinx):

bash

# Install Doxygen and Graphviz
sudo apt-get update
sudo apt-get install doxygen graphviz

# Install Python and Sphinx
sudo apt-get install python3 python3-pip
pip3 install sphinx
pip3 install sphinxcontrib-doxylink
sudo apt-get install libomp-dev

    Download the most recent Boost binaries and set up Boost:

bash

# Download Boost and extract it to a desired location, e.g., /path/to/boost_1.59/
# Replace the URL below with the actual download link
wget -O boost_1_59.tar.gz https://example.com/boost_1_59.tar.gz
tar -xzf boost_1_59.tar.gz

# Add the Boost library directory to the system's environment variable
echo 'export BOOST_LIBRARYDIR=/path/to/boost_1.59/lib64-msvc-12.0' >> ~/.bashrc
source ~/.bashrc

    Clone the Shark repository and prepare the build:

bash

# Clone the Shark repository
git clone https://github.com/Shark-ML/Shark.git
cd Shark

# Install CMake
sudo apt-get install cmake

# Create a build directory
mkdir build
cd build

    Configure and generate the documentation:

bash

cmake -DBUILD_DOCUMENTATION=ON ..

# Build the documentation
make doc

Troubleshooting:

If Shark cannot find Boost automatically or if you installed Boost in a custom location:

bash

# Configure and generate using explicit Boost paths
cmake -DBUILD_DOCUMENTATION=ON -DBoost_NO_SYSTEM_PATHS=TRUE -DBOOST_INCLUDEDIR=/path/to/boost/include/ -DBOOST_LIBRARYDIR=/path/to/boost/lib/ ..
make doc

Note: Replace "/path/to/boost/include/" and "/path/to/boost/lib/" with the actual paths to the Boost include and library directories.

# XGBoost
git clone --recursive https://github.com/dmlc/xgboost

cd xgboost
mkdir build
cd build
cmake ..
make -j$(nproc)


