/*
 * ======== knn_example.cpp =========
 * This very simple example takes a dataset of vectors and finds the nearest 
 * neighbor for each data point. It uses the dataset both as the reference and
 * the query vectors.
 *
 * mlpack documentation is here:
 * http://www.mlpack.org/docs/mlpack-2.0.2/doxygen.php
 */

#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

using namespace mlpack;
using namespace mlpack::neighbor; // NeighborSearch and NearestNeighborSort
using namespace mlpack::metric; // ManhattanDistance

int main()
{
    // Armadillo is a C++ linear algebra library; mlpack uses its matrix data type.
    arma::mat data;
    
    /*
     * Load the data from a file. mlpack does not provide an example dataset, 
     * so I've included a tiny one.
     *
     * 'data' is a helper class in mlpack that facilitates saving and loading 
     * matrices and models.
     *
     * Pass the filename, matrix to hold the data, and set fatal = true to have
     * it throw an exception if there is an issue.
     *
     * 'Load' excepts comma-separated and tab-separated text files, and will 
     * infer the format.
     */
    data::Load("data.csv", data, true);
    
    /* 
     * Create a NeighborSearch model. The parameters of the model are specified
     * with templates:
     *  - Sorting method: "NearestNeighborSort" - This class sorts by increasing
     *    distance.
     *  - Distance metric: "ManhattanDistance" - The L1 distance, sum of absolute
     *    distances.
     *
     * Pass the reference dataset (the vectors to be searched through) to the
     * constructor.
     */
    NeighborSearch<NearestNeighborSort, ManhattanDistance> nn(data);
    
    /*
     * Create the matrices to hold the results of the search. 
     *   neighbors [k  x  n] - Indeces of the nearest neighbor(s). 
     *                         One column per data query vector and one row per
     *                        'k' neighbors.
     *   distances [k  x  n] - Calculated distance values.
     *                         One column per data query vector and one row per
     *                        'k' neighbors.
     */
    arma::Mat<size_t> neighbors;
    arma::mat distances; 
    
    /*
     * Find the nearest neighbors. 
     *
     * If no query vectors are provided (as is the case here), then the 
     * reference vectors are searched against themselves.
     *
     * Specify the number of neighbors to find, k = 1, and provide matrices
     * to hold the result indeces and distances.
     */ 
    nn.Search(1, neighbors, distances);
    
    // Print out each neighbor and its distance.
    for (size_t i = 0; i < neighbors.n_elem; ++i)
    {
        std::cout << "Nearest neighbor of point " << i << " is point "
        << neighbors[i] << " and the distance is " << distances[i] << ".\n";
    }
}