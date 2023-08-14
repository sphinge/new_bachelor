#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>

#define LEARNING_RATE 0.05

using namespace std;

//ofstream weightsFile;

// gives random number between -1 and +1
double unitrand(){
    return (2.0*(double)rand() / RAND_MAX) - 1.0;
}

double sigmoid(double x){
    return 1 / (1 + exp(-x));
}

class Neuron {
public:
    int nInputs;
    vector<double> weights;
    double bias;
    double y;  // store this when we forward-pass as we use it when calculating the delta in the backward-pass
    // double* x;  // store this as we use it when updating the weights
    // ^ Note that the same x vector will be used by all neurons in the layer, so store a reference to avoid duplicating
    // double dC_dy;  // this will get set by the next layer when we backpropagate
    // double dy_dz, dz_dx;
vector<double> x;
    Neuron(int _nInputs) : nInputs(_nInputs) {
        cout << "    Constructing neuron with " << nInputs << " inputs!" << endl;
        // randomly initialize weights between 0 and 1
        weights = vector<double>(nInputs);
        for (auto& w : weights)
            w = unitrand();
        bias = unitrand();
    }

    double feed(vector<double>& _x) {
        x = _x;  // &_x[0];  // assign the reference
        double z = bias;
        for (int i = 0; i < nInputs; ++i)
            z += x[i] * weights[i];
        y = sigmoid(z);
        return y;
    }

    // dOutput / dPreActivation
    double dy_dz() {
        return y * (1 - y);
    }
    // dPreActivation / dInput[i]
    double dz_dx(int i) {
        return weights[i];
    }

    void update_weights(double dC_dy) {
        // example: suppose dC__dw = 0.1
        // what does this mean?
        // it means if we tweak the weight by 0.1, the cost will increase by 0.01
        // But we want to REDUCE the cost
        // so we should be tweaking it in the OPPOSITE direction!
        double dC_dz = dC_dy * dy_dz();
        for (int i=0; i < nInputs; ++i) {
            // z = b + x0 w0 + x1 w1 + ...
            //   => dz_dw_i = x_i
            double dz_dw = x[i];
            double dC_dw = dC_dz * dz_dw;
            weights[i] -= LEARNING_RATE * dC_dw;
        }

        double dz_db = 1;
        double dC_db = dC_dz * dz_db;
        bias -= LEARNING_RATE * dC_db;

        // Write weights to file
        // for (const auto& w : weights) {
        //     //weightsFile << w << " ";
        // }
        // weightsFile << bias << endl;  // also writing the bias

        // weightsFile.close();
    }
};

class Layer {
    vector<Neuron> neurons;
    vector<double> dC_dy;
    int nInputs, nNeurons;
public:
    // CHANGED
    Layer(int _nInputs, int _nNeurons) : nInputs(_nInputs), nNeurons(_nNeurons) {
        cout << "  Creating layer with " << nInputs << " inputs, " << nNeurons << " neurons!" << endl;
        for (int i = 0; i < nNeurons; ++i)
            neurons.push_back(Neuron(nInputs));
    }


    vector<double> forward(vector<double>& x) {
        vector<double> y;
        for (auto& n : neurons)
            y.push_back(n.feed(x));
        return y;
    }

    /*
    Supposing we have dC/dy for this layer (Q), we want to compute dC/dy for the previous layer (P),
    thus backpropagating the delta/error/cost
    */
    vector<double> backward(vector<double>& _dC_dy) {
        dC_dy = _dC_dy;  // copy

        // calculate dC/d{each input to this layer}
        // Note:
        //   Remember: The inputs to this layer are the outputs of the previous layer
        //   So we'll refer to them as such: dC/dy_layerP
        vector<double> dC_dy__prevLayer(nInputs, 0);
        for (int i = 0; i < nInputs; ++i) {
            // add the contribution from each neuron j in the layer
            //     dC/dyPrevLayer_i = sum_j dC/dy_j * dy_j/dz_j * dz_j/dx_j
            //                      = sum_j dC/dy_j * y(1-y)    * w[i]
            for(int j=0; j < neurons.size(); ++j)
                dC_dy__prevLayer[i] += dC_dy[j] * neurons[j].dy_dz() * neurons[j].dz_dx(i);  // chain rule
        }
        return dC_dy__prevLayer;
    }

    // update the weights of the neurons in this layer based on the deltas
    void update_weights() {
        for (size_t i = 0; i < neurons.size(); ++i) {
            // dC/dw = dC/dy * dy/dz * dz/dw
            //       = dC/dy * y(1-y) * x
            // w -> w - LEARNING_RATE * dC/dw
            neurons[i].update_weights(dC_dy[i]);
        }
    }

};

class Network {
public:
    vector<Layer> layers;
    Network(vector<int> nNeuronsEachLayer) {
        cout << "🔸 Creating network with " << nNeuronsEachLayer.size() - 1 << " layers!" << endl;
        // first element is nInputs
        layers = vector<Layer>();
        for(int i=0; i < nNeuronsEachLayer.size() - 1; ++i)
            layers.push_back(Layer(nNeuronsEachLayer[i], nNeuronsEachLayer[i+1]));
    }
    vector<double> feed(vector<double>& _x) {
        vector<double> x = _x;  // copy
        for(auto& layer : layers) 
            x = layer.forward(x);
        return x;
    }
};



double trainOneSample(Network& network, vector<double>& x, vector<double>& y) {
    // forward pass
    vector<double> y_hat = network.feed(x);

    // backward pass
    vector<double> dC_dy = vector<double>(y.size());
    for (int i = 0; i < y.size(); ++i)
        dC_dy[i] = y_hat[i] - y[i];

    //cout << "Test train" << endl;

    for (int i = network.layers.size() - 1; i >= 0; --i) {
        dC_dy = network.layers[i].backward(dC_dy);
        network.layers[i].update_weights();
        //weightsFile << "Layer: " << i << endl;
    }

    //cout << "Test train" << endl;

    double mse = 0.0;
    for (int i = 0; i < y.size(); ++i) {
        mse += pow(y_hat[i] - y[i], 2);
    }

    //cout << "Test train" << endl;

    return mse / y.size();
}

#define SHUFFLE_TRAINING_DATA 1

#if SHUFFLE_TRAINING_DATA
#include <algorithm>  // for std::shuffle
#include <numeric>    // for std::iota
#include <random>     // for std::default_random_engine

double trainOneEpoch(Network& network, vector<vector<double>>& X, vector<vector<double>>& Y) {
    double totalMSE = 0.0;

    // Create an index array [0, 1, ..., N-1]
    vector<int> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);  // fills the vector with increasing values

    // Shuffle the index array
    auto rng = std::default_random_engine {};
    std::shuffle(indices.begin(), indices.end(), rng);

    // Use shuffled indices to access X and Y
    for (int i : indices) {
        double error = trainOneSample(network, X[i], Y[i]);
        totalMSE += error;
    }

    return totalMSE / X.size();
}
#else
double trainOneEpoch(Network& network, vector<vector<double>>& X, vector<vector<double>>& Y) {
    double totalMSE = 0.0;

    // TODO: shuffle training data!
    for(int i=0; i < X.size(); ++i) {
        double error = trainOneSample(network, X[i], Y[i]);
        // cout << "Error: " << error << endl;
        totalMSE += error;
    }
    //cout << "Test 3" << endl;
    return totalMSE / X.size();
}
#endif




int main() {

    //weightsFile.open("plot_weights.csv", ios::app);  // opens the file in append mode

    // seed the random number generator with system time
    srand(time(NULL));


#if 0
    // for debugging, we might want to fix the seed, so we get the same random numbers each time
    // i.e. deterministic behaviour
    srand(41);
#endif

    vector<int> nNeuronsEachLayer = {2, 3, 1};  // first value is nInputs, second value is nNeurons in 0th layer, etc.
    //cout << "Test" << endl;
    Network network(nNeuronsEachLayer);

    // train on XOR
    vector<vector<double>> X = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    vector<vector<double>> Y = {
        {0},
        {1},
        {1},
        {0}
    };

#define NEPOCH 10000

    // Open the CSV file for writing
    //ofstream outputFile("plot_errors.csv");
    
    // Write header to the CSV file
    //outputFile << "Epoch, MSE" << endl;

    cout << "🔸 Training with " << NEPOCH << " epochs!" << endl;
    for (int i = 0; i < NEPOCH; ++i) {
        //cout << "Test" << endl;
        double epochMSE = trainOneEpoch(network, X, Y);

        //Write iteration to weight file
        //weightsFile << "Iteration: " << i << endl;

        // Write the result to the CSV file
        //outputFile << (i+1) << ", " << epochMSE << endl;

        if(i % 100 == 0)
            cout << "Epoch " << (i+1) << " - MSE: " << epochMSE << endl;

    }

    return 0;
}
