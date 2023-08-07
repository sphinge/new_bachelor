#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
using namespace std;

const double learning_rate = 0.01;
const double RELU_LEAK = 0.01;

// Read data from a CSV file and return input_data and output_data
pair<vector<vector<double>>, vector<double>> readCSV(const string& filename) {
    ifstream file(filename);
    vector<vector<double>> input_data;
    vector<double> output_data;
    string line;

    while (getline(file, line)) {
        vector<double> row;
        stringstream ss(line);
        string cell;

        while (getline(ss, cell, ',')) {
            if (ss.peek() != EOF) {
                row.push_back(stod(cell));
            }
            else {
                output_data.push_back(stod(cell));
            }
        }

        input_data.push_back(row);
    }

    return make_pair(input_data, output_data);
}

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double relu(double x) {
    return max(0.0, x);
}

double lrelu(double x) {
    return x > 0 ? x : x * RELU_LEAK;
}

double dsigmoid(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

double drelu(double x) {
    return x > 0 ? 1 : 0;
}

double dlrelu(double x) {
    return x > 0 ? 1 : RELU_LEAK;
}

// Helper function for data with multiple classes, turns a vector with tags into an expanded binary vector.
vector<vector<double>> extendVector(vector<double> raw, int row_size) {
    vector<vector<double>> extended(row_size, vector<double>(raw.size(), 0));

    for (int i = 0; i < row_size; ++i) {
        extended[i][i] = 1;
    }

    return extended;
}

class Layer {
public:
    // Constructor to create a layer with the specified number of neurons.
    Layer(int neuron_count);

    // Set the activation values (z_vals) for the layer from a raw array of doubles.
    void setValues(double* z_vals);

    // Set the activation values (z_vals) for the layer from a 2D vector.
    void setValues(vector<vector<double>>& z_vals);

    // Show the current activation values (z_vals) and active values (a_vals) of the layer.
    void showActiveValues();

    // Compute the activation values (z_vals) of the layer using the given weights and biases.
    void computeZVals(vector<vector<double>>& weight, vector<double>& bias, vector<vector<double>>& a_vals);

    // Get the number of neurons in the layer.
    int getCount() const { return neuron_count; }

    // Get a reference to the activation values (a_vals) of the layer.
    vector<vector<double>>& getA() { return a_vals; }

    // Get a reference to the activation values (z_vals) of the layer.
    vector<vector<double>>& getZ() { return z_vals; }

private:
    int neuron_count;               // Number of neurons in the layer.
    vector<vector<double>> a_vals;  // Activation values of the neurons.
    vector<vector<double>> z_vals;  // Active values of the neurons.
};


Layer::Layer(int neuron_count) : a_vals(neuron_count, vector<double>(1, 1)), z_vals(neuron_count, vector<double>(1, 1)) {
    this->neuron_count = neuron_count;
}

void Layer::setValues(double* z_vals) {
    for (int i = 0; i < neuron_count; ++i) {
        this->z_vals[i][0] = z_vals[i];
    }
    this->a_vals = this->z_vals;
}

void Layer::setValues(vector<vector<double>>& z_vals) {
    this->z_vals = z_vals;
    this->a_vals = this->z_vals;
}

void Layer::showActiveValues() {
    cout << "z: ";
    for (int i = 0; i < neuron_count; i++) {
        cout << z_vals[i][0] << " ";
    }
    cout << endl;

    cout << "a: ";
    for (int i = 0; i < neuron_count; i++) {
        cout << a_vals[i][0] << " ";
    }
    cout << endl;
}

void Layer::computeZVals(vector<vector<double>>& weight, vector<double>& bias, vector<vector<double>>& a_vals) {
    assert(weight.size() == z_vals.size());
    assert(weight[0].size() == a_vals.size());
    assert(bias.size() == weight.size());

    for (int i = 0; i < neuron_count; i++) {
        z_vals[i][0] = 0;
        for (int j = 0; j < a_vals.size(); j++) {
            z_vals[i][0] += weight[i][j] * a_vals[j][0];
        }
        z_vals[i][0] += bias[i];
    }
}

class Network {
public:
    Network(int layer_count, vector<int> neuron_counts, vector<int> neuron_types);
    void forwardPropagate(const vector<double>& input_vals);
    void backPropagate(const vector<double>& output_data);
    void train(const vector<vector<double>>& input_data, const vector<double>& output_data, int epochs, int batch_size, const vector<vector<double>>& validation_input, const vector<double>& validation_output);
    void computeGradients(const vector<vector<double>>& delta_output, const vector<vector<double>>& a_vals, const vector<vector<double>>& weight, vector<vector<double>>& grad_weights, vector<double>& grad_biases);
    void updateWeights(int layer_index, double learning_rate);

private:
    int layer_count;
    vector<int> neuron_types;
    vector<Layer*> layers;
    vector<vector<vector<double>>> weights;
    vector<vector<double>> biases;
    vector<vector<vector<double>>> grad_weights;
    vector<vector<double>> grad_biases;
    void miniBatchSGD(const vector<vector<double>>& input_data, const vector<double>& output_data, int batch_size, double learning_rate);
};

Network::Network(int layer_count, vector<int> neuron_counts, vector<int> neuron_types)
    : weights(layer_count), biases(layer_count), grad_weights(layer_count), grad_biases(layer_count, vector<double>()), neuron_types(neuron_types)
{
    this->layer_count = layer_count;
    layers = vector<Layer*>(layer_count);

    default_random_engine generator;
    normal_distribution<double> distribution(0.0, 0.1);

    for (int i = 0; i < layer_count; ++i) {
        switch (neuron_types[i]) {
            case 0:
                layers[i] = new Layer(neuron_counts[i]);
                break;
            default:
                throw string("Unidentified activation function!");
        }

        weights[i] = vector<vector<double>>(neuron_counts[i], vector<double>(i > 0 ? neuron_counts[i - 1] : 1, distribution(generator))); // initializes the weight matrix for the current layer. It creates a 2D vector with neuron_counts[i] rows and i > 0 ? neuron_counts[i - 1] : 1 columns, and fills the matrix with random values from the distribution.
        grad_weights[i] = vector<vector<double>>(neuron_counts[i], vector<double>(i > 0 ? neuron_counts[i - 1] : 1, 0.0)); // initializes the gradient weights matrix for the current layer. It creates a 2D vector with the same size as the weight matrix and fills it with zeros.

        biases[i] = vector<double>(neuron_counts[i], distribution(generator)); // initializes the biases vector for the current layer. It creates a 1D vector with neuron_counts[i] elements and fills it with random values from the distribution.
        grad_biases[i] = vector<double>(neuron_counts[i], 0.0); // initializes the gradient biases vector for the current layer. It creates a 1D vector with the same size as the biases vector and fills it with zeros.
    }
}

void Network::forwardPropagate(const vector<double>& input_vals) {
    if (input_vals.size() != layers[0]->getCount()) {
        throw runtime_error("Input data size does not match the number of input neurons.");
    }

    // provide the initial input data to the neural network's first layer during the forward propagation step
    for (int i = 0; i < input_vals.size(); ++i) {
        layers[0]->getA()[i][0] = input_vals[i];
    }

    //the loop iterates over all layers (excluding the first layer) and computes the activation values (z_vals)
    //of the neurons for each layer using the corresponding weights, biases, and active values from the previous layer
    for (int i = 1; i < layer_count; i++) {
        vector<vector<double>>& weight = weights[i];
        vector<double>& bias = biases[i];
        
        /*
        // Print weights
        cout << "Layer " << i << " weights:" << endl;
        for (const auto& row : weight) {
            for (double w : row) {
                cout << w << " ";
            }
            cout << endl;
        }
        
        // Print biases
        cout << "Layer " << i << " biases:" << endl;
        for (double b : bias) {
            cout << b << " ";
        }
        cout << endl;
        */

        layers[i]->computeZVals(weight, bias, layers[i - 1]->getA());
        
    }
}

void Network::backPropagate(const vector<double>& output_data) {
    int last_layer_idx = layer_count - 1;
    Layer* output_layer = layers[last_layer_idx];
    const vector<vector<double>>& predicted_output = output_layer->getA();

    if (output_data.size() != predicted_output.size()) {
        throw runtime_error("Output data size does not match the predicted output size.");
    }

    vector<vector<double>> delta_output(predicted_output.size(), vector<double>(1, 0));
    for (int i = 0; i < predicted_output.size(); ++i) {
        delta_output[i][0] = predicted_output[i][0] - output_data[i];
    }

    for (int i = last_layer_idx; i >= 0; --i) {
        const vector<vector<double>>& weight = weights[i];
        vector<double>& bias = biases[i];
        Layer* current_layer = layers[i];
        const vector<vector<double>>& a_vals = current_layer->getA();
        const vector<vector<double>>& z_vals = current_layer->getZ();
        vector<vector<double>> delta_next;

        if (i < last_layer_idx) {
            delta_next = vector<vector<double>>(a_vals.size(), vector<double>(1, 0));
            for (int j = 0; j < a_vals.size(); ++j) {
                double sum = 0;
                if (a_vals.size() > 1) {
                    for (int k = 0; k < delta_output.size(); ++k) {
                        sum += weights[i + 1][k][j] * delta_output[k][0];
                    }
                    if (neuron_types[i] == 0) {
                        delta_next[j][0] = dlrelu(z_vals[j][0]) * sum;
                    }
                    else {
                        throw runtime_error("Unidentified activation function!");
                    }
                }
                else {
                    if (neuron_types[i] == 0) {
                        delta_next[j][0] = dlrelu(z_vals[j][0]) * delta_output[0][0];
                    }
                    else {
                        throw runtime_error("Unidentified activation function!");
                    }
                }
            }
        }

        computeGradients(delta_output, a_vals, weight, grad_weights[i], grad_biases[i]);
        updateWeights(i, learning_rate);
        delta_output = delta_next;
    }
}

void Network::computeGradients(const vector<vector<double>>& delta_output, const vector<vector<double>>& a_vals, const vector<vector<double>>& weight, vector<vector<double>>& grad_weights, vector<double>& grad_biases) {
    for (size_t j = 0; j < a_vals.size(); ++j) {
        for (size_t k = 0; k < a_vals[j].size(); ++k) {
            if (j < delta_output.size() && k < delta_output[j].size() &&
                j < grad_weights.size() && k < grad_weights[j].size()) {
                double gradient = delta_output[j][k] * a_vals[j][k];
                grad_weights[j][k] = gradient;
            }
        }

        if (j < delta_output.size()) {
            grad_biases[j] = delta_output[j][0];
        }
    }
}

void Network::updateWeights(int layer_index, double learning_rate) {
    vector<vector<double>>& weight = weights[layer_index];
    vector<double>& bias = biases[layer_index];
    vector<vector<double>>& grad_weight = grad_weights[layer_index];
    vector<double>& grad_bias = grad_biases[layer_index];

    for (int j = 0; j < weight.size(); ++j) {
        for (int k = 0; k < weight[j].size(); ++k) {
            weight[j][k] -= learning_rate * grad_weight[j][k];
        }

        bias[j] -= learning_rate * grad_bias[j];
    }
}

void Network::train(const vector<vector<double>>& input_data, const vector<double>& output_data, int epochs, int batch_size, const vector<vector<double>>& validation_input, const vector<double>& validation_output) {
    if (input_data.size() != output_data.size()) {
        throw runtime_error("Input data and output data sizes do not match.");
    }

    if (!validation_input.empty() && validation_input.size() != validation_output.size()) {
        throw runtime_error("Validation input data and output data sizes do not match.");
    }

    vector<size_t> indices(input_data.size());

    for (size_t i = 0; i < input_data.size(); ++i) {
        indices[i] = i;
    }

    random_device rd;
    mt19937 rng(rd());

    for (int epoch = 0; epoch < epochs; ++epoch) {
        cout << "Epoch " << epoch + 1 << "/" << epochs << endl;
        shuffle(indices.begin(), indices.end(), rng);

        for (size_t start = 0; start < input_data.size(); start += batch_size) {
            size_t end = min(start + batch_size, input_data.size());

            //store the data samples (input and output pairs) corresponding to the current mini-batch
            vector<vector<double>> batch_input;
            vector<double> batch_output;

            for (size_t i = start; i < end; ++i) {
                batch_input.push_back(input_data[indices[i]]);
                batch_output.push_back(output_data[indices[i]]);
            }

            // Assuming you are iterating over the batch data
            for (size_t i = 0; i < batch_input.size(); ++i) {
                // Forward propagate
                forwardPropagate(batch_input[i]);

                // Back propagate
                backPropagate({ batch_output[i] });
            }

             // Print changes after each mini-batch
            cout << "Changes after mini-batch [" << start << " - " << end - 1 << "]:" << endl;
            for (size_t i = start; i < end; ++i) {
                cout << "Validation Input [" << i << "]: ";
                for (size_t j = 0; j < input_data[i].size(); ++j) {
                    cout << input_data[i][j] << " ";
                }
                cout << endl;

                cout << "Validation Output [" << i << "]: " << output_data[i] << endl;
            }
        
        }

        if (!validation_input.empty()) {
            double validation_loss = 0.0;
            //squared error between the predicted output of the network and the actual output (ground truth) 
            //during the validation process
            for (size_t i = 0; i < validation_input.size(); ++i) {
                forwardPropagate(validation_input[i]);
                /*
                //print validation input
                if (i >= 0 && i < validation_input.size()) {
                    std::cout << "Validation Input [" << i << "]: ";
                    for (const double& value : validation_input[i]) {
                        std::cout << value << " ";
                    }
                    std::cout << std::endl;
                } else {
                    std::cout << "Invalid index: " << i << std::endl;
                }
                */
                //print validation output
                if (i >= 0 && i < validation_output.size()) {
                    std::cout << "Validation Output [" << i << "]: " << validation_output[i] << std::endl;
                } else {
                    std::cout << "Invalid index: " << i << std::endl;
                }
                
                validation_loss += 0.5 * pow(validation_output[i] - layers[layer_count - 1]->getA()[0][0], 2);
            }
            cout << "Validation Loss: " << validation_loss / validation_input.size() << endl;
            //std::cout << "Layer Value: " << layers[layer_count - 1]->getA()[0][0] << std::endl;
            //std::cout << "Constant Value: " << 2 << std::endl;
        }
    }
}

int main() {
    // Read data from the CSV file
    pair<vector<vector<double>>, vector<double>> data = readCSV("iris2.csv");
    vector<vector<double>> input_data = data.first;
    vector<double> output_data = data.second;

    /*
    // Print input_data
    cout << "Input Data:" << endl;
    for (const auto& row : input_data) {
        for (const auto& value : row) {
            cout << value << " ";
        }
        cout << endl;
    }

    // Print output_data
    cout << "Output Data:" << endl;
    for (const auto& value : output_data) {
        cout << value << " ";
    }
    cout << endl;
    */

    // Split data into training and validation sets
    // Assuming the first 100 samples for training and the remaining for validation
    vector<vector<double>> training_input(input_data.begin(), input_data.begin() + 100);
    vector<vector<double>> validation_input(input_data.begin() + 100, input_data.end());
    vector<double> training_output(output_data.begin(), output_data.begin() + 100);
    vector<double> validation_output(output_data.begin() + 100, output_data.end());

    // Example usage of the network
    vector<int> neuron_counts = { 4, 10, 1 }; // 4 input features, 10 neurons in hidden layer, 1 neuron in output layer
    vector<int> neuron_types = { 0, 0, 0 }; // Assuming all layers use the LReLU activation function
    Network inet = Network(neuron_counts.size(), neuron_counts, neuron_types);

    // Train the network
    int epochs = 100;
    int batch_size = 32;
    inet.train(training_input, training_output, epochs, batch_size, validation_input, validation_output); // Training for 100 epochs with batch size of 32

    return 0;
}
