//g++ Neural.cpp -o neural_network -I//home/wiktoria/Desktop/Thesis/graphviz-8.1.0 -L//home/wiktoria/Desktop/Thesis/graphviz-8.1.0/lib -lgvc -lcgraph
#include <fstream>
#include <sstream>
#include <string>
#include <cmath> 
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <vector>
#include <cstdlib>
//#include <graphviz/gvc.h>

typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;

std::ofstream weightFile("weights.csv");

class NeuralNetwork {
public:
    NeuralNetwork(std::vector<uint> topology, Scalar learningRate = Scalar(0.005));

    void propagateForward(RowVector& input);
    void propagateBackward(RowVector& output);
    void calcErrors(RowVector& output);
    void updateWeights();
    void train(std::vector<RowVector*>& data, std::vector<RowVector*>& labels);
    void test(std::vector<RowVector*>& data, std::vector<RowVector*>& labels);

    std::vector<RowVector*> neuronLayers;
    std::vector<RowVector*> cacheLayers;
    std::vector<RowVector*> deltas;
    std::vector<Matrix*> weights;
    Scalar learningRate;
};

// Constructor of the neural network class
NeuralNetwork::NeuralNetwork(std::vector<uint> topology, Scalar learningRate)
{
    this->learningRate = learningRate;
    for (uint i = 0; i < topology.size(); i++) {
        // Initialize neuron layers
        if (i == topology.size() - 1)
            neuronLayers.push_back(new RowVector(topology[i]));
        else
            neuronLayers.push_back(new RowVector(topology[i] + 1));

        // Initialize cache and delta vectors
        cacheLayers.push_back(new RowVector(neuronLayers[i]->size()));
        deltas.push_back(new RowVector(neuronLayers[i]->size()));

        // Add bias neuron (set to 1.0) to all layers except the output layer
        if (i != topology.size() - 1) {
            neuronLayers.back()->coeffRef(topology[i]) = 1.0;
            cacheLayers.back()->coeffRef(topology[i]) = 1.0;
        }

        // Initialize weights matrix
        if (i > 0) {
            if (i != topology.size() - 1) {
                weights.push_back(new Matrix(topology[i - 1] + 1, topology[i] + 1));
                weights.back()->setRandom();
                weights.back()->col(topology[i]).setZero();
                weights.back()->coeffRef(topology[i - 1], topology[i]) = 1.0;
            } else {
                weights.push_back(new Matrix(topology[i - 1] + 1, topology[i]));
                weights.back()->setRandom();
            }
        }
    }
}

// Tanh activation function
Scalar activationFunction(Scalar x)
{
    return std::tanh(x);
}

// Derivative of the tanh activation function
Scalar activationFunctionDerivative(Scalar x)
{
    Scalar tanh_x = std::tanh(x);
    return 1 - tanh_x * tanh_x;
}


// Function for forward propagation of data
void NeuralNetwork::propagateForward(RowVector& input)
{
    // Set the input to the input layer
    neuronLayers.front()->block(0, 0, 1, neuronLayers.front()->size() - 1) = input;

    // Propagate the data forward and apply the activation function to your network
    for (uint i = 1; i < neuronLayers.size(); i++) {
        (*neuronLayers[i]) = (*neuronLayers[i - 1]) * (*weights[i - 1]);
        neuronLayers[i]->block(0, 0, 1, neuronLayers[i]->size()).unaryExpr(std::ptr_fun(activationFunction));

        // Apply the activation function for hidden layers
        if (i != neuronLayers.size() - 1) {
            (*neuronLayers[i]) = (*neuronLayers[i]).unaryExpr(std::ptr_fun(activationFunction));
        }
    }

    // Apply softmax to the outputs of the output layer
    RowVector& outputLayer = *neuronLayers.back();
    outputLayer = outputLayer.array().exp();
    outputLayer /= outputLayer.sum();
}



void NeuralNetwork::calcErrors(RowVector& output)
{
    // Calculate the errors made by neurons of the last layer
    (*deltas.back()) = output - (*neuronLayers.back());

    // Error calculation of hidden layers
    for (uint i = deltas.size() - 2; i > 0; i--) {
        (*deltas[i]) = (*deltas[i + 1]) * (weights[i]->transpose());
        for (uint j = 0; j < deltas[i]->size(); j++) {
            deltas[i]->coeffRef(j) *= activationFunctionDerivative(neuronLayers[i]->coeffRef(j));
        }
    }
}


void NeuralNetwork::updateWeights()
{
    for (uint i = 0; i < weights.size(); i++) {
        weightFile << "Layer " << i << ":"; // Add layer information
        for (uint c = 0; c < weights[i]->cols(); c++) {
            for (uint r = 0; r < weights[i]->rows(); r++) {
                // Consider the bias neuron (always set to 1.0) for all layers
                Scalar delta = learningRate * deltas[i + 1]->coeffRef(c) * neuronLayers[i]->coeffRef(r);
                weights[i]->coeffRef(r, c) += delta;
                weightFile << "," << weights[i]->coeffRef(r, c);
            }
        }
        weightFile << "\n";
    }
}


// Function for backward propagation of errors made by neurons
void NeuralNetwork::propagateBackward(RowVector& output)
{
    calcErrors(output);
    updateWeights();
}



void NeuralNetwork::train(std::vector<RowVector*>& data, std::vector<RowVector*>& labels)
{
    for (uint i = 0; i < data.size(); i++) {
        propagateForward(*data[i]);
        propagateBackward(*labels[i]); // Use labels for supervised learning

        // Calculate the cross-entropy loss between the predicted output and the true label
        RowVector& predictedOutput = *neuronLayers.back();
        RowVector& trueLabel = *labels[i];

        //std::cout << "Predicted output: " << predictedOutput << std::endl;
        //std::cout << "True label: " << trueLabel << std::endl;

        Scalar crossEntropyLoss =
            -trueLabel.dot(predictedOutput.unaryExpr([](Scalar x) { return std::log(x); })) -
            (1 - trueLabel.array()).matrix().dot(((1 - predictedOutput.array()).unaryExpr([](Scalar x) { return std::log(1 - x); })).matrix());

        weightFile << "i: " << i << " Cross-Entropy Loss: " << crossEntropyLoss << "\n";
    }
}


void NeuralNetwork::test(std::vector<RowVector*>& data, std::vector<RowVector*>& labels)
{
    Scalar totalCrossEntropyLoss = 0.0;
    int numSamples = data.size();

    for (int i = 0; i < numSamples; i++) {
        propagateForward(*data[i]);

        // Calculate the cross-entropy loss between the predicted output and the true label
        RowVector& predictedOutput = *neuronLayers.back();
        RowVector& trueLabel = *labels[i];

        std::cout << "True label: " << trueLabel << std::endl;

        // Convert predictedOutput to one-hot encoded representation
        RowVector predictedClass = predictedOutput.unaryExpr([](Scalar x) { return x > 0.5 ? 1.0f : 0.0f; });

        std::cout << "Predicted label: " << predictedClass << std::endl;

        Scalar crossEntropyLoss =
            -trueLabel.dot(predictedOutput.unaryExpr([](Scalar x) { return std::log(x); })) -
            (1 - trueLabel.array()).matrix().dot(((1 - predictedOutput.array()).unaryExpr([](Scalar x) { return std::log(1 - x); })).matrix());

        totalCrossEntropyLoss += crossEntropyLoss;
    }

    // Calculate the Mean Cross-Entropy Loss over the test data
    Scalar meanCrossEntropyLoss = totalCrossEntropyLoss / numSamples;

    std::cout << "Mean Cross-Entropy Loss on test data: " << meanCrossEntropyLoss << std::endl;
}

// Function to read the Iris dataset from a CSV file
void ReadIrisData(std::string filename, std::vector<RowVector*>& data, std::vector<RowVector*>& labels)
{
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }

    // Read data from the CSV file
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        RowVector* input = new RowVector(4); // 4 features in Iris dataset
        RowVector* label = new RowVector(3); // 3 classes (one-hot encoded)

        // Read features (sepal length, sepal width, petal length, petal width)
        for (int i = 0; i < 4; ++i) {
            float value;
            char delimiter;
            ss >> value;
            input->coeffRef(0, i) = value;
            if (i < 3) ss >> delimiter; // Read the comma delimiter
        }

        // Read label (species) and convert it to one-hot encoded vector
        std::string species;
        ss >> species;
        //std::cout << "Read species: '" << species << "', length: " << species.length() << std::endl;  // Add this line
        label->setZero();  // Set all classes to 0 by default
        if (species == ",0") {
            label->coeffRef(0) = 1.0; // Set the appropriate class to 1
        } else if (species == ",1") {
            label->coeffRef(1) = 1.0;
        } else if (species == ",2") {
            label->coeffRef(2) = 1.0;
        }


        // Add the data and labels to the vectors
        data.push_back(input);
        //std::cout << "Label: " << label << std::endl;
        labels.push_back(label);
    }

    file.close();
}
/*
void visualizeNetworkArchitecture(const NeuralNetwork& network) {
    Agraph_t* graph = agopen(const_cast<char*>("Network"), Agdirected, nullptr);

    // Create nodes for neurons
    std::vector<std::vector<Agnode_t*>> neuronNodes; // Store created nodes for each layer
    for (size_t i = 0; i < network.neuronLayers.size(); ++i) {
        std::vector<Agnode_t*> layerNodes; // Store created nodes for this layer
        for (size_t j = 0; j < network.neuronLayers[i]->size(); ++j) {
            std::string neuronName = "Neuron_" + std::to_string(i) + "_" + std::to_string(j);
            layerNodes.push_back(agnode(graph, const_cast<char*>(neuronName.c_str()), 1)); // Use cflag = 1 to create a new node
        }
        neuronNodes.push_back(layerNodes);
    }

    // Manually create edges between neurons
    for (size_t i = 0; i < neuronNodes.size() - 1; ++i) {
        for (Agnode_t* sourceNode : neuronNodes[i]) {
            for (Agnode_t* targetNode : neuronNodes[i + 1]) {
                Agedge_t* edge = agedge(graph, sourceNode, targetNode, nullptr, 1); // Use cflag = 1 to create a new edge

                // You can customize the edge properties using agsafeset or other functions here
            }
        }
    }

    // Set layout engine and render the graph
    GVC_t* gvc = gvContext();
    gvLayout(gvc, graph, "dot");
    gvRenderFilename(gvc, graph, "svg", "network.svg");
    gvFreeLayout(gvc, graph);
    agclose(graph);
    gvFreeContext(gvc);
}

*/

typedef std::vector<RowVector*> data;



int main()
{
    // Create a neural network with topology: 4 input neurons, 3 hidden neurons, 3 output neurons (one-hot encoded)
    NeuralNetwork n({4, 3, 3});

    // Read training data from CSV files
    data train_data, train_labels;
    ReadIrisData("iris_augmented.csv", train_data, train_labels);

    // Train the neural network with the training data
    n.train(train_data, train_labels);

    // Read test data from CSV files
    data test_data, test_labels;
    ReadIrisData("iris_aug_test.csv", test_data, test_labels);

    // Evaluate the neural network on the test data
    n.test(test_data, test_labels);

    //visualizeNetworkArchitecture(n);

    // Free memory for training data and labels
    for (RowVector* data : train_data) {
        delete data;
    }
    for (RowVector* label : train_labels) {
        delete label;
    }

    // Free memory for test data and labels
    for (RowVector* data : test_data) {
        delete data;
    }
    for (RowVector* label : test_labels) {
        delete label;
    }

    return 0;
}
