#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm> // for std::shuffle
#include <numeric>   // for std::iota
#include <random>    // for std::default_random_engine
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <iostream>
#include <cstring>

#define LEARNING_RATE 0.001
#define NEPOCH 5
const float RELU_LEAK = 0.001;

using namespace std;

float unitrand()
{
    return (2.0 * (float)rand() / RAND_MAX) - 1.0;
}

float sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}
float relu(float x)
{
    return x > 0 ? x : 0;
}
float drelu(float x)
{
    return x >= 0 ? 1 : 0;
}
vector<float> softmax(vector<float> &z)
{
    float max_val = *max_element(z.begin(), z.end());
    vector<float> exp_values(z.size());
    float sum_exp = 0.0;
    for (unsigned int i= 0; i < z.size(); i++)
    {
        exp_values[i] = exp(z[i] - max_val);
        sum_exp += exp_values[i];
    }
    for (unsigned int i= 0; i < z.size(); i++)
    {
        exp_values[i] /= sum_exp;
    }
    return exp_values;
}

float cross_entropy_loss(vector<float> &y, vector<float> &y_hat)
{
    float loss = 0.0;
    for (unsigned int i= 0; i < y.size(); i++)
    {
        loss -= y[i] * log(y_hat[i] + 1e-9);
    }
    return loss;
}

class Neuron
{
public:
    unsigned int iLayer, iNeuron, nInputs;
    vector<float> weights;
    float bias;
    vector<float> x;
    float y;
    bool isHidden;

    Neuron(unsigned int _iLayer, unsigned int _iNeuron, unsigned int _nInputs, bool _isHidden)
        : iLayer(_iLayer), iNeuron(_iNeuron), nInputs(_nInputs), isHidden(_isHidden)
    {
        weights = vector<float>(nInputs);
        for (auto &w : weights)
            w = unitrand();
        bias = unitrand();
    }

    float feed(vector<float> &_x)
    {
        x = _x;
        float z = bias;
        for (unsigned int i= 0; i < nInputs; ++i)
            z += x[i] * weights[i];
        y = isHidden ? relu(z) : (z);
        return y;
    }

    float dy_dz()
    {
        return isHidden ? drelu(y) : 1;
    }

    float dz_dx(int i)
    {
        return weights[i];
    }

    void update_weights(float dC_dy)
    {
        float dC_dz = dC_dy * dy_dz();
        for (unsigned int i= 0; i < nInputs; ++i)
        {
            float dz_dw = x[i];
            float dC_dw = dC_dz * dz_dw;
            weights[i] -= LEARNING_RATE * dC_dw;
        }

        float dz_db = 1;
        float dC_db = dC_dz * dz_db;
        bias -= LEARNING_RATE * dC_db;
    }

};

class Layer
{
public:
    unsigned int iLayer, nInputs, nNeurons;
    vector<Neuron> neurons;
    vector<float> dC_dy;

    Layer(int _iLayer, int _nInputs, int _nNeurons, bool isHidden = true)
        : iLayer(_iLayer), nInputs(_nInputs), nNeurons(_nNeurons)
    {
        for (unsigned int i= 0; i < nNeurons; ++i)
            neurons.push_back(Neuron(iLayer, i, nInputs, isHidden));
    }

    vector<float> forward(vector<float> &x)
    {
        vector<float> y;
        for (auto &n : neurons)
            y.push_back(n.feed(x));
        return y;
    }

    vector<float> backward(vector<float> &_dC_dy)
    {
        dC_dy = _dC_dy;
        vector<float> dC_dy__prevLayer(nInputs, 0);
        for (unsigned int i= 0; i < nInputs; ++i)
        {
            for (unsigned int j = 0; j < neurons.size(); ++j)
                dC_dy__prevLayer[i] += dC_dy[j] * neurons[j].dy_dz() * neurons[j].weights[i];
        }
        return dC_dy__prevLayer;
    }

    void update_weights()
    {
        for (size_t i = 0; i < neurons.size(); ++i)
        {
            neurons[i].update_weights(dC_dy[i]);
        }
    }
};

class Network
{
public:
    vector<Layer> layers;

    Network(vector<int> nNeuronsEachLayer)
    {
        layers = vector<Layer>();
        unsigned int i;
        for (i = 0; i < nNeuronsEachLayer.size() - 1; ++i)
        {
            layers.push_back(Layer(i, nNeuronsEachLayer[i], nNeuronsEachLayer[i + 1], i < nNeuronsEachLayer.size() - 1));
        }
    }

    vector<float> feed(vector<float> &_x)
    {
        vector<float> x = _x;
        for (unsigned int i= 0; i < layers.size(); i++)
        {
            x = layers[i].forward(x);
        }
        return softmax(x); // Applying softmax here
    }

    vector<vector<float>> getWeights() const{
        vector<vector<float>> all_weights;
        for (const auto &layer : layers) {
            for (auto &neuron : layer.neurons) {
                all_weights.push_back(neuron.weights);
            }
        }
        return all_weights;
    }


    void setWeights(const vector<vector<float>>& weights) {
        int counter = 0;
        for (auto &layer : layers) {
            for (auto &neuron : layer.neurons) {
                neuron.weights = weights[counter++];
            }
        }
    }

    vector<float> getBiases() const {
        vector<float> all_biases;
        for (const auto &layer : layers) {
            for (auto &neuron : layer.neurons) {
                all_biases.push_back(neuron.bias);
            }
        }
        return all_biases;
    }

    void setBiases(const vector<float>& biases) {
        int counter = 0;
        for (auto &layer : layers) {
            for (auto &neuron : layer.neurons) {
                neuron.bias = biases[counter++];
            }
        }
    }

    std::string serialize() const {
        std::ostringstream oss;

        auto weights = getWeights();
        for (const auto& neuronWeights : weights) {
            for (float weight : neuronWeights) {
                oss << weight << " ";
            }
        }

        auto biases = getBiases();
        for (float bias : biases) {
            oss << bias << " ";
        }

        std::string result = oss.str();
        if (!result.empty()) {
            result.pop_back(); // Remove the last space
        }

        return result;
    }

    void deserialize(const std::string& serialized) {
        std::istringstream iss(serialized);

        std::vector<float> allValues;
        float value;
        while (iss >> value) {
            allValues.push_back(value);
        }

        std::vector<std::vector<float>> weights = getWeights();
        size_t index = 0;

        for (auto& neuronWeights : weights) {
            for (size_t j = 0; j < neuronWeights.size(); ++j) {
                neuronWeights[j] = allValues[index++];
            }
        }

        setWeights(weights);

        std::vector<float> biases;
        while (index < allValues.size()) {
            biases.push_back(allValues[index++]);
        }

        setBiases(biases);
    }

};

float trainOneSample(Network &network, vector<float> &x, vector<float> &y)
{
    vector<float> y_hat = network.feed(x); // Now y_hat directly has softmax applied values

    vector<float> dC_dlogits = vector<float>(y.size());
    for (unsigned int i= 0; i < y.size(); ++i)
        dC_dlogits[i] = y_hat[i] - y[i];

    for ( int i= network.layers.size() - 1; i >= 0; --i)
    {
        dC_dlogits = network.layers[i].backward(dC_dlogits);
        network.layers[i].update_weights();
    }

    float cross_entropy = cross_entropy_loss(y, y_hat); // using your cross_entropy_loss function

    return cross_entropy;
}

float trainOneEpoch(Network &network, vector<vector<float>> &X, vector<vector<float>> &Y)
{
    float total_loss = 0.0;

    vector<int> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);
    auto rng = std::default_random_engine{};
    std::shuffle(indices.begin(), indices.end(), rng);

    for (unsigned int i: indices)
    {
        float error = trainOneSample(network, X[i], Y[i]);
        total_loss += error;
    }

    return total_loss / X.size();
}

int predictClass(const vector<float> &y_hat)
{
    return max_element(y_hat.begin(), y_hat.end()) - y_hat.begin();
}

float computeAccuracy(Network &network, vector<vector<float>> &X, vector<vector<float>> &Y)
{
    int correctPredictions = 0;

    for (size_t i = 0; i < X.size(); ++i)
    {
        vector<float> y_hat_test = network.feed(X[i]);
        int predictedLabel = predictClass(y_hat_test);
        int realLabel = max_element(Y[i].begin(), Y[i].end()) - Y[i].begin();

        if (predictedLabel == realLabel)
            correctPredictions++;
    }

    return (float)correctPredictions / X.size();
}



namespace {
    const int BUFFER_SIZE = 30720;

    void log(const std::string& message) {
        std::cout << message << std::endl;
    }

    void exitWithError(const std::string& errorMessage) {
        perror(errorMessage.c_str()); // perror will print the system error message
        exit(1);
    }

};

std::string receiveWithDynamicBuffering(int &sock) {
    char buffer[BUFFER_SIZE] = {0};
    std::string accumulator;


    while (true) {
        int bytesReceived = recv(sock, buffer, BUFFER_SIZE, 0);
        if (bytesReceived < 0) {
            exitWithError("Failed to receive bytes from server socket connection");
        }
        
        accumulator += std::string(buffer, bytesReceived);


        // Example: Assuming a "\n" as the end-of-message delimiter.
        // You can use any unique delimiter consistent between the server and the client.
        if (accumulator.find("\n") != std::string::npos) {
            break;
        }
    }
    return accumulator;
}

const int EXCHANGES = 10; // N is the number of times the exchange happens

int main() {

    vector<vector<float>> X; // Features
    vector<vector<float>> Y; // Labels

    ifstream file("iris_test.csv"); //TODO: CHANGE IRIS TEST HAS 10 SAMPLES 
    string line, value;

    while (getline(file, line))
    {
        stringstream ss(line);
        vector<float> features;

        // Read first 4 values as features
        for (unsigned int i= 0; i < 4; ++i)
        {
            getline(ss, value, ',');
            features.push_back(stod(value));
        }
        X.push_back(features);

        // Read last value as label and one-hot encode
        getline(ss, value);
        int label = stoi(value);
        vector<float> oneHot(3, 0.0);
        oneHot[label] = 1.0;
        Y.push_back(oneHot);
    }

    file.close();

    vector<vector<float>> X_test; // Features
    vector<vector<float>> Y_test; // Labels

    ifstream test_file("iris_test.csv");
    string line_test, value_test;

    while (getline(test_file, line_test))
    {
        stringstream ss_test(line_test);
        vector<float> features;

        // Read first 4 values as features
        for (unsigned int i= 0; i < 4; ++i)
        {
            getline(ss_test, value_test, ',');
            features.push_back(stod(value_test));
        }
        X_test.push_back(features);

        // Read last value as label and one-hot encode
        getline(ss_test, value_test);
        int label = stoi(value_test);
        vector<float> oneHot(3, 0.0);
        oneHot[label] = 1.0;
        Y_test.push_back(oneHot);
    }

    test_file.close();

    int clientSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (clientSocket == -1) {
        exitWithError("Cannot create socket");
    }

    struct sockaddr_in serverAddress;
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_port = htons(8080);  // Assuming the server is running on port 8080
    serverAddress.sin_addr.s_addr = inet_addr("127.0.0.1");

    if (connect(clientSocket, (struct sockaddr*) &serverAddress, sizeof(serverAddress)) == -1) {
        exitWithError("Failed to connect to server");
    } else {
        log("Connection established.");
    }

    for (int exchange = 0; exchange < EXCHANGES; ++exchange) {
        log("Waiting for model...\n");
        sleep(3);
        std::string received_str = receiveWithDynamicBuffering(clientSocket);
        log("Received network from server: " + received_str + "\n");

        Network networkClient({4,8,3});  // Initialize with an empty structure. Will be updated by deserialize.
        networkClient.deserialize(received_str);

        float accuracy = 0.0;
        for (unsigned int i= 0; i < NEPOCH; ++i) {
            float loss = trainOneEpoch(networkClient, X, Y);
            float accuracy_i = computeAccuracy(networkClient, X_test, Y_test);
            accuracy += accuracy_i;

            printf("Epoch %d, loss: %.4f, accuracy per iter: %.2f%%\n", i + 1, loss, accuracy_i * 100);
        }

        std::string updated_network_str = networkClient.serialize() + "\n";

        log("Sending network to server: " + updated_network_str + "\n");

        if (send(clientSocket, updated_network_str.c_str(), updated_network_str.length(), 0) < 0) {
            exitWithError("Failed to send network to server");
        }
    }

    close(clientSocket);
    return 0;
}

