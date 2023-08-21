#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm> // for std::shuffle
#include <numeric>   // for std::iota
#include <random>    // for std::default_random_engine
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>

#define LEARNING_RATE 0.001
#define NEPOCH 100
const double RELU_LEAK = 0.001;

using namespace std;

double unitrand()
{
    return (2.0 * (double)rand() / RAND_MAX) - 1.0;
}

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}
double lrelu(double x)
{
    return max(0.0, x);
}
double drelu(double x)
{
    return x >= 0 ? 1 : 0;
}
vector<double> softmax(vector<double> &z)
{
    double max_val = *max_element(z.begin(), z.end());
    vector<double> exp_values(z.size());
    double sum_exp = 0.0;
    for (int i = 0; i < z.size(); i++)
    {
        exp_values[i] = exp(z[i] - max_val);
        sum_exp += exp_values[i];
    }
    for (int i = 0; i < z.size(); i++)
    {
        exp_values[i] /= sum_exp;
    }
    return exp_values;
}

double cross_entropy_loss(vector<double> &y, vector<double> &y_hat)
{
    double loss = 0.0;
    for (int i = 0; i < y.size(); i++)
    {
        loss -= y[i] * log(y_hat[i] + 1e-9);
    }
    return loss;
}

class Neuron
{
public:
    int iLayer, iNeuron, nInputs;
    vector<double> weights;
    double bias;
    vector<double> x;
    double y;
    bool isHidden;

    Neuron(int _iLayer, int _iNeuron, int _nInputs, bool _isHidden)
        : iLayer(_iLayer), iNeuron(_iNeuron), nInputs(_nInputs), isHidden(_isHidden)
    {
        weights = vector<double>(nInputs);
        for (auto &w : weights)
            w = unitrand();
        bias = unitrand();
    }

    double feed(vector<double> &_x)
    {
        x = _x;
        double z = bias;
        for (int i = 0; i < nInputs; ++i)
            z += x[i] * weights[i];
        y = isHidden ? lrelu(z) : (z);
        return y;
    }

    double dy_dz()
    {
        return isHidden ? drelu(y) : 1;
    }

    double dz_dx(int i)
    {
        return weights[i];
    }

    void update_weights(double dC_dy)
    {
        double dC_dz = dC_dy * dy_dz();
        for (int i = 0; i < nInputs; ++i)
        {
            double dz_dw = x[i];
            double dC_dw = dC_dz * dz_dw;
            weights[i] -= LEARNING_RATE * dC_dw;
        }

        double dz_db = 1;
        double dC_db = dC_dz * dz_db;
        bias -= LEARNING_RATE * dC_db;
    }
};

class Layer
{
public:
    int iLayer, nInputs, nNeurons;
    vector<Neuron> neurons;
    vector<double> dC_dy;

    Layer(int _iLayer, int _nInputs, int _nNeurons, bool isHidden = true)
        : iLayer(_iLayer), nInputs(_nInputs), nNeurons(_nNeurons)
    {
        for (int i = 0; i < nNeurons; ++i)
            neurons.push_back(Neuron(iLayer, i, nInputs, isHidden));
    }

    vector<double> forward(vector<double> &x)
    {
        vector<double> y;
        for (auto &n : neurons)
            y.push_back(n.feed(x));
        return y;
    }

    vector<double> backward(vector<double> &_dC_dy)
    {
        dC_dy = _dC_dy;
        vector<double> dC_dy__prevLayer(nInputs, 0);
        for (int i = 0; i < nInputs; ++i)
        {
            for (int j = 0; j < neurons.size(); ++j)
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
        int i;
        for (i = 0; i < nNeuronsEachLayer.size() - 1; ++i)
        {
            layers.push_back(Layer(i, nNeuronsEachLayer[i], nNeuronsEachLayer[i + 1], i < nNeuronsEachLayer.size() - 1));
        }
    }

    vector<double> feed(vector<double> &_x)
    {
        vector<double> x = _x;
        for (int i = 0; i < layers.size(); i++)
        {
            x = layers[i].forward(x);
        }
        return softmax(x); // Applying softmax here
    }
};

double trainOneSample(Network &network, vector<double> &x, vector<double> &y)
{
    vector<double> y_hat = network.feed(x); // Now y_hat directly has softmax applied values

    vector<double> dC_dlogits = vector<double>(y.size());
    for (int i = 0; i < y.size(); ++i)
        dC_dlogits[i] = y_hat[i] - y[i];

    for (int i = network.layers.size() - 1; i >= 0; --i)
    {
        dC_dlogits = network.layers[i].backward(dC_dlogits);
        network.layers[i].update_weights();
    }

    double cross_entropy = cross_entropy_loss(y, y_hat); // using your cross_entropy_loss function

    return cross_entropy;
}

double trainOneEpoch(Network &network, vector<vector<double>> &X, vector<vector<double>> &Y)
{
    double total_loss = 0.0;

    vector<int> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);
    auto rng = std::default_random_engine{};
    std::shuffle(indices.begin(), indices.end(), rng);

    for (int i : indices)
    {
        double error = trainOneSample(network, X[i], Y[i]);
        total_loss += error;
    }

    return total_loss / X.size();
}

int predictClass(const vector<double> &y_hat)
{
    return max_element(y_hat.begin(), y_hat.end()) - y_hat.begin();
}

double computeAccuracy(Network &network, vector<vector<double>> &X, vector<vector<double>> &Y)
{
    int correctPredictions = 0;

    for (size_t i = 0; i < X.size(); ++i)
    {
        vector<double> y_hat_test = network.feed(X[i]);
        int predictedLabel = predictClass(y_hat_test);
        int realLabel = max_element(Y[i].begin(), Y[i].end()) - Y[i].begin();

        if (predictedLabel == realLabel)
            correctPredictions++;
    }

    return (double)correctPredictions / X.size();
}

int main()
{
    srand(time(NULL));

    vector<int> nNeuronsEachLayer = {4, 8, 3}; // Assuming 4 input features and 3 output classes
    Network network(nNeuronsEachLayer);

    vector<vector<double>> X; // Features
    vector<vector<double>> Y; // Labels

    ifstream file("iris_augmented.csv");
    string line, value;

    while (getline(file, line))
    {
        stringstream ss(line);
        vector<double> features;

        // Read first 4 values as features
        for (int i = 0; i < 4; ++i)
        {
            getline(ss, value, ',');
            features.push_back(stod(value));
        }
        X.push_back(features);

        // Read last value as label and one-hot encode
        getline(ss, value);
        int label = stoi(value);
        vector<double> oneHot(3, 0.0);
        oneHot[label] = 1.0;
        Y.push_back(oneHot);
    }

    file.close();

    vector<vector<double>> X_test; // Features
    vector<vector<double>> Y_test; // Labels

    ifstream test_file("iris_aug_test.csv");
    string line_test, value_test;

    while (getline(test_file, line_test))
    {
        stringstream ss_test(line_test);
        vector<double> features;

        // Read first 4 values as features
        for (int i = 0; i < 4; ++i)
        {
            getline(ss_test, value_test, ',');
            features.push_back(stod(value_test));
        }
        X_test.push_back(features);

        // Read last value as label and one-hot encode
        getline(ss_test, value_test);
        int label = stoi(value_test);
        vector<double> oneHot(3, 0.0);
        oneHot[label] = 1.0;
        Y_test.push_back(oneHot);
    }

    test_file.close();

    double accuracy = 0.0;
    for (int i = 0; i < NEPOCH; ++i)
    {
        double loss = trainOneEpoch(network, X, Y);
        double accuracy_i = computeAccuracy(network, X_test, Y_test);
        accuracy += accuracy_i;

        if (i % 10 == 0)
            cout << "Epoch " << (i + 1) << ", loss: " << loss << ", accuracy per iter: " << accuracy_i * 100 << "%" << endl;
    }

    // After training, predict and compare
    cout << "🔸 Predictions after training:" << endl;
    for (size_t i = 0; i < X_test.size(); ++i)
    {
        vector<double> y_hat_test = network.feed(X_test[i]);
        double predictedLabel = predictClass(y_hat_test);
        double realLabel = max_element(Y_test[i].begin(), Y_test[i].end()) - Y_test[i].begin();

        cout << "Sample " << i + 1 << ": Predicted label = " << predictedLabel << ", Real label = " << realLabel << endl;
    }

    return 0;
}