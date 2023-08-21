// g++ iris_clf.cpp -I/usr/include/eigen3  -o clf
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

using namespace Eigen;
using namespace std;

class NeuralNetwork
{
public:
    NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
        : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize)
    {
        // Initialize weights and biases
        W1 = MatrixXd::Random(hiddenSize, inputSize);
        b1 = VectorXd::Zero(hiddenSize);
        W2 = MatrixXd::Random(outputSize, hiddenSize);
        b2 = VectorXd::Zero(outputSize);
    }

    VectorXd relu(const VectorXd &z)
    {
        return z.array().max(0.0);
    }
    VectorXd softmax(const VectorXd &z)
    {
        VectorXd expZ = z.array().exp();
        return expZ / expZ.sum();
    }
    VectorXd relu_derivative(const VectorXd &z)
    {
        return (z.array() > 0.0).cast<double>();
    }
    VectorXd forward(const VectorXd &x)
    {
        hiddenLayer = relu(W1 * x + b1);
        outputLayer = softmax(W2 * hiddenLayer + b2);
        return outputLayer;
    }

    void train(const MatrixXd &X, const MatrixXd &y, double learningRate, int epochs)
    {
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            for (int i = 0; i < X.rows(); ++i)
            {
                // Forward pass
                VectorXd x = X.row(i);
                VectorXd target = y.row(i);
                forward(x);

                // Backpropagation
                VectorXd dL_dz2 = outputLayer - target;
                VectorXd dL_dh = W2.transpose() * dL_dz2;
                VectorXd dL_dz1 = dL_dh.array() * relu_derivative(hiddenLayer).array();
                ;

                // Update weights and biases
                W2 -= learningRate * dL_dz2 * hiddenLayer.transpose();
                b2 -= learningRate * dL_dz2;
                W1 -= learningRate * dL_dz1 * x.transpose();
                b1 -= learningRate * dL_dz1;
            }
        }
    }

    int predict(const VectorXd &x)
    {
        VectorXd probabilities = forward(x);
        int predictedClass = 0;
        double maxProbability = probabilities[0];

        for (int i = 1; i < probabilities.size(); ++i)
        {
            if (probabilities[i] > maxProbability)
            {
                maxProbability = probabilities[i];
                predictedClass = i;
            }
        }

        return predictedClass;
    }

private:
    int inputSize;
    int hiddenSize;
    int outputSize;
    MatrixXd W1, W2;
    VectorXd b1, b2;
    VectorXd hiddenLayer, outputLayer;
};

pair<MatrixXd, MatrixXd> readCSV(const string &filename, int numFeatures, int numLabels)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    vector<vector<double>> data;
    string line;
    while (getline(file, line))
    {
        stringstream ss(line);
        vector<double> row;
        double val;
        while (ss >> val)
        {
            row.push_back(val);
            if (ss.peek() == ',')
            {
                ss.ignore();
            }
        }
        data.push_back(row);
    }

    int numRows = data.size();
    MatrixXd X(numRows, numFeatures);
    MatrixXd y(numRows, numLabels);
    y.setZero();

    for (int i = 0; i < numRows; ++i)
    {
        for (int j = 0; j < numFeatures; ++j)
        {
            X(i, j) = data[i][j];
        }

        int labelIndex = static_cast<int>(data[i].back()); // Last value is the label
        y(i, labelIndex) = 1.0;
    }

    return make_pair(X, y);
}

int main()
{
    // Load Iris dataset (you need to replace this with your data loading code)

    string filename_train = "../iris_data/iris_aug_train.csv"; // Replace with your CSV file name
    int numFeatures = 4;                             // Number of features
    int numLabels = 3;                               // Number of label classes

    pair<MatrixXd, MatrixXd> data_train = readCSV(filename_train, numFeatures, numLabels);
    MatrixXd X_train = data_train.first;
    MatrixXd y_train = data_train.second;
    // Create and train the neural network
    int inputSize = X_train.cols();
    int hiddenSize = 10;
    int outputSize = y_train.cols();
    NeuralNetwork nn(inputSize, hiddenSize, outputSize);

    double learningRate = 0.01;
    int epochs = 1000;

    nn.train(X_train, y_train, learningRate, epochs);

    string filename_test = "../iris_data/iris_aug_test.csv"; // Replace with your CSV file name

    pair<MatrixXd, MatrixXd> data_test = readCSV(filename_test, numFeatures, numLabels);
    MatrixXd X_test = data_test.first;
    MatrixXd y_test = data_test.second;

    // Predict and print results
    for (int i = 0; i < X_test.rows(); ++i)
    {
        VectorXd x = X_test.row(i);
        int activatedNeurons = nn.predict(x);
        cout << "Sample " << i + 1 << ": Activated neurons in hidden layer = " << activatedNeurons << endl;
    }

    return 0;
}
