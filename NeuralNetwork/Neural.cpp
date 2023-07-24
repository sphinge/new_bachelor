//./Neural /home/wiktoria/Desktop/Thesis/Shark/examples/Supervised/data/mnist_subset.libsvm

#include <shark/Models/LinearModel.h>
#include <shark/Models/ConcatenatedModel.h>
#include <shark/ObjectiveFunctions/ErrorFunction.h>
#include <shark/ObjectiveFunctions/Loss/CrossEntropy.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/Algorithms/GradientDescent/Adam.h>
#include <shark/Data/SparseData.h>
#include <iostream>
#include <fstream>

using namespace shark;

std::string runPythonCommand(const std::string &command)
{
    std::array<char, 128> buffer;
    std::string result;
    FILE *pipe = popen(command.c_str(), "r");
    if (!pipe)
    {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr)
    {
        result += buffer.data();
    }
    pclose(pipe);
    return result;
}

// Function to plot learning curve and accuracy
void plotLearningCurve(const std::vector<std::size_t> &iterations, const std::vector<double> &errors, const std::vector<double> &accuracy)
{
    // Save the data to a CSV file
    std::ofstream file("learning_curve.csv");
    file << "Iterations,Error,Accuracy" << std::endl;
    for (std::size_t i = 0; i < iterations.size(); ++i)
    {
        file << iterations[i] << "," << errors[i] << "," << accuracy[i] << std::endl;
    }
    file.close();

    // Use Python to plot the learning curve with a delay of 2 seconds
    // std::string pythonCommand = "python -c \"import pandas as pd; import matplotlib.pyplot as plt; import time; data = pd.read_csv('learning_curve.csv'); plt.plot(data['Iterations'], data['Error'], label='Error'); plt.plot(data['Iterations'], data['Accuracy'], label='Accuracy', linestyle='dashed'); plt.xlabel('Iterations'); plt.ylabel('Error / Accuracy'); plt.title('Learning Curve and Accuracy'); plt.legend(); plt.show(); time.sleep(2)\"";
    // runPythonCommand(pythonCommand);
}

// Function to plot classification error
// Function to plot classification error and accuracy
void plotClassificationError(const std::string &trainError, const std::string &testError, const std::vector<double> &accuracy)
{
    // Convert accuracy data to a string
    std::ostringstream accuracyStream;
    accuracyStream << "[";
    for (std::size_t i = 0; i < accuracy.size(); ++i)
    {
        if (i > 0)
            accuracyStream << ",";
        accuracyStream << accuracy[i];
    }
    accuracyStream << "]";
    std::string accuracyData = accuracyStream.str();

    // Use Python to plot the classification error and accuracy with a delay of 2 seconds
    // std::string command = "python -c \"import matplotlib.pyplot as plt; import time; train_error = " + trainError + "; test_error = " + testError + "; accuracy = " + accuracyData + "; plt.plot(train_error, label='Train'); plt.plot(test_error, label='Test'); plt.plot(accuracy, label='Accuracy', linestyle='dashed'); plt.xlabel('Iterations'); plt.ylabel('Error / Accuracy'); plt.title('Classification Error and Accuracy'); plt.legend(); plt.show(); time.sleep(2)\"";
    // runPythonCommand(command);
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: " << argv[0] << "/home/wiktoria/Desktop/Thesis/Shark/examples/Supervised/data/mnist_subset.libsvm" << std::endl;
        return 1;
    }
    std::size_t hidden1 = 200;
    std::size_t hidden2 = 100;
    // std::size_t hidden3 = 100;
    std::size_t iterations = 1000;
    std::size_t batchSize = 256;

    LabeledData<RealVector, unsigned int> data;
    importSparseData(data, argv[1], 0, batchSize);
    data.shuffle();
    auto test = splitAtElement(data, 70 * data.numberOfElements() / 100);
    std::size_t numClasses = numberOfClasses(data);

    typedef LinearModel<RealVector, RectifierNeuron> DenseLayer;

    DenseLayer layer1(data.inputShape(), hidden1);
    DenseLayer layer2(layer1.outputShape(), hidden2);
    // DenseLayer layer3(layer2.outputShape(), hidden3);
    LinearModel<RealVector> output(layer2.outputShape(), numClasses);
    auto network = layer1 >> layer2 >> output;
    // auto network = layer1 >> output;

    CrossEntropy<unsigned int, RealVector> loss;
    ErrorFunction<RealVector> error(data, &network, &loss, true);

    std::cout << "training network" << std::endl;
    initRandomNormal(network, 0.001);
    Adam<RealVector> optimizer;
    optimizer.setEta(0.0001);
    error.init();
    optimizer.init(error);

    std::vector<std::size_t> iterationsVec;
    std::vector<double> errorVec;
    std::vector<double> accuracyVec;

    for (std::size_t i = 0; i != iterations; ++i)
    {
        std::cout << "i " << i << std::endl;
        optimizer.step(error);
        iterationsVec.push_back(i + 1);
        errorVec.push_back(optimizer.solution().value);

        // Calculate accuracy at current iteration
        Data<RealVector> currentPrediction = network(data.inputs());
        double currentAccuracy = 1.0 - loss.eval(data.labels(), currentPrediction);
        accuracyVec.push_back(currentAccuracy);
    }

    network.setParameterVector(optimizer.solution().point);

    ZeroOneLoss<unsigned int, RealVector> loss01;
    Data<RealVector> predictionTrain = network(data.inputs());
    std::cout << "classification error, train: " << loss01.eval(data.labels(), predictionTrain) << std::endl;

    Data<RealVector> prediction = network(test.inputs());
    std::cout << "classification error, test: " << loss01.eval(test.labels(), prediction) << std::endl;

    // Plot the learning curve
    plotLearningCurve(iterationsVec, errorVec, accuracyVec);

    // Prepare classification error data for plotting
    std::ostringstream trainErrorStream, testErrorStream;
    trainErrorStream << "[" << loss01.eval(data.labels(), predictionTrain) << "]";
    testErrorStream << "[" << loss01.eval(test.labels(), prediction) << "]";
    std::string trainError = trainErrorStream.str();
    std::string testError = testErrorStream.str();

    // Plot the classification error
    // plotClassificationError(trainError, testError, accuracyVec);

    return 0;
}