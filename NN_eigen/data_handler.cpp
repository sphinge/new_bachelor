#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

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
    int numFeatures = 4;
    int numLabels = 3;

    pair<MatrixXd, MatrixXd> data = readCSV("../iris_aug_test.csv", numFeatures, numLabels);

    MatrixXd X = data.first;
    MatrixXd y = data.second;

    cout << "Features (X):\n"
         << X << endl;
    cout << "One-hot encoded labels (y):\n"
         << y << endl;
    cout << "number features:"
         << X.cols() << endl;

    return 0;
}
