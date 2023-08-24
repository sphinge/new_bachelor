#include <vector>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

const int num_clients = 2;


using namespace std;


float unitrand() {
    return (2.0 * (float)rand() / RAND_MAX) - 1.0;
}


class Neuron {
public:
    unsigned int nInputs;
    vector<float> weights;
    float bias;
    
    Neuron(unsigned int _nInputs) : nInputs(_nInputs) {
        weights = vector<float>(nInputs);
        for (auto &w : weights) {
            w = unitrand();
        }
        bias = unitrand();
    }
};


class Layer {
public:
    int nInputs, nNeurons;
    vector<Neuron> neurons;


    Layer(int _nInputs, int _nNeurons) : nInputs(_nInputs), nNeurons(_nNeurons) {
        for (unsigned int i = 0; i < nNeurons; ++i) {
            neurons.push_back(Neuron(nInputs));
        }
    }

};


class Network {
public:
    vector<Layer> layers;


    Network(vector<int> nNeuronsEachLayer) {
        unsigned int i;
        for (i = 0; i < nNeuronsEachLayer.size() - 1; ++i) {
            layers.push_back(Layer(nNeuronsEachLayer[i], nNeuronsEachLayer[i + 1]));
        }
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

class GlobalAggregator {
public:
    Network globalModel;


    GlobalAggregator(vector<int> nNeuronsEachLayer) : globalModel(nNeuronsEachLayer) {}

    // Aggregate models from clients
    void aggregate(const std::vector<std::string>& serializedModels) {
        std::vector<std::vector<float>> sumWeights = globalModel.getWeights();
        std::vector<float> sumBiases = globalModel.getBiases();

        for (const auto& serialized : serializedModels) {
            Network clientModel = globalModel;
            clientModel.deserialize(serialized);

            auto clientWeights = clientModel.getWeights();
            auto clientBiases = clientModel.getBiases();

            for (size_t i = 0; i < sumWeights.size(); i++) {
                for (size_t j = 0; j < sumWeights[i].size(); j++) {
                    sumWeights[i][j] += clientWeights[i][j];
                }
            }

            for (size_t i = 0; i < sumBiases.size(); i++) {
                sumBiases[i] += clientBiases[i];
            }
        }

        // Average the weights and biases
        for (auto& neuronWeights : sumWeights) {
            for (float& weight : neuronWeights) {
                weight /= serializedModels.size();
            }
        }

        for (float& bias : sumBiases) {
            bias /= serializedModels.size();
        }

        globalModel.setWeights(sumWeights);
        globalModel.setBiases(sumBiases);
    }

    // Get the serialized global model to distribute to clients
    std::string getSerializedGlobalModel() {
        return globalModel.serialize();
    }
};

namespace {
    const int BUFFER_SIZE = 30720;


    void log(const std::string &message) {
        std::cout << message << std::endl;
    }


    void exitWithError(const std::string &errorMessage) {
        perror(errorMessage.c_str());  // Use perror for more detailed error
        log("ERROR: " + errorMessage);
        exit(1);
    }
}

namespace http
{

    class TcpServer {
    public:
        TcpServer(std::string ip_address, int port, GlobalAggregator& globalModel);
        ~TcpServer();
        void startListen();
        void sendNetworkUpdateToClients(const std::string& networkData);
        std::string receiveNetworkUpdateFromClient(int client_socket);
        std::vector<int> all_connected_clients;
        

    private:

        std::string m_ip_address;
        int m_port;
        int m_socket;
        int m_new_socket;
        long m_incomingMessage;
        struct sockaddr_in m_socketAddress;
        socklen_t m_socketAddress_len;
        std::string m_serverMessage;
        GlobalAggregator& m_globalModel;
        int startServer();
        void closeServer();
        int acceptConnection();
        std::string buildResponse();
        void sendResponse();
    };

} // namespace http


namespace http {


    TcpServer::TcpServer(std::string ip_address, int port, GlobalAggregator& globalModel)
        : m_ip_address(ip_address), m_port(port), m_socket(-1), 
          m_new_socket(-1), m_socketAddress(),
          m_socketAddress_len(sizeof(m_socketAddress)), m_globalModel(globalModel)
    {
        m_socketAddress.sin_family = AF_INET;
        m_socketAddress.sin_port = htons(m_port);
        m_socketAddress.sin_addr.s_addr = inet_addr(m_ip_address.c_str());


        if (startServer() != 0) {
            std::ostringstream ss;
            ss << "Failed to start server with PORT: " << ntohs(m_socketAddress.sin_port);
            log(ss.str());
        }
    }


    void TcpServer::sendNetworkUpdateToClients(const std::string& networkData) {
        std::string dataToSend = networkData + "\n";
        for (int client_socket : all_connected_clients) {
            send(client_socket, dataToSend.c_str(), dataToSend.size(), 0);
        }
    }

    std::string TcpServer::receiveNetworkUpdateFromClient(int client_socket) {
        char buffer[BUFFER_SIZE] = {0};
        std::string accumulator;

        while (true) {
            int bytesReceived = recv(client_socket, buffer, BUFFER_SIZE, 0);
            if (bytesReceived < 0) {
                // Handle the error here
                return "";
            } else if (bytesReceived == 0) {
                // Connection closed by the client
                break;
            }

            accumulator += std::string(buffer, bytesReceived);

            if (accumulator.find("\n") != std::string::npos) {
                // If delimiter found, break the loop
                break;
            }
        }

        // Removing the delimiter before returning
        size_t endPos = accumulator.find("\n");
        if (endPos != std::string::npos) {
            accumulator = accumulator.substr(0, endPos);
        }

        return accumulator;
    }



    TcpServer::~TcpServer() {
        for (int client_socket : all_connected_clients) {
            close(client_socket);
        }
        closeServer();
    }

    int TcpServer::startServer()
    {
        m_socket = socket(AF_INET, SOCK_STREAM, 0);
        if (m_socket < 0)
        {
            exitWithError("Cannot create socket");
            return 1;
        }

        if (bind(m_socket, (sockaddr *)&m_socketAddress, m_socketAddress_len) < 0)
        {
            exitWithError("Cannot connect socket to address");
            return 1;
        }

        return 0;
    }

    void TcpServer::closeServer()
    {
        close(m_socket);
        close(m_new_socket);
        exit(0);
    }

    void TcpServer::startListen()
    {
        if (listen(m_socket, 20) < 0)
        {
            exitWithError("Socket listen failed");
        }

        std::ostringstream ss;
        ss << "\n*** Listening on ADDRESS: " << inet_ntoa(m_socketAddress.sin_addr) << " PORT: " << ntohs(m_socketAddress.sin_port) << " ***\n\n";
        log(ss.str());


        // Array to store client socket descriptors
        std::vector<int> client_sockets(num_clients); 

        for (int i = 0; i < num_clients; i++) 
        {
            log("====== Waiting for client " + to_string(i+1) + "======\n");
            client_sockets[i] = acceptConnection();
            log(std::to_string(i+1) + ": client connected");
        }

        // Send the global model to each client
        std::string serializedGlobalModel = m_globalModel.getSerializedGlobalModel();
        log("Network sending model: " + serializedGlobalModel + "\n\n");
        sendNetworkUpdateToClients(serializedGlobalModel);

        // Let's assume a constant for the number of exchanges
        const int EXCHANGES = 10;
        for (int iteration = 0; iteration < EXCHANGES; ++iteration) 
        {
            std::vector<std::string> receivedModels;
            log("Waiting for models...\n");
            sleep(5);

            // Wait for each client to send their updated model
            for (int i = 0; i < num_clients; ++i) 
            {
                std::string clientModelUpdate = receiveNetworkUpdateFromClient(client_sockets[i]);
                log("Network update received from client" + to_string(i));
                receivedModels.push_back(clientModelUpdate);
            }

            // Aggregate the client models to update the global model
            m_globalModel.aggregate(receivedModels);

            // Serialize aggregated network
            serializedGlobalModel = m_globalModel.getSerializedGlobalModel();
            log("This is the new model: " + serializedGlobalModel + "\n\n");

            // Send the updated global model back to clients
            sendNetworkUpdateToClients(serializedGlobalModel);
            log("Model sent");
        }

        // Close all client sockets
        for (int i = 0; i < num_clients; ++i) 
        {
            close(client_sockets[i]);
        }
    }


    int TcpServer::acceptConnection() 
    {
        int new_socket = accept(m_socket, (sockaddr *)&m_socketAddress, &m_socketAddress_len);
        if (new_socket != -1) 
        {
            all_connected_clients.push_back(new_socket);

            // Log the acceptance of the new connection
            std::ostringstream ss;
            ss << "Accepted connection from ADDRESS: " << inet_ntoa(m_socketAddress.sin_addr) << "; PORT: " << ntohs(m_socketAddress.sin_port);
            log(ss.str());

            return new_socket; // Return the socket descriptor
        } 
        else 
        {
            std::ostringstream ss;
            ss << "Server failed to accept incoming connection from ADDRESS: " << inet_ntoa(m_socketAddress.sin_addr) << "; PORT: " << ntohs(m_socketAddress.sin_port);
            exitWithError(ss.str());
        }

        return -1; // If we get here, there was an error
    }


    std::string TcpServer::buildResponse()
    {
        std::ostringstream ss;
        ss << "Received updated network.";

        return ss.str();
    }

    void TcpServer::sendResponse()
    {
        int bytesSent;
        long totalBytesSent = 0;

        while (totalBytesSent < m_serverMessage.size())
        {
            bytesSent = send(m_new_socket, m_serverMessage.c_str(), m_serverMessage.size(), 0);
            if (bytesSent < 0)
            {
                break;
            }
            totalBytesSent += bytesSent;
        }

        if (totalBytesSent == m_serverMessage.size())
        {
            log("------ Server Response sent to client ------\n\n");
        }
        else
        {
            log("Error sending response to client.");
        }
    }

}

int main()
{
    srand(time(NULL));

    vector<int> nNeurons = {4, 8, 3};  // example architecture: 2 neurons in input layer, 3 in hidden layer, 1 in output layer
    GlobalAggregator globalModel(nNeurons);

    http::TcpServer http_server("127.0.0.1", 8080, globalModel); // passing the reference here
    http_server.startListen();

    return 0;
}
