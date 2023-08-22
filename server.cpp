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

const int EXCHANGES = 2;
const int num_clients = 1;


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
        for (auto &layer : layers) {
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
};


class FederatedServer {
public:
    Network global_model;
    vector<Network> client_models;

    FederatedServer(const Network& model, int num_clients) : global_model(model), client_models(num_clients, model) {}


    string serialize_weights(const vector<vector<float>>& weights) {
        ostringstream oss;
        for (const auto& weight_list : weights) {
            for (float w : weight_list) {
                oss << w << ' ';  // Space-separated values within a layer
            }
            oss << ';';  // Semicolon-separated layers
        }
        return oss.str();
    }


    vector<vector<float>> deserialize_weights(const string& serialized_weights) {
        vector<vector<float>> weights;
        stringstream ss(serialized_weights);
        string layer_str;


        while (getline(ss, layer_str, ';')) {
            stringstream layer_ss(layer_str);
            vector<float> weight_list;
            float weight;
            while (layer_ss >> weight) {
                weight_list.push_back(weight);
            }
            weights.push_back(weight_list);
        }


        return weights;
    }

    std::string serialize_network(const Network &network) {
        std::ostringstream oss;
        // Serialize the network's topology
        for (size_t i = 0; i < network.layers.size(); ++i) {
            oss << network.layers[i].nNeurons << (i < network.layers.size() - 1 ? "," : ";");
        }
        // Serialize the network's weights
        oss << serialize_weights(network.getWeights());
        return oss.str();
    }

    Network deserialize_network(const std::string &serialized_network) {
        std::istringstream iss(serialized_network);
        std::string topology, weights_str;
        getline(iss, topology, ';');
        getline(iss, weights_str);

        std::vector<int> nNeuronsEachLayer;
        std::istringstream top_iss(topology);
        std::string neuron_count_str;
        while (getline(top_iss, neuron_count_str, ',')) {
            nNeuronsEachLayer.push_back(stoi(neuron_count_str));
        }

        Network network(nNeuronsEachLayer);
        network.setWeights(deserialize_weights(weights_str));
        return network;
    }


    void distribute_weights() {
        auto global_weights = global_model.getWeights();
        for (auto& client_model: client_models) {
            client_model.setWeights(global_weights);
        }
    }


    void aggregate_weights() {
        vector<vector<float>> summed_weights = global_model.getWeights();
        for (auto& weight_list: summed_weights) {
            for (float& w: weight_list) {
                w = 0;
            }
        }


        for (auto& client_model: client_models) {
            auto client_weights = client_model.getWeights();
            for (size_t i = 0; i < client_weights.size(); ++i) {
                for (size_t j = 0; j < client_weights[i].size(); ++j) {
                    summed_weights[i][j] += client_weights[i][j];
                }
            }
        }


        for (auto& weight_list: summed_weights) {
            for (float& w: weight_list) {
                w /= client_models.size();
            }
        }


        global_model.setWeights(summed_weights);
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
        TcpServer(std::string ip_address, int port, FederatedServer& fed_server);
        ~TcpServer();
        void startListen();
        void sendNetworkUpdateToClients(const std::string& networkData);
        std::string receiveNetworkUpdateFromClient();
        std::vector<int> all_connected_clients;
        

    private:
        FederatedServer& federated_server;
        std::string m_ip_address;
        int m_port;
        int m_socket;
        int m_new_socket;
        long m_incomingMessage;
        struct sockaddr_in m_socketAddress;
        socklen_t m_socketAddress_len;
        std::string m_serverMessage;

        int startServer();
        void closeServer();
        void acceptConnection(int &new_socket);
        std::string buildResponse();
        void sendResponse();
    };

} // namespace http


namespace http {


    TcpServer::TcpServer(std::string ip_address, int port, FederatedServer& fed_server)
        : m_ip_address(ip_address), m_port(port), federated_server(fed_server), m_socket(-1), 
          m_new_socket(-1), m_socketAddress(),
          m_socketAddress_len(sizeof(m_socketAddress))
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

    std::string TcpServer::receiveNetworkUpdateFromClient() {
        char buffer[BUFFER_SIZE] = {0};
        std::string accumulator;


        while (true) {
            int bytesReceived = recv(m_new_socket, buffer, BUFFER_SIZE, 0);
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


        log("====== Waiting for a connection ======\n\n\n");
        acceptConnection(m_new_socket);

        // Send the global model to each client
        std::string serialized_global_model = federated_server.serialize_network(federated_server.global_model);
        sendNetworkUpdateToClients(serialized_global_model);

        for (int iteration = 0; iteration < EXCHANGES; ++iteration) {

            // Wait for each client to send their updated model
            for (int i = 0; i < num_clients; ++i) {
                log("con");
                std::string client_model_str = receiveNetworkUpdateFromClient();
                log(client_model_str);
                Network client_model = federated_server.deserialize_network(client_model_str);
                federated_server.client_models[i] = client_model;
            }


            // Aggregate the client models to update the global model
            federated_server.aggregate_weights();

            //serialize aggregated network
            serialized_global_model = federated_server.serialize_network(federated_server.global_model);

            //send network back to clients
            sendNetworkUpdateToClients(serialized_global_model);
            
        }


        close(m_new_socket);

}



    void TcpServer::acceptConnection(int &new_socket) {
        new_socket = accept(m_socket, (sockaddr *)&m_socketAddress, &m_socketAddress_len);
        if (new_socket != -1) {
            all_connected_clients.push_back(new_socket);
        } else {
            std::ostringstream ss;
            ss << "Server failed to accept incoming connection from ADDRESS: " << inet_ntoa(m_socketAddress.sin_addr) << "; PORT: " << ntohs(m_socketAddress.sin_port);
            exitWithError(ss.str());
        }
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

    vector<int> nNeuronsEachLayer = {4, 8, 3}; 
    Network global_model(nNeuronsEachLayer);

    std::vector<std::vector<float>> weights = global_model.getWeights();

    for (size_t layer_idx = 0; layer_idx < weights.size(); ++layer_idx) {
        std::cout << "Weights for Layer " << layer_idx + 1 << ":\n";
        for (size_t weight_idx = 0; weight_idx < weights[layer_idx].size(); ++weight_idx) {
            std::cout << weights[layer_idx][weight_idx] << " ";
        }
        std::cout << "\n\n";
    }

    FederatedServer federated_server(global_model, num_clients);

    std::string printing = federated_server.serialize_network(federated_server.global_model);

    std::cout << printing << std::endl;

    http::TcpServer http_server("127.0.0.1", 8080, federated_server); // passing the reference here
    http_server.startListen();

    return 0;
}