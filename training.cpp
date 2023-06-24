#include <eigen3/Eigen/Dense>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <filesystem>
// #include "nn.h"
#include "mlp.h"
using namespace std;

#define output_nodes 24
// #define num_batches 10

filesystem::path currentPath = filesystem::current_path();

Eigen::MatrixXd read_csv(string matrix_path){
    Eigen::MatrixXd matrix; 

    ifstream matrix_file(matrix_path);

    if (!matrix_file.is_open())
        cerr << "Failed to open the CSV file.\n";

    // Count the number of rows and columns in the CSV file in order to resize the matrix
    string line;
    int numRows = 0;
    int numCols = 0;
    getline(matrix_file, line); // to get the first line that are labels
    while (getline(matrix_file, line)) {
        ++numRows;
        if (numCols == 0) {
            std::stringstream ss(line);
            std::string cell;
            while (std::getline(ss, cell, ',')) {
                ++numCols;
            }
        }
    }

    matrix_file.clear();  // Clear the end-of-file flag
    matrix_file.seekg(0); // Reset the file pointer to the beginning

    matrix_file.close();

    ifstream n_matrix_file(matrix_path);

    if (!n_matrix_file.is_open())
        cerr << "Failed to open the CSV file.\n";

    // resize the matrix
    matrix.resize(numRows, numCols);

    // Read the data from the CSV file into the Eigen matrix
    int row = 0;
    getline(n_matrix_file, line); // to get the first line that are labels
    while (std::getline(n_matrix_file, line)) {
        std::stringstream ss(line);
        int col = 0;
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            matrix(row, col) = std::stod(cell);
            ++col;
        }
        ++row;
    }

    // Close the CSV file
    n_matrix_file.close();
    return matrix;
}

std::vector<int> create_hidden_layers(int layers, int neurons){
    std::vector<int> hidden_layers;
    for(int i = 0 ; i < layers ; i++)
        hidden_layers.push_back(neurons);
    return hidden_layers;
}

// void test1(Eigen::MatrixXd x_train_norm, Eigen::MatrixXd y_train, Eigen::MatrixXd x_test_norm, Eigen::MatrixXd y_test, string activation_function, int layers, int neurons){
//     // vector<int> hidden_layers = create_hidden_layers(layers, neurons);
//     vector<int> hidden_layers = create_hidden_layers(layers, neurons);
//     red_neuronal nn(x_train_norm.cols(), hidden_layers, output_nodes, activation_function);
//     nn.train(x_train_norm, y_train);
//     // testing
//     nn.evaluate(x_test_norm, y_test);
//     // adam optimizer?
// }

void test2(Eigen::MatrixXd x_train_norm, Eigen::MatrixXd y_train, Eigen::MatrixXd x_test_norm, Eigen::MatrixXd y_test, string activation_function, int layers, int neurons){
    vector<int> Layers = create_hidden_layers(layers, neurons);
    Layers.insert(Layers.begin(), x_train_norm.cols());
    Layers.push_back(y_train.cols());
    // Layers = [128, ... , 24]

    string filename = activation_function + "_" + to_string(layers) + "_" + to_string(neurons) + ".bin";
    MLP mlp;
    mlp.train(x_train_norm, y_train, Layers, filename);
}


int main() {

    map<int, string> act_func;
    act_func[1] = "relu";
    // act_func[2] = "sigmoid";
    // act_func[3] = "tanh";

    // Define the path to the CSV file
    string x_train_norm_path    = currentPath.string() + "/x_train_norm.csv";
    string x_test_norm_path     = currentPath.string() + "/x_test_norm.csv";
    string x_train_path         = currentPath.string() + "/x_train.csv";
    string x_test_path          = currentPath.string() + "/x_test.csv";
    string y_train_path         = currentPath.string() + "/y_train.csv";
    string y_test_path          = currentPath.string() + "/y_test.csv";

    Eigen::MatrixXd x_train, x_test, x_train_norm, x_test_norm, y_train, y_test;

    x_train         = read_csv(x_train_path);
    x_test          = read_csv(x_test_path);
    x_train_norm    = read_csv(x_train_norm_path);
    x_test_norm     = read_csv(x_test_norm_path);
    y_train         = read_csv(y_train_path);
    y_test          = read_csv(y_test_path);

    vector<string> files = {"relu_1_50.bin", "relu_1_100.bin", "relu_1_200.bin", 
                            "relu_2_50.bin", "relu_2_100.bin", "relu_2_200.bin", 
                            "relu_3_50.bin", "relu_3_100.bin", "relu_3_200.bin"};

    
    ifstream file(files[0]);

    if(!file){ // train
        cout << "training" << endl;
        cout << "\n\n---------------------------------------------  Normalized Data  ---------------------------------------------\n";
        for (const auto& pair : act_func){
            string activation_function = pair.second; // relu, sigmoid, tanh, etc...
            for (int layers = 1 ; layers <= 3 ; layers++){
                for(int neurons = 50 ; neurons <= 200 ; neurons = neurons*2){
                    cout << "--------- " << activation_function << " activation function | " << layers << " layer(s) | " << neurons << " neurons ---------\n";
                    // test1(x_train_norm, y_train, x_test_norm, y_test, activation_function, layers, neurons);
                    test2(x_train_norm, y_train, x_test_norm, y_test, activation_function, layers, neurons);
                    cout << "\n------------------------------------\n\n";
                }
            }
        }
    } else { // test
    cout << "testing" << endl;
        for(int i = 0 ; i < files.size() ; i++){
            MLP mlp;
            mlp.test(x_test_norm, y_test, files[i]);
        }
    }
    
    
}



/*
Assuming X has the following structure:
X_data = [
[1, 2, 3, 4, 5, ..., 128],
[1, 2, 3, 4, 5, ..., 128],
...
[1, 2, 3, 4, 5, ..., 128]
]
rows = n (number of audios)
cols = 128 (number of features in the feature vector)

Y_data = [
[0, 0, 0, 1, 0, ..., 0],
[0, 0, 0, 0, 0, ..., 1],
...
[0, 1, 0, 0, 0, ..., 0],
]
*/


// x       = [x1,x2,x3,...,xn] (1, n)
// w       = [w11,w12, w13, w14, w15, ... w1k] (n, k)
// hidden_1   = [x1*w11 + x2*w21 + ... xn*wk1,
//       x1*w21 + x2*w22 + ... xn*wk2,
//       ...
//       x1*w1n + x2*w2n + ... xn*wkn] (1, k)
// hidden_1 = [h11,h12,h13,...,h1k]
//
// s       = [s11,s12,s13,s14,s15, ... s1m] (k, m)
// hidden_2   = [h11*s11 + h12*s12 + ... + h1k*s1k,
//               h11*s21 + h12*s22 + ... + h1k*s2k,
//               ...
//               h11*sm1 + h12*sm2 + ... + h1k*smk]
// hidden_2 = [h21,h22,h23,...,h2k]
