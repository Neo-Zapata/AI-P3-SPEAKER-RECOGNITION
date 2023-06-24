#include <eigen3/Eigen/Dense>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <numeric>
#include <filesystem>
namespace fs = std::filesystem;
using namespace std;
#define parameters_filename "parameters.bin"


struct MLP{
    Eigen::MatrixXd X;
    Eigen::MatrixXd Y;
    int training_examples;
    map<string, Eigen::MatrixXd> parameters;
    double learningRate;
    int epochs;
    double gradientThreshold;
    string filename;

    MLP(){
        // hyperparameters
        this->learningRate  = 0.1;
        this->epochs        = 21;
        this->gradientThreshold = 5.0; // Gradient clipping threshold
    }

    vector<double> Matrix_to_vector(Eigen::MatrixXd mat){
        vector<double> retu;
        for(int i  = 0 ; i < mat.rows() ; i++){
            for(int j = 0 ; j < mat.cols() ; j++){
                double num = ((mat(i,j) >= 0.2) ? 1 : 0);
                retu.push_back(num);
            }
        }
        return retu;
    }

    void test(Eigen::MatrixXd X_test, Eigen::MatrixXd Y_test, string filename){
        this->filename = filename;
        cout << "\nTesting for " << filename << endl;
        this->parameters = retrieveMatricesFromFile();
        this->X = X_test;
        this->Y = Y_test;
        vector<double> Testing_error;
        vector<double> Precision, Recall, F1_score;

        for(int i = 0 ; i < X.rows() ; i++){
            Eigen::Matrix<double, 1, Eigen::Dynamic> Xi_matrix = X.row(i).transpose(); // (1, 128)
            Eigen::Matrix<double, 1, Eigen::Dynamic> Yi_matrix = Y.row(i).transpose(); // (1, 24)

            map<string, Eigen::MatrixXd> cache;
            Eigen::MatrixXd y_pred = forward_propagation(Xi_matrix, cache);
            double loss = calculateMSE(y_pred, Yi_matrix);

            vector<double> pred = Matrix_to_vector(y_pred);
            vector<double> _true = Matrix_to_vector(Yi_matrix);

            int TP = 0;
            int FP = 0;
            int TN = 0; 
            int FN = 0;
            get_var(TP, FP, TN, FN, pred, _true);

            double precision = calculatePrecision(TP, FP, TN, FN);
            double recall = calculateRecall(TP, FP, TN, FN);
            double f1Score = calculateF1Score(precision, recall);

            Precision.push_back(precision);
            Recall.push_back(recall);
            F1_score.push_back(f1Score);

            Testing_error.push_back(loss);

            // for(int i = 0 ; i < pred.size() ; i++){
            //     cout << pred[i] << " ";
            // }
            // cout << endl;

            // for(int i = 0 ; i < _true.size() ; i++){
            //     cout << _true[i] << " ";
            // }
            // cout << endl;
            // string line;
            // cin >> line;
        }
        double Loss_mean = accumulate(Testing_error.begin(), Testing_error.end(), 0.0) / Testing_error.size();
        cout << "\nTesting Error: " << Loss_mean;

        double precision_mean   = accumulate(Precision.begin(), Precision.end(), 0.0) / Precision.size();
        double recall_mean      = accumulate(Recall.begin(), Recall.end(), 0.0) / Recall.size();
        double f1_score_mean    = accumulate(F1_score.begin(), F1_score.end(), 0.0) / F1_score.size();
        cout << "\nPrecision: " << precision_mean;
        cout << "\nRecall: " << recall_mean;
        cout << "\nF1 Score: " << f1_score_mean << endl;
    }

    void get_var(int& TP, int& FP, int& TN, int& FN, const std::vector<double>& predicted_labels, const std::vector<double>& true_labels){
        // Calculate TP, FP, TN, FN
        for (std::size_t i = 0; i < predicted_labels.size(); ++i) {
            if (predicted_labels[i] == 1 && true_labels[i] == 1) {
                TP++;
            } else if (predicted_labels[i] == 1 && true_labels[i] == 0) {
                FP++;
            } else if (predicted_labels[i] == 0 && true_labels[i] == 0) {
                TN++;
            } else if (predicted_labels[i] == 0 && true_labels[i] == 1) {
                FN++;
            }
        }
    }

    double calculatePrecision(int& TP, int& FP, int& TN, int& FN) {
        // double truePositives = 0;
        // double falsePositives = 0;

        // for (size_t i = 0; i < predictions.size(); ++i) {
        //     if (predictions[i] == targets[i]) {
        //         truePositives++;
        //     } else if (predictions[i] != targets[i]) {
        //         falsePositives++;
        //     }
        // }

        if (TP + FP == 0) {
            return 0;  // handle the case when there are no positive predictions
        }

        return TP / (TP + FP);
    }

    double calculateRecall(int& TP, int& FP, int& TN, int& FN) {
        // double truePositives = 0;
        // double falseNegatives = 0;

        // for (size_t i = 0; i < predictions.size(); ++i) {
        //     if (predictions[i] == 1 && targets[i] == 1) {
        //         truePositives++;
        //     } else if (predictions[i] == 0 && targets[i] == 1) {
        //         falseNegatives++;
        //     }
        // }

        if (TP + FN == 0) {
            return 0;  // handle the case when there are no positive targets
        }

        return TP / (TP + FN);
    }

    double calculateF1Score(double precision, double recall) {
        if (precision + recall == 0) {
            return 0;  // handle the case when both precision and recall are 0
        }

        return 2 * (precision * recall) / (precision + recall);
    }

    void train(Eigen::MatrixXd X, Eigen::MatrixXd Y, vector<int> layers, string filename){
        this->X = X;
        this->Y = Y; 
        this->training_examples = X.rows();
        this->filename = filename;

        // Initialiize weights and biases
        for(int i = 1 ; i < layers.size() ; i++){
            this->parameters["W" + to_string(i)] = Eigen::MatrixXd::Random(layers[i-1], layers[i]) * 0.01;
            this->parameters["b" + to_string(i)] = Eigen::MatrixXd::Zero(1, layers[i]);
        }

        vector<double> Training_error;
        for(int epoch = 0 ; epoch < this->epochs ; epoch++){
            double loss;
            for (int i = 0; i < X.rows(); ++i) {
                Eigen::Matrix<double, 1, Eigen::Dynamic> x      = X.row(i).transpose(); // (1, 128)
                Eigen::Matrix<double, 1, Eigen::Dynamic> y_true = Y.row(i).transpose(); // (1, 24)

                map<string, Eigen::MatrixXd> cache;
                Eigen::MatrixXd y_pred  = forward_propagation(x, cache);
                loss = calculateMSE(y_pred, y_true);
                map<string, Eigen::MatrixXd> gradients = backward_propagation(y_pred, y_true, cache);
                update_parameters(gradients);
                Training_error.push_back(loss);
            }
            // if (epoch % 10 == 0) 
            //     std::cout << "Epoch " << epoch << ": Loss = " << loss << std::endl;
        }
        double Loss_mean = accumulate(Training_error.begin(), Training_error.end(), 0.0) / Training_error.size();
        cout << "Training Error: " << Loss_mean;
        storeMatricesToFile(this->parameters);
    }

    map<string, Eigen::MatrixXd> backward_propagation(Eigen::MatrixXd y_pred, Eigen::MatrixXd y_true, map<string, Eigen::MatrixXd>& cache){
        map<string, Eigen::MatrixXd> gradients;
        Eigen::MatrixXd dZ = (y_pred - y_true);

        for(int i = this->parameters.size() / 2 ; i >= 1 ; i--){
            Eigen::MatrixXd dW      = (cache.at("A" + to_string(i-1)).transpose() * dZ ) / training_examples;
            Eigen::MatrixXd dB      = dZ.colwise().mean();
            Eigen::MatrixXd new_dZ  = (dZ * this->parameters.at("W" + to_string(i)).transpose()).array() * relu(cache.at("Z" + to_string(i-1))).array();
            dZ = new_dZ; 
            gradients["dW" + to_string(i)] = dW.array().max(-gradientThreshold).min(gradientThreshold);;
            gradients["db" + to_string(i)] = dB.array().max(-gradientThreshold).min(gradientThreshold);;
        }
        return gradients;
    }

    Eigen::MatrixXd forward_propagation(Eigen::MatrixXd x, map<string, Eigen::MatrixXd>& cache){
        cache["A0"] = x;
        cache["Z0"] = x;
        for(int i = 0 ; i < this->parameters.size() / 2 ; i++){
            Eigen::MatrixXd Z = x * this->parameters.at("W" + to_string(i+1)) + this->parameters.at("b" + to_string(i+1));
            Eigen::MatrixXd A(Z.rows(), Z.cols());
            // if(i+1 == this->parameters.size()/2)
            //     A = softmax(Z);
            // else
            A = relu(Z);
            x = A;
            cache["Z" + to_string(i+1)] = Z;
            cache["A" + to_string(i+1)] = A;
        }
        return x;
    }

    double calculateMSE(Eigen::MatrixXd y_pred, Eigen::MatrixXd y_true){
        Eigen::MatrixXd diff = y_pred - y_true;
        double mse = (diff.array() + diff.array()).mean();
        return mse;
    }

    Eigen::MatrixXd relu(Eigen::MatrixXd Z){
        return Z.array().max(0.0);
    }

    Eigen::MatrixXd softmax(const Eigen::MatrixXd& Z) {
        Eigen::MatrixXd expZ = Z.array().exp();
        double sumExpZ = expZ.sum();
        Eigen::MatrixXd A = expZ / sumExpZ;
        return A;
    }


    void update_parameters(map<string, Eigen::MatrixXd> gradients){
        for(int i = 0 ; i < this->parameters.size() / 2 ; i++){
            this->parameters["W" + to_string(i+1)] = this->parameters["W" + to_string(i+1)] - learningRate * gradients.at("dW" + to_string(i+1));
            this->parameters["b" + to_string(i+1)] = this->parameters["b" + to_string(i+1)] - learningRate * gradients.at("db" + to_string(i+1));
        }
    }

    void storeMatricesToFile(const std::map<std::string, Eigen::MatrixXd>& matrices) {
        std::ofstream file(filename, std::ios::binary);

        if (file.is_open()) {
            size_t numMatrices = matrices.size();
            file.write(reinterpret_cast<char*>(&numMatrices), sizeof(numMatrices));

            for (const auto& entry : matrices) {
                const std::string& matrixName = entry.first;
                const Eigen::MatrixXd& matrix = entry.second;

                // Write matrix name
                size_t matrixNameSize = matrixName.size();
                file.write(reinterpret_cast<char*>(&matrixNameSize), sizeof(matrixNameSize));
                file.write(matrixName.c_str(), matrixNameSize);

                // Write matrix dimensions
                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Index rows = matrix.rows();
                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Index cols = matrix.cols();
                file.write(reinterpret_cast<char*>(&rows), sizeof(rows));
                file.write(reinterpret_cast<char*>(&cols), sizeof(cols));

                // Write matrix data
                file.write(reinterpret_cast<const char*>(matrix.data()), matrix.size() * sizeof(double));
            }

            file.close();
            // std::cout << "Matrices stored in file: " << filename << std::endl;
        } else {
            std::cerr << "Unable to open file: " << filename << std::endl;
        }
    }

    std::map<std::string, Eigen::MatrixXd> retrieveMatricesFromFile() {
        std::ifstream file(filename, std::ios::binary);
        std::map<std::string, Eigen::MatrixXd> matrices;

        if (file.is_open()) {
            size_t numMatrices;
            file.read(reinterpret_cast<char*>(&numMatrices), sizeof(numMatrices));

            for (size_t i = 0; i < numMatrices; ++i) {
                // Read matrix name
                size_t matrixNameSize;
                file.read(reinterpret_cast<char*>(&matrixNameSize), sizeof(matrixNameSize));
                std::string matrixName(matrixNameSize, '\0');
                file.read(&matrixName[0], matrixNameSize);

                // Read matrix dimensions
                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Index rows, cols;
                file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
                file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

                // Read matrix data
                Eigen::MatrixXd matrix(rows, cols);
                file.read(reinterpret_cast<char*>(matrix.data()), matrix.size() * sizeof(double));

                matrices[matrixName] = matrix;
            }

            file.close();
            // std::cout << "Matrices retrieved from file: " << filename << std::endl;
        } else {
            std::cerr << "Unable to open file: " << filename << std::endl;
        }

        return matrices;
    }
};