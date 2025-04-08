#include "linear_classifier.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <tuple>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// Helper function to read integers from MNIST binary files
int read_int(std::ifstream& file) {
    int value;
    file.read(reinterpret_cast<char*>(&value), sizeof(value));
    // MNIST files are in big-endian format, so we need to swap bytes
    return ((value & 0xFF) << 24) | ((value & 0xFF00) << 8) | 
           ((value & 0xFF0000) >> 8) | ((value & 0xFF000000) >> 24);
}

struct Sample {
    //tvm::runtime::NDArray input;
    tvm::runtime::NDArray input;
    tvm::runtime::NDArray labels;
    int labels_int;
};


// Load MNIST Data
std::tuple<cv::Mat, int> get_image(const std::string& image_file, const std::string& label_file) {

    std::ifstream img_stream(image_file, std::ios::binary);
    std::ifstream lbl_stream(label_file, std::ios::binary);

    if (!img_stream.is_open() || !lbl_stream.is_open()) {
        throw std::runtime_error("Failed to open MNIST files.");
    }

    // Read headers
    int magic_number = read_int(img_stream);
    int num_images = read_int(img_stream);
    int rows = read_int(img_stream);
    int cols = read_int(img_stream);
    int lbl_magic = read_int(lbl_stream);
    int num_labels = read_int(lbl_stream);

    if (magic_number != 2051 || lbl_magic != 2049 || num_images != num_labels) {
        throw std::runtime_error("Invalid MNIST file format.");
    }

    //for (int i = 0; i < num_images; i++) {
        cv::Mat img(rows, cols, CV_8UC1);
        img_stream.read(reinterpret_cast<char*>(img.data), rows * cols);

        uint8_t label;
        lbl_stream.read(reinterpret_cast<char*>(&label), 1);
        return std::make_tuple(img, label);
    //}
}


Sample convertToTVMFormat(cv::Mat image, uint8_t label, DLDevice device) {
    Sample sample;  // Create Sample directly instead of using unique_ptr
   
    const int num_classes_ = 10; // MNIST has 10 classes
    cv::Mat float_img;
    image.convertTo(float_img, CV_32F, 1.0/255.0);  // Normalize to [0,1]
    
    // Apply MNIST normalization (mean=0.1307, std=0.3081)
    float_img = (float_img - 0.1307f) / 0.3081f;
    
    std::vector<float> img_data(float_img.ptr<float>(), 
                              float_img.ptr<float>() + float_img.total());
    
    std::vector<int64_t> shape = { 1, image.rows, image.cols};  // NCHW format
    tvm::runtime::NDArray arr = tvm::runtime::NDArray::Empty(
        shape, DLDataType{kDLFloat, 32, 1}, device);
    arr.CopyFromBytes(img_data.data(), img_data.size() * sizeof(float));
    sample.input = arr;        // Create one-hot encoding for the label
    std::vector<int64_t> label_shape = {num_classes_};  // Assuming num_classes_ is a member variable
    std::vector<float> one_hot(num_classes_, 0.0f);
    if (label >= 0 && label < num_classes_) {
        one_hot[label] = 1.0f;
    }

    tvm::runtime::NDArray label_arr = tvm::runtime::NDArray::Empty(
        label_shape, DLDataType{kDLFloat, 32, 1}, device);
    label_arr.CopyFromBytes(one_hot.data(), one_hot.size() * sizeof(float));
    sample.labels = label_arr;
    sample.labels_int = label;
    return sample;
}

std::vector<uint8_t> get_span_from_file(std::string file_path) {
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + file_path);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Failed to read file: " + file_path);
    }
    return buffer;  // Return the vector directly
}

int main(int argc, char* argv[]) {
    DLDevice device{kDLVulkan, 0};;
    std::string model_path = (argc > 1) ? argv[1] : "/home/ubuntu/qvac_trainer/simple_model_compiled/model.so";
    
    std::unordered_map<std::string, std::string> config_filenames;
    // Create and run trainer - pass dataloader by value as per Trainer constructor
    qvac::LinearClassifier model("vulkan", model_path, config_filenames);

    std::string model_folder = "/home/ubuntu/qvac_trainer/simple_model_compiled";
    int weight_files_num=1;
     for (int i=0;i<weight_files_num; i++)
    {
        std::string file_name = std::string("params_shard_").append(std::to_string(i)).append(".bin");
        std::string full_path = std::string("").append(model_folder).append("/"+file_name);
        std::vector<uint8_t> bytes = get_span_from_file(full_path);
       model.set_weights_for_file(file_name, bytes, true); 
    }


    auto sample = get_image("/home/ubuntu/qvac_trainer/data/MNIST/raw/train-images-idx3-ubyte", "/home/ubuntu/qvac_trainer/data/MNIST/raw/train-labels-idx1-ubyte");
    Sample s = convertToTVMFormat(std::get<0>(sample), std::get<1>(sample), device);


    model.process(s.input); 
    std::unordered_map<std::string, tvm::runtime::NDArray> gradient_map = model.get_gradients(s.input, s.labels);  // Train for 10 epochs
    
    return 0;
}