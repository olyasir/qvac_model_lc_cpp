# QVAC Finetunable Simple Model

This project provides an implementation of a linear classifier using the `FinetunableModel` base class from the QVAC ecosystem, with MNIST dataset support.

## Project Structure

```
qvac_finetunable_simple_model_cpp/
├── include/           # Header files
│   └── linear_classifier.hpp
├── src/              # Source files
│   └── linear_classifier.cpp
├── qvac_finetunable_model_cpp/  # Base class implementation
│   ├── include/
│   │   └── finetunable_model.h
│   └── src/
│       ├── finetunable_model.cpp
│       └── parameter_manager.cpp
├── example.cc        # Example usage with MNIST
├── CMakeLists.txt    # Build configuration
└── README.md         # This file
```

## Dependencies

- TVM (Tensor Virtual Machine)
- OpenCV
- MLC-LLM
- Tokenizers

## Building the Project

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

The example demonstrates loading and processing MNIST data with a linear classifier:

```cpp
#include "linear_classifier.hpp"
#include <opencv2/opencv.hpp>

int main() {
    // Initialize model with Vulkan device
    DLDevice device{kDLVulkan, 0};
    std::string model_path = "/path/to/model.so";
    std::unordered_map<std::string, std::string> config_filenames;
    
    // Create linear classifier
    qvac::LinearClassifier model("vulkan", model_path, config_filenames);

    // Load model weights
    std::string model_folder = "/path/to/model/folder";
    for (int i = 0; i < weight_files_num; i++) {
        std::string file_name = "params_shard_" + std::to_string(i) + ".bin";
        std::string full_path = model_folder + "/" + file_name;
        std::vector<uint8_t> bytes = get_span_from_file(full_path);
        model.set_weights_for_file(file_name, bytes, true);
    }

    // Load and process MNIST data
    auto sample = get_image("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
    Sample s = convertToTVMFormat(std::get<0>(sample), std::get<1>(sample), device);

    // Process input and get gradients
    model.process(s.input);
    auto gradient_map = model.get_gradients(s.input, s.labels);
}
```

## Implementation Details

This implementation demonstrates:
- Loading MNIST dataset in binary format
- Converting images to TVM NDArray format
- Processing images through a linear classifier
- Computing gradients for training
- Model weight management through ParameterManager

The model supports:
- MNIST image processing
- One-hot label encoding
- TVM-based inference
- Gradient computation for training

## License

[Add your license information here] 