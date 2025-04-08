# QVAC Finetunable Model Base Class

This project provides a base class for all finetunable models in the QVAC ecosystem. It defines a common interface and basic functionality that all finetunable models should implement.

## Project Structure

```
qvac_finetunable_model_cpp/
├── include/           # Header files
│   └── finetunable_model.hpp
├── src/              # Source files
│   └── finetunable_model.cpp
├── tests/            # Test files
├── CMakeLists.txt    # Build configuration
└── README.md         # This file
```

## Building the Project

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

To use this base class in your project:

1. Include the header:
```cpp
#include "finetunable_model.hpp"
```

2. Create your model class by inheriting from `qvac::FinetunableModel`:
```cpp
class MyModel : public qvac::FinetunableModel {
    // Implement all virtual methods
};
```

## Required Implementations

Derived classes must implement the following pure virtual methods:

- `initialize(const std::string& config_path)`
- `finetune(const std::vector<std::string>& training_data, int epochs)`
- `save(const std::string& model_path)`
- `load(const std::string& model_path)`
- `predict(const std::string& input)`

## License

[Add your license information here] 