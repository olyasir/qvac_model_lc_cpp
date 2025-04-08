#pragma once

#include <finetunable_model.hpp>
#include <string>
#include <vector>
#include <memory>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <span>
#include <param_manager.hpp>

namespace qvac {

using tvm::runtime::NDArray;
using tvm::runtime::memory::AllocatorType;

/**
 * @brief A simple linear classifier with two layers
 * 
 * This class implements a basic linear classifier using two fully connected layers
 * for classification tasks.
 */
class LinearClassifier : public FinetunableModel {
public:
    LinearClassifier() = default;
    ~LinearClassifier() override = default;

   
    LinearClassifier(const std::string& device_name,
                   const std::string& model_lib_name_with_path,
                   std::unordered_map<std::string, std::string>& config_filemap);


    /* Gradually append to the contents for shard files, and update cache once parameters are there */
    bool set_weights_for_file(const std::string& filename,
                                std::span<const uint8_t> bytes,
                                bool is_finished_current);

    /* Run inference and get translated output */
    int process(NDArray image);

    std::unordered_map<std::string, NDArray> get_gradients(NDArray image, NDArray label);

private:
    tvm::Device device_;
    std::unique_ptr<ParameterManager> param_manager_;
    tvm::runtime::Module model_;
    tvm::runtime::Module local_vm_;
    tvm::runtime::PackedFunc train_step_func_;
    tvm::runtime::PackedFunc val_step_func_; 
};

} // namespace qvac 