#include "linear_classifier.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <memory>

namespace qvac {
using tvm::runtime::NDArray;
using tvm::runtime::memory::AllocatorType;
using tvm::runtime::ObjectRef;
using tvm::runtime::Array;
using tvm::runtime::ArrayNode;
using tvm::runtime::Downcast;
using tvm::Device;

std::pair<std::string, int> detect_device(std::string device) {
  using tvm::runtime::DeviceAPI;

  std::string device_name;
  int device_id;
  int delimiter_pos = device.find(":");

  if (delimiter_pos == std::string::npos) {
    device_name = device;
    device_id = 0;
  }
  else {
    device_name = device.substr(0, delimiter_pos);
    device_id = std::stoi(device.substr(delimiter_pos + 1, device.length()));
  }

  if (device_name == "auto") {
    bool allow_missing = true;

    if (DeviceAPI::Get(DLDevice{ kDLCUDA, 0 }, allow_missing))
      return { "cuda", device_id };

    if (DeviceAPI::Get(DLDevice{ kDLMetal, 0 }, allow_missing))
      return { "metal", device_id };

    if (DeviceAPI::Get(DLDevice{ kDLROCM, 0 }, allow_missing))
      return { "rocm", device_id };

    if (DeviceAPI::Get(DLDevice{ kDLVulkan, 0 }, allow_missing))
      return { "vulkan", device_id };

    if (DeviceAPI::Get(DLDevice{ kDLOpenCL, 0 }, allow_missing))
      return { "opencl", device_id };

    // TODO: Auto-detect devices for mali
    LOG(FATAL) << "Cannot auto detect device-name";
  }

  return { device_name, device_id };
}


DLDevice get_device(const std::string& device_name, int device_id) {
  if (device_name == "cuda")
    return { kDLCUDA, device_id };

  if (device_name == "metal")
    return { kDLMetal, device_id };

  if (device_name == "rocm")
    return { kDLROCM, device_id };

  if (device_name == "vulkan")
    return { kDLVulkan, device_id };

  if (device_name == "opencl" || device_name == "mali")
    return { kDLOpenCL, device_id };

  LOG(FATAL) << "Invalid device name: " << device_name
             << ". Please enter the device in the form 'device_name:device_id'"
                " or 'device_name', where 'device_name' needs to be one of 'cuda', 'metal', "
                "'vulkan', 'rocm', 'opencl', 'auto'.";

  return { kDLCPU, 0 };
}

LinearClassifier::LinearClassifier(const std::string& device_name,
                         const std::string& model_lib_name_with_path,
                         std::unordered_map<std::string, std::string>& config_filenames)
  { 
    auto pr_dev = detect_device(device_name);

    device_ = get_device( pr_dev.first,  pr_dev.second);
    // Load the model
    model_ = tvm::runtime::Module::LoadFromFile(model_lib_name_with_path, "so");
    auto fload_exec = model_->GetFunction("vm_load_executable");
    ICHECK(fload_exec.defined()) << "TVM runtime cannot find vm_load_executable";
    
    // Initialize VM
    local_vm_ = fload_exec();
    local_vm_->GetFunction("vm_initialization")(
        static_cast<int>(device_.device_type), 
        device_.device_id,
        static_cast<int>(AllocatorType::kPooled), 
        static_cast<int>(kDLCPU), 
        0,
        static_cast<int>(AllocatorType::kPooled));
    
   
    std::string gradient_function_name = "main_adjoint";
    std::string forward_function_name = "forward";
    train_step_func_ = local_vm_->GetFunction(gradient_function_name, false);
    val_step_func_ = local_vm_->GetFunction(forward_function_name, false);
    param_manager_ = std::make_unique<ParameterManager>(this->local_vm_, this->device_);
}

bool LinearClassifier::set_weights_for_file(const std::string& filename,
                                     std::span<const uint8_t> bytes,
                                     bool is_finished_current){

                                        return this->param_manager_->SetWeightsForFile(filename, bytes, is_finished_current);
                                      
                                        }
      


  std::unordered_map<std::string, NDArray> LinearClassifier::get_gradients(NDArray image, NDArray label)
  {
       ObjectRef obj_ref = train_step_func_(image, label, 
        param_manager_->getParam("fc1.weight"), 
        param_manager_->getParam("fc1.bias"),
        param_manager_->getParam("fc2.weight"), 
        param_manager_->getParam("fc2.bias"));
    
    std::unordered_map<std::string, NDArray> gradient_map;
        
    // Handle Array type
    if (obj_ref->IsInstance<ArrayNode>()) {
        Array<ObjectRef> grad_func_return = Downcast<Array<ObjectRef>>(obj_ref);
        ObjectRef grads = grad_func_return[1];
        
        Array<ObjectRef> grad_array = Downcast<Array<ObjectRef>>(grads);
        //std::cout << "Number of gradients: " << grad_array.size() << std::endl;
        

         NDArray gradient = Downcast<NDArray>(grad_array[2]);
         gradient_map["fc1.weight"] = gradient;
         
           gradient = Downcast<NDArray>(grad_array[3]);
          
         gradient_map["fc1.bias"] = gradient;
           gradient = Downcast<NDArray>(grad_array[4]);
          
         gradient_map["fc2.weight"] = gradient;
           gradient = Downcast<NDArray>(grad_array[5]);
         
         gradient_map["fc2.bias"] = gradient;
    } else {
        std::cout << "Unexpected return type" << std::endl;
    }

    return gradient_map;
}

int LinearClassifier::process(NDArray image)
{
  ObjectRef obj_ref = val_step_func_(image, param_manager_->getParam("fc1.weight"), 
            param_manager_->getParam("fc1.bias"), param_manager_->getParam("fc2.weight"), 
            param_manager_->getParam("fc2.bias"));
        
        auto predictions = obj_ref.as<NDArray>();
        ICHECK(predictions.defined()) << "Model predictions are not defined";
        
        // Create a CPU NDArray and copy predictions to it
        NDArray predictions_cpu = NDArray::Empty(predictions.value().Shape(), 
                                               predictions.value().DataType(), 
                                               Device{kDLCPU, 0});
        predictions.value().CopyTo(predictions_cpu);
        
         // Now access the CPU data
        auto pred_data = static_cast<float*>(predictions_cpu.ToDLPack()->dl_tensor.data);
        int num_classes = predictions.value().Shape()[1];

         // Find predicted class (index of highest probability)
        int pred_class = 0;
        float max_prob = pred_data[0];
        for (int j = 1; j < num_classes; j++) {
            if (pred_data[j] > max_prob) {
                max_prob = pred_data[j];
                pred_class = j;
            }
        }

        return pred_class;
}

}//namespace qvac