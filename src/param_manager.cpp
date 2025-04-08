#include "param_manager.hpp"
#include <fstream>
#include <filesystem>

using namespace tvm::runtime;
namespace fs = std::filesystem;

ParameterManager::ParameterManager(Module local_vm, tvm::Device device) 
    : local_vm_(local_vm), 
      device_(device) {

  model_metadata_ = ::mlc::llm::ModelMetadata::FromModule(local_vm_, {});
  
  std::ifstream file("/home/ubuntu/qvac_trainer/simple_model_compiled/ndarray-cache.json");//TODO olya update
  std::stringstream buffer;
  buffer << file.rdbuf();
            
  ndarray_cache_metadata_ = tvm::runtime::relax_vm::NDArrayCacheMetadata::LoadFromStr(
      buffer.str(),
      ""
    );

  params_shard_files_to_load_ = ndarray_cache_metadata_.records.size();
  device_ = device;
}

void ParameterManager::UpdateNDArrayCache(
    const std::string& filename) {
  using tvm::runtime::ShapeTuple;
  using FileRecord = tvm::runtime::relax_vm::NDArrayCacheMetadata::FileRecord;
  using ParamRecord = FileRecord::ParamRecord;

  const size_t filename_chars = std::string("params_shard_").size();
  size_t stop_idx = filename.find_last_of('.');
  std::string remaining = filename.substr(filename_chars, stop_idx - filename_chars);
  size_t file_record_idx = std::stoi(remaining);
  const FileRecord& file_record = ndarray_cache_metadata_.records[file_record_idx];
  size_t total_param_records = file_record.records.size();
  Array<NDArray> params;
  const auto& params_shard_file = params_shard_filemap_.at(filename);
  //const auto& params_shard_file = model.get_weights_for_file(filename);
  Optional<NDArray> staging_buffer;

  std::cerr << filename << " has these many parameter records: " << total_param_records << '\n';

  params.reserve(total_param_records);

  for (size_t i = 0; i < total_param_records; ++i) {
    const ParamRecord& param_record = file_record.records[i];
    params.push_back(param_record.Load(this->device_,
                                     &params_shard_file,
                                     &staging_buffer));
  }

  const PackedFunc* fload_cache_update = Registry::Get("vm.builtin.ndarray_cache.update");
  ICHECK(fload_cache_update) << "TVM runtime cannot find vm.builtin.ndarray_cache.update";

  /* Update the global cache with all the various parameters */
  for (size_t i = 0; i < params.size(); ++i) {
    (*fload_cache_update)(file_record.records[i].name, params[i], true);
  }
}

void ParameterManager::LoadParams() {

    constexpr const char* name_loader = "vm.builtin.param_array_from_cache_by_name";
    const PackedFunc* fload_params = Registry::Get(name_loader);
    ICHECK(fload_params) << "Cannot find env function: " << name_loader;

    Array<String> param_names;
    param_names.reserve(model_metadata_.params.size());

    for (const auto& param : model_metadata_.params) {
      param_names.push_back(param.name);
    }
  TVMRetValue ret = (*fload_params)(param_names);
  params_ = ret.AsObjectRef<Array<NDArray>>();

  // for(int i =0 ; i<4; ++i)
  // {
  //   std::cout << param_names[param_names.size()-1-i]<<"\n";
  // }

  // std::vector<int64_t> shape = {2, 3};  // example shape
  // NDArray zero_array = NDArray::Empty(shape, DataType::Float(32), device_);
  // std::vector<float> zeros(2 * 3, 0.0f);  // size should match total elements
  // zero_array.CopyFromBytes(zeros.data(), zeros.size() * sizeof(float));

  // std::cout<< params_[0].Shape()<<"\n";
  // params_.Set(0, zero_array);

  // std::cout<< "parameters size: " <<  params_.size()<<"\n";
  // std::cout<< params_[0].Shape()<<"\n";
  ClearNDArrayCache();
}


void print_tensor( NDArray tensor, std::string name, int amount_to_print )
{
   // Create a CPU NDArray and copy param to it for printing
    NDArray param_cpu = NDArray::Empty(tensor.Shape(), tensor.DataType(), tvm::Device{kDLCPU, 0});
    tensor.CopyTo(param_cpu);
    auto param_data = static_cast<float*>(param_cpu.ToDLPack()->dl_tensor.data);
    
    std::cout << "First  values of "<<name<<": ";
    for (int i = 0; i < amount_to_print; i++) {
        std::cout << param_data[i] << " ";
    }
    std::cout << std::endl;
}

tvm::runtime::NDArray ParameterManager::getParam(std::string param_name)
{
    // Find parameter index in model metadata
    for (size_t i = 0; i < model_metadata_.params.size(); ++i) {
        if (model_metadata_.params[i].name == param_name) {
            return params_[i];
        }
    }
    throw std::runtime_error("Parameter '" + param_name + "' not found in model parameters");
}

void ParameterManager::ClearNDArrayCache() {
  const PackedFunc* fclear_ndarray_cache = Registry::Get("vm.builtin.ndarray_cache.clear");
  ICHECK(fclear_ndarray_cache) << "Cannot find env function: vm.builtin.ndarray_cache.clear";
  (*fclear_ndarray_cache)();
}



bool ParameterManager::SetWeightsForFile(const std::string& filename,
                                            std::span<const uint8_t> bytes,
                                            bool is_finished_current) {


  auto& params_shard_file = params_shard_filemap_[filename];

  /* Since parameter shard records are quite large upto 80MiB,
   * we may have to send contents for the same file multiple times
   */
  params_shard_file.insert(params_shard_file.end(), bytes.begin(), bytes.end());

  if (is_finished_current) {
    UpdateNDArrayCache(filename);
    params_shard_filemap_.erase(filename);
  
    if (!(--params_shard_files_to_load_)) {
      LoadParams();
    }
  }

  return true;
}




