#ifndef MLC_LLM_PARAMETER_MANAGER_H_
#define MLC_LLM_PARAMETER_MANAGER_H_

#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/relax_vm/ndarray_cache_support.h>
#include "cpp/metadata/model.h"
#include <span>

class ParameterManager {
public:
  explicit ParameterManager(tvm::runtime::Module local_vm, tvm::Device device); 
  bool SetWeightsForFile(const std::string& filename,
                                            std::span<const uint8_t> bytes,
                                            bool is_finished_current);
  tvm::runtime::Array<tvm::runtime::NDArray> params_;
  tvm::runtime::NDArray getParam(std::string param_name);
  
private:
  void UpdateNDArrayCache(const std::string& filename);
  void LoadParams();
  void ClearNDArrayCache();
  tvm::runtime::Module local_vm_;
  ::mlc::llm::ModelMetadata model_metadata_;
  tvm::runtime::relax_vm::NDArrayCacheMetadata ndarray_cache_metadata_;
  std::unordered_map<std::string, std::string> params_shard_filemap_;
  int params_shard_files_to_load_; 
  tvm::Device device_;
 
  
};

#endif  // MLC_LLM_PARAMETER_MANAGER_H_