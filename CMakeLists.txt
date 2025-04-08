cmake_minimum_required(VERSION 3.25)
project(qvac_linear_classifier LANGUAGES CXX C)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build type and compiler flags
set(CMAKE_BUILD_TYPE Debug)
if(NOT MSVC)
  set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 ${CMAKE_CXX_FLAGS_DEBUG}")
  set(CMAKE_CXX_FLAGS "-fPIC -std=c++20 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_C_FLAGS_DEBUG "-g -O0 ${CMAKE_C_FLAGS_DEBUG}")
  set(CMAKE_C_FLAGS "-fPIC ${CMAKE_C_FLAGS}")
else()
  set(CMAKE_CXX_FLAGS_DEBUG "/Zi /Od /DEBUG ${CMAKE_CXX_FLAGS_DEBUG}")
  set(CMAKE_CXX_FLAGS "/std:c++20 ${CMAKE_CXX_FLAGS}")
endif()

# Find required packages
find_package(OpenCV REQUIRED)


# MLC-LLM configuration
set(MLC_ENABLE_SENTENCEPIECE_TOKENIZER ON CACHE BOOL "")
if(APPLE)
  set(USE_METAL ON CACHE BOOL "Use the Metal API for TVM runtime")
else()
  set(USE_VULKAN ON CACHE BOOL "Use the Vulkan API for TVM runtime")
endif()

# Add mlc-llm subdirectory
add_subdirectory(mlc-llm)

# Set paths
set(MLC_LLM_HOME ${PROJECT_SOURCE_DIR}/mlc-llm)
set(TVM_HOME ${MLC_LLM_HOME}/3rdparty/tvm)

# Set include directories
set(TVM_INCLUDES ${TVM_HOME}/include
                 ${TVM_HOME}/3rdparty/dlpack/include
                 ${TVM_HOME}/3rdparty/dmlc-core/include
                 ${TVM_HOME}/3rdparty/picojson)
set(TOKENIZERS_INCLUDES ${MLC_LLM_HOME}/3rdparty/tokenizers-cpp/include)

# Create the library
set(LIB_NAME qvac_linear_classifier)
set(LIB_SOURCES 
    include/linear_classifier.hpp
    include/param_manager.hpp
    src/linear_classifier.cpp
    src/param_manager.cpp
)



# Add finetunable model include directory
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${TVM_INCLUDES}
    ${OpenCV_INCLUDE_DIRS}
)

# Create library
add_library(${LIB_NAME} SHARED
    ${LIB_SOURCES}
   
)

# Create example executable
add_executable(linear_classifier_example example.cc)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
target_include_directories(${LIB_NAME} PUBLIC 
    ${PROJECT_SOURCE_DIR}/include
    ${MLC_LLM_HOME}
    ${TVM_INCLUDES}
    ${TOKENIZERS_INCLUDES}
    ${OpenCV_INCLUDE_DIRS}
   
)

# Set include directories for the example
target_include_directories(linear_classifier_example PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${MLC_LLM_HOME}
    ${TVM_INCLUDES}
    ${TOKENIZERS_INCLUDES}
    ${OpenCV_INCLUDE_DIRS}
   
)

# Link libraries
target_link_libraries(${LIB_NAME} PRIVATE 
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,mlc_llm_static,tvm_runtime,tokenizers_cpp>" 
    ${OpenCV_LIBS}
)

# Link example executable with our library
target_link_libraries(linear_classifier_example PRIVATE 
    ${LIB_NAME}
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,mlc_llm_static,tvm_runtime,tokenizers_cpp>"
    ${OpenCV_LIBS}
)

# Install rules
install(TARGETS ${LIB_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# Enable testing for future use
enable_testing()