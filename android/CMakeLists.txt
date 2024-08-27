cmake_minimum_required(VERSION 3.10)

set(MAID_LLM_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src)
set(LLAMA_CPP_DIR ${MAID_LLM_SRC_DIR}/llama_cpp)

set(BUILD_SHARED_LIBS ON)
set(CMAKE_INSTALL_LIBDIR lib CACHE PATH "library install dir" FORCE)

find_package(Vulkan)
if (Vulkan_FOUND)
  add_compile_definitions(GGML_USE_VULKAN)
  set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} Vulkan::Vulkan)
  set(GGML_HEADERS_VULKAN ${LLAMA_CPP_DIR}/ggml/src/ggml-vulkan.h)
  set(GGML_SOURCES_VULKAN ${LLAMA_CPP_DIR}/ggml/src/ggml-vulkan.cpp)
else()
  message(WARNING "Vulkan not found, building without Vulkan support")
endif()

add_library(maid_llm SHARED
  ${MAID_LLM_SRC_DIR}/maid_llm.cpp
  ${MAID_LLM_SRC_DIR}/build-info.c
  ${LLAMA_CPP_DIR}/ggml/src/ggml.c
  ${LLAMA_CPP_DIR}/ggml/src/ggml-aarch64.c
  ${LLAMA_CPP_DIR}/ggml/src/ggml-alloc.c
  ${LLAMA_CPP_DIR}/ggml/src/ggml-backend.c
  ${LLAMA_CPP_DIR}/ggml/src/ggml-quants.c
  ${LLAMA_CPP_DIR}/src/llama.cpp
  ${LLAMA_CPP_DIR}/src/llama-grammar.cpp
  ${LLAMA_CPP_DIR}/src/llama-sampling.cpp
  ${LLAMA_CPP_DIR}/src/llama-vocab.cpp
  ${LLAMA_CPP_DIR}/src/unicode.cpp
  ${LLAMA_CPP_DIR}/src/unicode-data.cpp
  ${LLAMA_CPP_DIR}/common/common.cpp
  ${LLAMA_CPP_DIR}/common/common.h
  ${LLAMA_CPP_DIR}/common/sampling.cpp
  ${LLAMA_CPP_DIR}/common/sampling.h
  ${LLAMA_CPP_DIR}/common/grammar-parser.cpp
  ${LLAMA_CPP_DIR}/common/json-schema-to-grammar.cpp
  ${GGML_SOURCES_VULKAN}
  ${GGML_HEADERS_VULKAN}
)

# Add this line to include the llama_cpp directory
target_include_directories(maid_llm PRIVATE
  ${MAID_LLM_SRC_DIR}
  ${LLAMA_CPP_DIR}/ggml/include
  ${LLAMA_CPP_DIR}/include
  ${LLAMA_CPP_DIR}/common
)

set_target_properties(maid_llm PROPERTIES
  PUBLIC_HEADER ${MAID_LLM_SRC_DIR}/maid_llm.h
  OUTPUT_NAME "maid"
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "$ORIGIN"
)

target_compile_definitions(maid_llm PUBLIC DART_SHARED_LIB)

target_compile_options(maid_llm PRIVATE 
  -O3 
  -DNDEBUG
  -ffunction-sections 
  -fdata-sections
)

target_link_options(maid_llm PRIVATE 
  -Wl,--gc-sections 
  -flto
)

# Enable compiler optimizations for arm64-v8a
if(ANDROID_ABI STREQUAL "arm64-v8a")
  target_compile_options(maid_llm PRIVATE -march=armv8.4-a+fp16+dotprod)
endif()