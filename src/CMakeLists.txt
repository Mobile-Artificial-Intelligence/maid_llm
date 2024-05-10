cmake_minimum_required(VERSION 3.10)

set(LLAMA_CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp)

set(BUILD_SHARED_LIBS ON)
set(CMAKE_INSTALL_LIBDIR lib CACHE PATH "library install dir" FORCE)

find_package(Vulkan)
if (Vulkan_FOUND)
  add_compile_definitions(GGML_USE_VULKAN)
  set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} Vulkan::Vulkan)
  set(GGML_HEADERS_VULKAN ggml-vulkan.h)
  set(GGML_SOURCES_VULKAN ggml-vulkan.cpp)
endif()

add_library(maid_llm SHARED
  maid_llm.cpp
  maid_llm.h
  utils.hpp
  ${LLAMA_CPP_DIR}/ggml.c
  ${LLAMA_CPP_DIR}/ggml.h
  ${LLAMA_CPP_DIR}/ggml-alloc.c
  ${LLAMA_CPP_DIR}/ggml-alloc.h
  ${LLAMA_CPP_DIR}/ggml-backend.c
  ${LLAMA_CPP_DIR}/ggml-backend.h
  ${LLAMA_CPP_DIR}/ggml-quants.c
  ${LLAMA_CPP_DIR}/ggml-quants.h
  ${LLAMA_CPP_DIR}/llama.cpp
  ${LLAMA_CPP_DIR}/llama.h
  ${LLAMA_CPP_DIR}/sgemm.cpp
  ${LLAMA_CPP_DIR}/sgemm.h
  ${LLAMA_CPP_DIR}/unicode.h
  ${LLAMA_CPP_DIR}/unicode.cpp
  ${LLAMA_CPP_DIR}/unicode-data.cpp
  ${LLAMA_CPP_DIR}/common/common.cpp
  ${LLAMA_CPP_DIR}/common/common.h
  ${LLAMA_CPP_DIR}/common/sampling.cpp
  ${LLAMA_CPP_DIR}/common/sampling.h
  ${LLAMA_CPP_DIR}/common/grammar-parser.cpp
  ${LLAMA_CPP_DIR}/common/grammar-parser.h
  ${LLAMA_CPP_DIR}/common/json.hpp
  ${LLAMA_CPP_DIR}/common/json-schema-to-grammar.cpp
  ${GGML_SOURCES_VULKAN}
  ${GGML_HEADERS_VULKAN}
)

# Add this line to include the llama_cpp directory
target_include_directories(maid_llm PRIVATE 
  ${LLAMA_CPP_DIR}
  ${LLAMA_CPP_DIR}/common
)

set_target_properties(maid_llm PROPERTIES
  PUBLIC_HEADER maid_llm.h
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
if(${ANDROID_ABI} STREQUAL "arm64-v8a")
  target_compile_options(maid_llm PRIVATE -march=armv8.4-a+fp16+dotprod)
endif()