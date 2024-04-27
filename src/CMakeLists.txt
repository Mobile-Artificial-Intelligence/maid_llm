cmake_minimum_required(VERSION 3.10)

set(BUILD_SHARED_LIBS ON)

# Set the linker flags for shared libraries
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--build-id=none")

set(LLAMA_NATIVE OFF CACHE BOOL "llama: disable -march=native flag" FORCE)
set(LLAMA_VULKAN ON CACHE BOOL "llama: enable vulkan" FORCE)

add_subdirectory(./llama_cpp)

add_library(maid_llm SHARED
  maid_llm.cpp
)

# Set output directories for libraries and executables to ensure they go to 'lib'
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add this line to include the llama_cpp directory
target_include_directories(maid_llm PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/common
)

target_link_libraries(maid_llm llama common)

set_target_properties(maid_llm PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  PUBLIC_HEADER maid_llm.h
  OUTPUT_NAME "maid"
  BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_definitions(maid_llm PUBLIC DART_SHARED_LIB)
