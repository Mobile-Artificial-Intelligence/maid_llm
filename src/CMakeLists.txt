cmake_minimum_required(VERSION 3.10)

set(BUILD_SHARED_LIBS ON)
set(CMAKE_INSTALL_LIBDIR lib CACHE PATH "library install dir" FORCE)

# Set the linker flags for shared libraries
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--build-id=none")

# Set LLAMA_NATIVE to on only for android abi arm64-v8a
if(ANDROID_ABI STREQUAL "arm64-v8a")
  set(LLAMA_NATIVE ON CACHE BOOL "llama: enable -march=native flag" FORCE)
else()
  set(LLAMA_NATIVE OFF CACHE BOOL "llama: disable -march=native flag" FORCE)
endif()

#  Enable vulkan for llama
set(LLAMA_VULKAN ON CACHE BOOL "llama: enable vulkan" FORCE)

add_subdirectory(./llama_cpp)

add_library(maid_llm SHARED
  maid_llm.cpp
)

# Add this line to include the llama_cpp directory
target_include_directories(maid_llm PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/common
)

target_link_libraries(maid_llm llama common)

set_target_properties(maid_llm PROPERTIES
  PUBLIC_HEADER maid_llm.h
  OUTPUT_NAME "maid"
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "$ORIGIN"
)

target_compile_definitions(maid_llm PUBLIC DART_SHARED_LIB)
