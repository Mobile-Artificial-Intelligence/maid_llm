# The Flutter tooling requires that developers have a version of Visual Studio
# installed that includes CMake 3.14 or later. You should not increase this
# version, as doing so will cause the plugin to fail to compile for some
# customers of the plugin.
cmake_minimum_required(VERSION 3.14)

# Project-level configuration.
set(PROJECT_NAME "maid_llm")
project(${PROJECT_NAME} LANGUAGES CXX)

set(BUILD_BUNDLE_DIR "${CMAKE_BINARY_DIR}/bundle")
set(CMAKE_INSTALL_BINDIR ${BUILD_BUNDLE_DIR})

set(MAID_LLM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src)

set(BUILD_SHARED_LIBS ON)
set(CMAKE_INSTALL_LIBDIR lib CACHE PATH "library install dir" FORCE)

# Set the linker flags for shared libraries
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--build-id=none")

set(LLAMA_NATIVE OFF CACHE BOOL "llama: disable -march=native flag" FORCE)
set(LLAMA_VULKAN ON CACHE BOOL "llama: enable vulkan" FORCE)

add_subdirectory(${MAID_LLM_DIR}/llama_cpp ${CMAKE_CURRENT_BINARY_DIR}/shared)

add_library(maid_llm SHARED
  ${MAID_LLM_DIR}/maid_llm.cpp
)

# Add this line to include the llama_cpp directory
target_include_directories(maid_llm PRIVATE
  ${MAID_LLM_DIR}
  ${MAID_LLM_DIR}/llama_cpp
  ${MAID_LLM_DIR}/llama_cpp/common
)

target_link_libraries(maid_llm llama common)

set_target_properties(maid_llm PROPERTIES
  PUBLIC_HEADER ${MAID_LLM_DIR}/maid_llm.h
  OUTPUT_NAME "maid"
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "$ORIGIN"
)

target_compile_definitions(maid_llm PUBLIC DART_SHARED_LIB)

set(maid_llm_bundled_libraries
  # Defined in ../src/CMakeLists.txt.
  # This can be changed to accommodate different builds.
  $<TARGET_FILE:maid_llm>
  PARENT_SCOPE
)