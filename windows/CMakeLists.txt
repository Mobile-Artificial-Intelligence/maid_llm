# The Flutter tooling requires that developers have a version of Visual Studio
# installed that includes CMake 3.14 or later. You should not increase this
# version, as doing so will cause the plugin to fail to compile for some
# customers of the plugin.
cmake_minimum_required(VERSION 3.14)

# Project-level configuration.
set(PROJECT_NAME "lcpp")
project(${PROJECT_NAME} LANGUAGES CXX)

set(BUILD_BUNDLE_DIR "${CMAKE_BINARY_DIR}/bundle")
set(CMAKE_INSTALL_BINDIR ${BUILD_BUNDLE_DIR})

set(LCPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src)

set(BUILD_SHARED_LIBS ON)
set(CMAKE_INSTALL_LIBDIR lib CACHE PATH "library install dir" FORCE)

# Set the linker flags for shared libraries
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--build-id=none")

set(LLAMA_NATIVE OFF CACHE BOOL "llama: disable -march=native flag" FORCE)
set(LLAMA_VULKAN ON CACHE BOOL "llama: enable vulkan" FORCE)

add_subdirectory(${LCPP_DIR}/llama_cpp ${CMAKE_CURRENT_BINARY_DIR}/shared)

add_library(lcpp SHARED
  ${LCPP_DIR}/lcpp.cpp
)

# Add this line to include the llama_cpp directory
target_include_directories(lcpp PRIVATE
  ${LCPP_DIR}
  ${LCPP_DIR}/llama_cpp/include
  ${LCPP_DIR}/llama_cpp/common
  ${LCPP_DIR}/llama_cpp/ggml/include
)

target_link_libraries(lcpp llama common)

set_target_properties(lcpp PROPERTIES
  PUBLIC_HEADER ${LCPP_DIR}/lcpp.h
  OUTPUT_NAME "maid"
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "$ORIGIN"
)

target_compile_definitions(lcpp PUBLIC DART_SHARED_LIB)

set(lcpp_bundled_libraries
  # Defined in ../src/CMakeLists.txt.
  # This can be changed to accommodate different builds.
  $<TARGET_FILE:lcpp>
  PARENT_SCOPE
)