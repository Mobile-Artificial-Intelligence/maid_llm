cmake_minimum_required(VERSION 3.10)

function(apply_optimizations target_name)
  message(STATUS "Applying optimizations to ${target_name}")

  target_compile_options(${target_name} PRIVATE -O3 -DNDEBUG)
  target_compile_options(${target_name} PRIVATE -ffunction-sections -fdata-sections)
  target_compile_options(${target_name} PRIVATE -march=armv8.4-a+fp16+dotprod)

  target_link_options(${target_name} PRIVATE -Wl,--gc-sections)
  target_link_options(${target_name} PRIVATE -flto)
endfunction()

set(BUILD_SHARED_LIBS ON)
set(CMAKE_INSTALL_LIBDIR lib CACHE PATH "library install dir" FORCE)

# Set the linker flags for shared libraries
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--build-id=none")

# Disable llama's -march=native flag
set(LLAMA_NATIVE OFF CACHE BOOL "llama: disable -march=native flag" FORCE)

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

# Enable compiler optimizations for arm64-v8a
if(${ANDROID_ABI} STREQUAL "arm64-v8a")
  apply_optimizations(maid_llm)
  apply_optimizations(llama)
  apply_optimizations(common)
endif()

target_link_libraries(maid_llm llama common)

set_target_properties(maid_llm PROPERTIES
  PUBLIC_HEADER maid_llm.h
  OUTPUT_NAME "maid"
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "$ORIGIN"
)

target_compile_definitions(maid_llm PUBLIC DART_SHARED_LIB)