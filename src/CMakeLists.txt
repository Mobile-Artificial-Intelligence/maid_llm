cmake_minimum_required(VERSION 3.10)

set(BUILD_SHARED_LIBS ON)

if(ANDROID)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DANDROID_ARM_NEON=TRUE -DANDROID_TOOLCHAIN=clang -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DANDROID_ARM_NEON=TRUE -DANDROID_TOOLCHAIN=clang -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23")
endif()

#set(LLAMA_STATIC ON CACHE BOOL "llama: enable static link libraries" FORCE)
set(LLAMA_NATIVE OFF CACHE BOOL "llama: disable -march=native flag" FORCE)

#if (NOT WIN32)
#  set(LLAMA_VULKAN ON CACHE BOOL "llama: enable vulkan" FORCE)
#endif()

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
)

target_compile_definitions(maid_llm PUBLIC DART_SHARED_LIB)
