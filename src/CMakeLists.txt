# The Flutter tooling requires that developers have CMake 3.10 or later
# installed. You should not increase this version, as doing so will cause
# the plugin to fail to compile for some customers of the plugin.
cmake_minimum_required(VERSION 3.10)

project(maid_llm_library VERSION 0.0.1 LANGUAGES CPP)

add_library(maid_llm SHARED
  "maid_llm.cpp"
)

set_target_properties(maid_llm PROPERTIES
  PUBLIC_HEADER maid_llm.hpp
  OUTPUT_NAME "maid_llm"
)

target_compile_definitions(maid_llm PUBLIC DART_SHARED_LIB)
