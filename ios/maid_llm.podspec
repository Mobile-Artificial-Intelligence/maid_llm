#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint maid_llm.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'maid_llm'
  s.version          = '0.0.1'
  s.summary          = 'A Flutter FFI plugin for interfacing with llama_cpp.'
  s.description      = <<-DESC
A new Flutter FFI plugin project.
                       DESC
  s.homepage         = 'https://github.com/Mobile-Artificial-Intelligence/maid_llm'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Dane Madsen' => 'dane_madsen@hotmail.com' }
  s.dependency 'Flutter'
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' }
  s.swift_version = '5.0'

  # This will ensure the source files in Classes/ are included in the native
  # builds of apps using this FFI plugin. Podspec does not support relative
  # paths, so Classes contains a forwarder C file that relatively imports
  # `../src/*` so that the C sources can be shared among all target platforms.
  s.source           = { :path => '.' }
  s.source_files = 'Classes/**/*', 
    'llama_cpp/src/llama.cpp',
    'llama_cpp/ggml/src/ggml.c',
    'llama_cpp/ggml/src/ggml-alloc.c',
    'llama_cpp/ggml/src/ggml-backend.c',
    'llama_cpp/ggml/src/ggml-metal.m',
    'llama_cpp/ggml/src/ggml-quants.c',
    'llama_cpp/src/unicode.cpp',
    'llama_cpp/src/unicode-data.cpp',
    'llama_cpp/common/common.cpp',
    'llama_cpp/common/build-info.cpp',
    'llama_cpp/common/grammar-parser.cpp',
    'llama_cpp/common/json-schema-to-grammar.cpp',
    'llama_cpp/common/sampling.cpp',
    'llama_cpp/common/stb_image.h',
  s.frameworks = 'Foundation', 'Metal', 'MetalKit'
  s.platform = :ios, '12.0'
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'USER_HEADER_SEARCH_PATHS' => [
      '$(PODS_TARGET_SRCROOT)/llama_cpp', 
      '$(PODS_TARGET_SRCROOT)/llama_cpp/common', 
      '$(PODS_TARGET_SRCROOT)/llama_cpp/ggml/include', 
      '$(PODS_TARGET_SRCROOT)/llama_cpp/include'
    ],
    'HEADER_SEARCH_PATHS' => [
      '$(PODS_TARGET_SRCROOT)/llama_cpp', 
      '$(PODS_TARGET_SRCROOT)/llama_cpp/common', 
      '$(PODS_TARGET_SRCROOT)/llama_cpp/ggml/include', 
      '$(PODS_TARGET_SRCROOT)/llama_cpp/include'
    ],
    'OTHER_CFLAGS' => ['$(inherited)', '-O3', '-flto', '-fno-objc-arc'],
    'OTHER_CPLUSPLUSFLAGS' => ['$(inherited)', '-O3', '-flto', '-fno-objc-arc'],
    'GCC_PREPROCESSOR_DEFINITIONS' => ['$(inherited)', 'GGML_USE_METAL=1'],
  }
  s.script_phases = [
    {
      :name => 'Build Metal Library',
      :input_files => ["${PODS_TARGET_SRCROOT}/llama_cpp/ggml/src/ggml-metal.metal"],
      :output_files => ["${METAL_LIBRARY_OUTPUT_DIR}/default.metallib"],
      :execution_position => :after_compile,
      :script => <<-SCRIPT
      set -e
      set -u
      set -o pipefail
      cd "${PODS_TARGET_SRCROOT}/llama_cpp/ggml/src"
      xcrun metal -target "air64-${LLVM_TARGET_TRIPLE_VENDOR}-${LLVM_TARGET_TRIPLE_OS_VERSION}${LLVM_TARGET_TRIPLE_SUFFIX:-\"\"}" -ffast-math -std=ios-metal2.3 -o "${METAL_LIBRARY_OUTPUT_DIR}/default.metallib" *.metal
      SCRIPT
    }
  ]
end
