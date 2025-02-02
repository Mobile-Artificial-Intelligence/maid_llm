#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint lcpp.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'lcpp'
  s.version          = '0.0.1'
  s.summary          = 'A Flutter FFI plugin for interfacing with llama_cpp.'
  s.description      = <<-DESC
A new Flutter FFI plugin project.
                       DESC
  s.homepage         = 'https://github.com/Mobile-Artificial-Intelligence/dart_lcpp'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Dane Madsen' => 'dane_madsen@hotmail.com' }
  s.dependency 'FlutterMacOS'
  s.swift_version = '5.0'

  # This will ensure the source files in Classes/ are included in the native
  # builds of apps using this FFI plugin. Podspec does not support relative
  # paths, so Classes contains a forwarder C file that relatively imports
  # `../src/*` so that the C sources can be shared among all target platforms.
  s.source           = { :path => '.' }
  s.source_files = 'build-info.c',
                   'llama_cpp/src/*.cpp',
                   'llama_cpp/common/*.cpp',
                   'llama_cpp/ggml/src/*.cpp',
                   'llama_cpp/ggml/src/ggml-cpu/*.cpp',
                   'llama_cpp/ggml/src/ggml-cpu/*.c',
                   'llama_cpp/ggml/src/ggml-metal/*.cpp',
                   'llama_cpp/ggml/src/ggml-metal/*.m',
                   'llama_cpp/src/*.c',
                   'llama_cpp/src/llama.cpp',
                   'llama_cpp/src/llama-sampling.cpp',
                   'llama_cpp/src/llama-grammar.cpp',
                   'llama_cpp/ggml/src/ggml.c',
                   'llama_cpp/ggml/src/ggml-alloc.c',
                   'llama_cpp/ggml/src/ggml-backend.c',
                   'llama_cpp/ggml/src/ggml-metal.m',
                   'llama_cpp/ggml/src/ggml-quants.c',
                   'llama_cpp/ggml/src/ggml-aarch64.c',
                   'llama_cpp/src/llama-vocab.cpp',
                   'llama_cpp/src/unicode.cpp',
                   'llama_cpp/src/unicode-data.cpp',
                   'llama_cpp/common/common.cpp',
                   'llama_cpp/common/build-info.cpp',
                   'llama_cpp/common/grammar-parser.cpp',
                   'llama_cpp/common/json-schema-to-grammar.cpp',
                   'llama_cpp/common/sampling.cpp',
                   'llama_cpp/common/stb_image.h',
  s.frameworks = 'Foundation', 'Metal', 'MetalKit'
  s.platform = :osx, '10.15'
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'USER_HEADER_SEARCH_PATHS' => [
      '$(PODS_TARGET_SRCROOT)/llama_cpp/ggml/include/*.h',
      '$(PODS_TARGET_SRCROOT)/llama_cpp/ggml/include',
      '$(PODS_TARGET_SRCROOT)/llama_cpp/include',
      '$(PODS_TARGET_SRCROOT)/llama_cpp/src',
      '$(PODS_TARGET_SRCROOT)/llama_cpp/**/*.h', 
      '$(PODS_TARGET_SRCROOT)/llama_cpp/common/**/*.h',
      '$(PODS_TARGET_SRCROOT)/llama_cpp/ggml/include',
      '$(PODS_TARGET_SRCROOT)/llama_cpp/ggml/src',
      '$(PODS_TARGET_SRCROOT)/llama_cpp/ggml/src/ggml-cpu',
      '$(PODS_TARGET_SRCROOT)/llama_cpp/include',
      '$(PODS_TARGET_SRCROOT)/llama_cpp/src',
      '$(PODS_TARGET_SRCROOT)/llama_cpp/ggml/include',
      '$(PODS_TARGET_SRCROOT)/llama_cpp/ggml/src',
      '$(PODS_TARGET_SRCROOT)/llama_cpp/common',
      '$(PODS_TARGET_SRCROOT)/llama_cpp/common'
    ],
    'HEADER_SEARCH_PATHS' => [
      '$(PODS_TARGET_SRCROOT)/llama_cpp/ggml/include',
      '$(PODS_TARGET_SRCROOT)/llama_cpp/ggml/include/*.h',
      '$(PODS_TARGET_SRCROOT)/llama_cpp/include',
      '$(PODS_TARGET_SRCROOT)/llama_cpp/src',
      '$(PODS_TARGET_SRCROOT)/llama_cpp/**/*.h', 
      '$(PODS_TARGET_SRCROOT)/llama_cpp/common/**/*.h',
      '$(PODS_TARGET_SRCROOT)/llama_cpp/common',
      '$(PODS_TARGET_SRCROOT)/llama_cpp/common'
    ],
    'OTHER_CFLAGS' => ['$(inherited)', '-O3', '-flto', '-fno-objc-arc', '-w', '-I$(PODS_TARGET_SRCROOT)/llama_cpp/include', '-I$(PODS_TARGET_SRCROOT)/llama_cpp/ggml/include', '-I$(PODS_TARGET_SRCROOT)/llama_cpp/common', '-DGGML_LLAMAFILE=OFF', '-DGGML_USE_CPU'],
    'OTHER_CPLUSPLUSFLAGS' => ['$(inherited)', '-O3', '-flto', '-fno-objc-arc', '-w', '-std=c++17', '-fpermissive', '-I$(PODS_TARGET_SRCROOT)/llama_cpp/include', '-I$(PODS_TARGET_SRCROOT)/llama_cpp/ggml/include', '-I$(PODS_TARGET_SRCROOT)/llama_cpp/common', '-DGGML_LLAMAFILE=OFF', '-DGGML_USE_CPU'],
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
cd "${PODS_TARGET_SRCROOT}/llama_cpp"
xcrun metal -target "air64-${LLVM_TARGET_TRIPLE_VENDOR}-${LLVM_TARGET_TRIPLE_OS_VERSION}${LLVM_TARGET_TRIPLE_SUFFIX:-\"\"}" -ffast-math -std=ios-metal2.3 -o "${METAL_LIBRARY_OUTPUT_DIR}/default.metallib" ggml/src/ggml-metal/*.metal
SCRIPT
    }
  ]
end