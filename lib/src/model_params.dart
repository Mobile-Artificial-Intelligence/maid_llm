part of '../lcpp.dart';

typedef OnProgressCallback = bool Function(double progress);

class ModelParams {
  bool? vocabOnly;
  bool? useMmap;
  bool? useMlock;
  bool? checkTensors;
  OnProgressCallback? onProgress;

  ModelParams({
    this.vocabOnly,
    this.useMmap,
    this.useMlock,
    this.checkTensors,
    this.onProgress,
  });

  llama_model_params toNative() {
    final llama_model_params modelParams = LlamaCPP.lib.llama_model_default_params();

    if (vocabOnly != null) {
      modelParams.vocab_only = vocabOnly!;
    }

    if (useMmap != null) {
      modelParams.use_mmap = useMmap!;
    }

    if (useMlock != null) {
      modelParams.use_mlock = useMlock!;
    }

    if (checkTensors != null) {
      modelParams.check_tensors = checkTensors!;
    }

    if (onProgress != null) {
      modelParams.progress_callback = ffi.Pointer.fromFunction(_onProgress, false);
    }

    return modelParams;
  }

  bool _onProgress(double progress, ffi.Pointer<ffi.Void> userData) {
    return onProgress!(progress);
  }
}