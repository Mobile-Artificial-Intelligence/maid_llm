part of '../lcpp.dart';

typedef OnProgressCallback = bool Function(double progress);

class ModelParams {
  bool? vocabOnly;
  bool? useMmap;
  bool? useMlock;
  bool? checkTensors;

  ModelParams({
    this.vocabOnly,
    this.useMmap,
    this.useMlock,
    this.checkTensors,
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

    return modelParams;
  }
}