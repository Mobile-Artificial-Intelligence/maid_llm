// ignore_for_file: constant_identifier_names
part of '../lcpp.dart';

class ContextParams {
  // text context, 0 = from model
  int? nCtx;

  // logical maximum batch size that can be submitted to llama_decode
  int? nBatch;

  // physical maximum batch size
  int? nUBatch;

  // max number of sequences (i.e. distinct states for recurrent models)
  int? nSeqMax;

  // number of threads to use for generation
  int? nThreads;

  // number of threads to use for batch processing
  int? nThreadsBatch;

  // RoPE scaling type, from `enum llama_rope_scaling_type`
  RopeScalingType? ropeScalingType;

  // whether to pool (sum) embedding results by sequence id
  PoolingType? poolingType;

  // attention type to use for embeddings
  AttentionType? attentionType;

  // RoPE base frequency, 0 = from model
  double? ropeFrequencyBase;

  // RoPE frequency scaling factor, 0 = from model
  double? ropeFrequencyScale;

  // YaRN extrapolation mix factor, negative = from model
  double? yarnExtrapolationFactor;

  // YaRN magnitude scaling factor
  double? yarnAttenuationFactor;

  // YaRN low correction dim
  double? yarnBetaFast;

  // YaRN high correction dim
  double? yarnBetaSlow;

  // YaRN original context size
  int? yarnOriginalContext;

  // defragment the KV cache if holes/size > thold, < 0 disabled (default)
  double? defragmentationThreshold;

  // data type for K cache
  GgmlType? typeK;

  // data type for V cache
  GgmlType? typeV;

  // if true, extract embeddings (together with logits)
  bool? embeddings;

  // whether to offload the KQV ops (including the KV cache) to GPU
  bool? offloadKqv;

  // whether to use flash attention
  bool? flashAttention;

  // whether to measure performance timings
  bool? noPerformance;

  ContextParams({
    this.nCtx,
    this.nBatch,
    this.nUBatch,
    this.nSeqMax,
    this.nThreads,
    this.nThreadsBatch,
    this.ropeScalingType,
    this.poolingType,
    this.attentionType,
    this.ropeFrequencyBase,
    this.ropeFrequencyScale,
    this.yarnExtrapolationFactor,
    this.yarnAttenuationFactor,
    this.yarnBetaFast,
    this.yarnBetaSlow,
    this.yarnOriginalContext,
    this.defragmentationThreshold,
    this.typeK,
    this.typeV,
    this.embeddings,
    this.offloadKqv,
    this.flashAttention,
    this.noPerformance,
  });

  llama_context_params toNative() {
    final llama_context_params contextParams = LlamaCPP.lib.llama_context_default_params();

    if (nCtx != null) {
      contextParams.n_ctx = nCtx!;
    }

    if (nBatch != null) {
      contextParams.n_batch = nBatch!;
    }

    if (nUBatch != null) {
      contextParams.n_ubatch = nUBatch!;
    }

    if (nSeqMax != null) {
      contextParams.n_seq_max = nSeqMax!;
    }

    if (nThreads != null) {
      contextParams.n_threads = nThreads!;
    }

    if (nThreadsBatch != null) {
      contextParams.n_threads_batch = nThreadsBatch!;
    }

    if (ropeScalingType != null) {
      contextParams.rope_scaling_type = ropeScalingType!.index;
    }

    if (poolingType != null) {
      contextParams.pooling_type = poolingType!.index;
    }

    if (attentionType != null) {
      contextParams.attention_type = attentionType!.index;
    }

    if (ropeFrequencyBase != null) {
      contextParams.rope_freq_base = ropeFrequencyBase!;
    }

    if (ropeFrequencyScale != null) {
      contextParams.rope_freq_scale = ropeFrequencyScale!;
    }

    if (yarnExtrapolationFactor != null) {
      contextParams.yarn_ext_factor = yarnExtrapolationFactor!;
    }

    if (yarnAttenuationFactor != null) {
      contextParams.yarn_attn_factor = yarnAttenuationFactor!;
    }

    if (yarnBetaFast != null) {
      contextParams.yarn_beta_fast = yarnBetaFast!;
    }

    if (yarnBetaSlow != null) {
      contextParams.yarn_beta_slow = yarnBetaSlow!;
    }

    if (yarnOriginalContext != null) {
      contextParams.yarn_orig_ctx = yarnOriginalContext!;
    }

    if (defragmentationThreshold != null) {
      contextParams.defrag_thold = defragmentationThreshold!;
    }

    if (typeK != null) {
      contextParams.type_k = typeK!.index;
    }

    if (typeV != null) {
      contextParams.type_v = typeV!.index;
    }

    if (embeddings != null) {
      contextParams.embeddings = embeddings!;
    }

    if (offloadKqv != null) {
      contextParams.offload_kqv = offloadKqv!;
    }

    if (flashAttention != null) {
      contextParams.flash_attn = flashAttention!;
    }

    if (noPerformance != null) {
      contextParams.no_perf = noPerformance!;
    }

    return contextParams;
  }
}

enum RopeScalingType {
  unspecified,
  none,
  linear,
  yarn,
  longrope;
}

enum PoolingType {
  unspecified,
  none,
  mean,
  cls,
  last,
  rank;
}

enum AttentionType {
  unspecified,
  causal,
  nonCausal;
}

enum GgmlType {
  f32,
  f16,
  q4_0,
  q4_1,
  q4_2,
  q4_3,
  q5_0,
  q5_1,
  q8_0,
  q8_1,
  q2_k,
  q3_k,
  q4_k,
  q5_k,
  q6_k,
  q8_k,
  iq2_xxs,
  iq2_xs,
  iq3_xxs,
  iq1_s,
  iq4_nl,
  iq3_s,
  iq2_s,
  iq4_xs,
  i8,
  i16,
  i32,
  i64,
  f64,
  iq1_m,
  bf16,
  q4_0_4_4,
  q4_0_4_8,
  q4_0_8_8,
  tq1_0,
  tq2_0;
}