import 'dart:ffi';
import 'dart:math';

import 'bindings.dart';

import 'package:ffi/ffi.dart';

import 'sampling_params.dart';

class GptParams {
  /// Seed for random number generation. Defaults to a random integer.
  /// A value of -1 indicates a random seed should be used.
  int seed = Random().nextInt(1000000);

  /// Number of threads to use for generation. Defaults to 8.
  int nThreads = 8;

  /// Possibly unused. Defaults to -1.
  int nThreadsDraft = -1;

  /// Number of threads to use for batch processing. Defaults to 8.
  int nThreadsBatch = 8;

  /// Possibly unused. Defaults to -1.
  int nThreadsBatchDraft = -1;

  /// Number of predictions to generate. Defaults to -1.
  int nPredict = -1;

  /// Number of tokens in the context. Defaults to 512.
  int nCtx = 512;

  /// Number of tokens in the batch. Defaults to 512.
  int nBatch = 512;

  /// Number of tokens to keep from initial prompt. Defaults to 0.
  int nKeep = 0;

  /// Number of tokens to keep from draft. Defaults to 8.
  int nDraft = 8;

  /// Maximum number of chunks to process. Defaults to -1.
  int nChunks = -1;

  /// Number of parallel sequences to decode. Defaults to 1.
  int nParallel = 1;

  /// Number of sequences to decode. Defaults to 1.
  int nSequences = 1;

  /// Speculative decoding split probability. Defaults to 0.1.
  double pSplit = 0.1;

  /// Number of layers to store in VRAM. Defaults to -1.
  int nGpuLayers = -1;

  /// Number of layers to store in VRAM for the draft model. Defaults to -1.
  int nGpuLayersDraft = -1;

  /// How to split the model across GPUs. Defaults to SplitMode.layer.
  SplitMode splitMode = SplitMode.layer;

  /// The GPU that is used for scratch and small tensors. Defaults to 0.
  int mainGpu = 0;

  /// If non-zero then use beam search of given width. Defaults to 0.
  int nBeams = 0;

  /// Group-attention factor. Defaults to 1.
  int grpAttnN = 1;

  /// Group-attention width. Defaults to 512.
  int grpAttnW = 512;

  /// Print token count every n tokens. Defaults to -1.
  int nPrint = -1;

  /// RoPE base frequency. Defaults to 0.0.
  double ropeFreqBase = 0.0;

  /// RoPE frequency scaling factor. Defaults to 0.0.
  double ropeFreqScale = 0.0;

  /// YaRN extrapolation mix factor. Defaults to -1.0.
  double yarnExtFactor = -1.0;

  /// YaRN magnitude scaling factor. Defaults to 1.0.
  double yarnAttnFactor = 1.0;

  /// YaRN low correction dim. Defaults to 32.0.
  double yarnBetaFast = 32.0;

  /// YaRN high correction dim. Defaults to 1.0.
  double yarnBetaSlow = 1.0;

  /// YaRN original context length. Defaults to 0.
  int yarnOrigCtx = 0;

  /// KV cache defragmentation threshold. Defaults to -1.0.
  double defragThold = -1.0;

  /// Rope scaling type. Defaults to -1.
  int ropeScalingType = -1;

  /// NUMA strategy. Defaults to GgmlNumaStrategy.disabled.
  GgmlNumaStrategy numa = GgmlNumaStrategy.disabled;

  /// Sampling parameters.
  SamplingParams sparams = SamplingParams();

  /// Model path.
  String model = '';

  /// Draft model for speculative decoding.
  String modelDraft = '';

  /// Model alias.
  String modelAlias = '';

  /// Prompt.
  String prompt = '';

  /// Store the external prompt file name.
  String promptFile = '';

  /// Path to file for saving/loading prompt eval state.
  String pathPromptCache = '';

  /// String to prefix user inputs with.
  String inputPrefix = '';

  /// String to suffix user inputs with.
  String inputSuffix = '';

  /// String upon seeing which more user input is prompted.
  String antiprompt = '';

  /// Directory in which to save YAML log files.
  String logdir = '';

  /// File for saving *all* logits.
  String logitsFile = '';

  /// Base model path for the lora adapter.
  String loraBase = '';

  /// Stride for perplexity calculations. If left at 0, the pre-existing approach will be used.
  int pplStride = 0;

  /// = 0 -> ppl output is as usual, = 1 -> ppl output is num_tokens, ppl, one per line
  int pplOutputType = 0;

  /// Compute HellaSwag score over random tasks from datafile supplied in prompt.
  bool hellaswag = false;

  /// Number of tasks to use when computing the HellaSwag score.
  int hellaswagTasks = 400;

  /// Compute Winogrande score over random tasks from datafile supplied in prompt.
  bool winogrande = false;

  /// Number of tasks to use when computing the Winogrande score. If 0, all tasks will be computed.
  int winograndeTasks = 0;

  /// Compute TruthfulQA score over random tasks from datafile supplied in prompt.
  bool multipleChoice = false;

  /// Number of tasks to use when computing the TruthfulQA score. If 0, all tasks will be computed.
  int multipleChoiceTasks = 0;

  /// Compute KL-divergence.
  bool klDivergence = false;

  /// Do not randomize prompt if none provided.
  bool randomPrompt = false;

  /// Use color to distinguish generations and inputs.
  bool useColor = false;

  /// Interactive mode.
  bool interactive = false;

  /// Chatml mode (used for models trained on chatml syntax).
  bool chatml = false;

  /// Save user input and generations to prompt cache.
  bool promptCacheAll = false;

  /// Open the prompt cache read-only and do not update it.
  bool promptCacheRo = false;

  /// Get only sentence embedding.
  bool embedding = false;

  /// Escape "\n", "\r", "\t", "\'", "\"", and "\\".
  bool escape = false;

  /// Wait for user input immediately.
  bool interactiveFirst = false;

  /// Reverse the usage of `\`.
  bool multilineInput = false;

  /// Improves compatibility with subprocesses and limited consoles.
  bool simpleIo = false;

  /// Insert new sequences for decoding on-the-fly.
  bool contBatching = false;

  /// Prefix BOS to user inputs, preceding input_prefix.
  bool inputPrefixBos = false;

  /// Ignore generated EOS tokens.
  bool ignoreEos = false;

  /// Instruction mode (used for Alpaca models).
  bool instruct = false;

  /// Return logits for all tokens in the batch.
  bool logitsAll = false;

  /// Use mmap for faster loads.
  bool useMmap = true;

  /// Use mlock to keep model in memory.
  bool useMlock = false;

  /// Print prompt tokens before generation.
  bool verbosePrompt = false;

  /// Print prompt before generation.
  bool displayPrompt = true;

  /// Use infill mode.
  bool infill = false;

  /// Dump the KV cache contents for debugging purposes.
  bool dumpKvCache = false;

  /// Disable KV offloading.
  bool noKvOffload = false;

  /// KV cache data type for the K.
  String cacheTypeK = 'f16';

  /// KV cache data type for the V.
  String cacheTypeV = 'f16';

  /// Path to multimodal projector.
  String mmproj = '';

  /// Path to an image file.
  String image = '';

  Pointer<gpt_c_params> toNative() {
    final gpt = calloc<gpt_c_params>();
    gpt.ref.seed = seed;
    gpt.ref.n_threads = nThreads;
    gpt.ref.n_threads_draft = nThreadsDraft;
    gpt.ref.n_threads_batch = nThreadsBatch;
    gpt.ref.n_threads_batch_draft = nThreadsBatchDraft;
    gpt.ref.n_predict = nPredict;
    gpt.ref.n_ctx = nCtx;
    gpt.ref.n_batch = nBatch;
    gpt.ref.n_keep = nKeep;
    gpt.ref.n_draft = nDraft;
    gpt.ref.n_chunks = nChunks;
    gpt.ref.n_parallel = nParallel;
    gpt.ref.n_sequences = nSequences;
    gpt.ref.p_split = pSplit;
    gpt.ref.n_gpu_layers = nGpuLayers;
    gpt.ref.n_gpu_layers_draft = nGpuLayersDraft;
    gpt.ref.split_mode = splitMode.index;
    gpt.ref.main_gpu = mainGpu;
    gpt.ref.n_beams = nBeams;
    gpt.ref.grp_attn_n = grpAttnN;
    gpt.ref.grp_attn_w = grpAttnW;
    gpt.ref.n_print = nPrint;
    gpt.ref.rope_freq_base = ropeFreqBase;
    gpt.ref.rope_freq_scale = ropeFreqScale;
    gpt.ref.yarn_ext_factor = yarnExtFactor;
    gpt.ref.yarn_attn_factor = yarnAttnFactor;
    gpt.ref.yarn_beta_fast = yarnBetaFast;
    gpt.ref.yarn_beta_slow = yarnBetaSlow;
    gpt.ref.yarn_orig_ctx = yarnOrigCtx;
    gpt.ref.defrag_thold = defragThold;
    gpt.ref.rope_scaling_type = ropeScalingType;
    gpt.ref.numa = numa.index;
    gpt.ref.sparams = sparams.toNative();
    gpt.ref.model = model.toNativeUtf8().cast<Char>();
    gpt.ref.model_draft = modelDraft.toNativeUtf8().cast<Char>();
    gpt.ref.model_alias = modelAlias.toNativeUtf8().cast<Char>();
    gpt.ref.prompt = prompt.toNativeUtf8().cast<Char>();
    gpt.ref.prompt_file = promptFile.toNativeUtf8().cast<Char>();
    gpt.ref.path_prompt_cache = pathPromptCache.toNativeUtf8().cast<Char>();
    gpt.ref.input_prefix = inputPrefix.toNativeUtf8().cast<Char>();
    gpt.ref.input_suffix = inputSuffix.toNativeUtf8().cast<Char>();
    gpt.ref.antiprompt = antiprompt.toNativeUtf8().cast<Char>();
    gpt.ref.logdir = logdir.toNativeUtf8().cast<Char>();
    gpt.ref.logits_file = logitsFile.toNativeUtf8().cast<Char>();
    gpt.ref.lora_base = loraBase.toNativeUtf8().cast<Char>();
    gpt.ref.ppl_stride = pplStride;
    gpt.ref.ppl_output_type = pplOutputType;
    gpt.ref.hellaswag = hellaswag;
    gpt.ref.hellaswag_tasks = hellaswagTasks;
    gpt.ref.winogrande = winogrande;
    gpt.ref.winogrande_tasks = winograndeTasks;
    gpt.ref.multiple_choice = multipleChoice;
    gpt.ref.multiple_choice_tasks = multipleChoiceTasks;
    gpt.ref.kl_divergence = klDivergence;
    gpt.ref.random_prompt = randomPrompt;
    gpt.ref.use_color = useColor;
    gpt.ref.interactive = interactive;
    gpt.ref.chatml = chatml;
    gpt.ref.prompt_cache_all = promptCacheAll;
    gpt.ref.prompt_cache_ro = promptCacheRo;
    gpt.ref.embedding = embedding;
    gpt.ref.escape = escape;
    gpt.ref.interactive_first = interactiveFirst;
    gpt.ref.multiline_input = multilineInput;
    gpt.ref.simple_io = simpleIo;
    gpt.ref.cont_batching = contBatching;
    gpt.ref.input_prefix_bos = inputPrefixBos;
    gpt.ref.ignore_eos = ignoreEos;
    gpt.ref.instruct = instruct;
    gpt.ref.logits_all = logitsAll;
    gpt.ref.use_mmap = useMmap;
    gpt.ref.use_mlock = useMlock;
    gpt.ref.verbose_prompt = verbosePrompt;
    gpt.ref.display_prompt = displayPrompt;
    gpt.ref.infill = infill;
    gpt.ref.dump_kv_cache = dumpKvCache;
    gpt.ref.no_kv_offload = noKvOffload;
    gpt.ref.cache_type_k = cacheTypeK.toNativeUtf8().cast<Char>();
    gpt.ref.cache_type_v = cacheTypeV.toNativeUtf8().cast<Char>();
    gpt.ref.mmproj = mmproj.toNativeUtf8().cast<Char>();
    gpt.ref.image = image.toNativeUtf8().cast<Char>();
    return gpt;
  }
}

enum SplitMode { none, layer, row }

enum GgmlNumaStrategy { disabled, distribute, isolate, numactl, mirror, count }
