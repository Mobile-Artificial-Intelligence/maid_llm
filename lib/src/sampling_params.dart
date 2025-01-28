part of '../maid_llm.dart';

class SamplingParams {
  // number of previous tokens to remember
  int nPrev = 64;

  // if greater than 0, output the probabilities of top n_probs tokens.
  int nProbs = 0;

  // 0 = disabled, otherwise samplers should return at least min_keep tokens
  int minKeep = 0;

  // <= 0 to use vocab size
  int topK = 40;

  // 1.0 = disabled
  double topP = 0.95;

  // 0.0 = disabled
  double minP = 0.05;

  // 1.0 = disabled
  double tfsZ = 1.0;

  // 1.0 = disabled
  double typicalP = 1.0;

  // <= 0.0 to sample greedily, 0.0 to not output probabilities
  double temp = 0.80;

  // 0.0 = disabled
  double dynatempRange = 0.0;

  // controls how entropy maps to temperature in dynamic temperature sampler
  double dynatempExponent = 1.0;

  // last n tokens to penalize (0 = disable penalty, -1 = context size)
  int penaltyLastN = 64;

  // 1.0 = disabled
  double penaltyRepeat = 1.10;

  // 0.0 = disabled
  double penaltyFreq = 0.0;

  // 0.0 = disabled
  double penaltyPresent = 0.0;

  // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
  int mirostat = 0;

  // target entropy
  double mirostatTau = 5.0;

  // learning rate
  double mirostatEta = 0.10;

  // consider newlines as a repeatable token
  bool penalizeNl = true;

  // optional BNF-like grammar to constrain sampling
  String grammar = "";

  // string to help guidance
  String cfgNegativePrompt = "";

  // how strong is guidance
  double cfgScale = 1.0;

  sampling_params toNative() {
    final sparams = calloc<sampling_params>();
    sparams.ref.n_prev = nPrev;
    sparams.ref.n_probs = nProbs;
    sparams.ref.min_keep = minKeep;
    sparams.ref.top_k = topK;
    sparams.ref.top_p = topP;
    sparams.ref.min_p = minP;
    sparams.ref.tfs_z = tfsZ;
    sparams.ref.typical_p = typicalP;
    sparams.ref.temp = temp;
    sparams.ref.dynatemp_range = dynatempRange;
    sparams.ref.dynatemp_exponent = dynatempExponent;
    sparams.ref.penalty_last_n = penaltyLastN;
    sparams.ref.penalty_repeat = penaltyRepeat;
    sparams.ref.penalty_freq = penaltyFreq;
    sparams.ref.penalty_present = penaltyPresent;
    sparams.ref.mirostat = mirostat;
    sparams.ref.mirostat_tau = mirostatTau;
    sparams.ref.mirostat_eta = mirostatEta;
    sparams.ref.penalize_nl = penalizeNl;
    sparams.ref.grammar = grammar.toNativeUtf8().cast<Char>();
    sparams.ref.cfg_negative_prompt = cfgNegativePrompt.toNativeUtf8().cast<Char>();
    sparams.ref.cfg_scale = cfgScale;
    return sparams.ref;
  }
}