part of '../lcpp.dart';

typedef PArgs = ({
  double p, 
  int minKeep
});

typedef TemperatureArgs = ({
  double temperature, 
  double? delta, 
  double? exponent
});

typedef XtcArgs = ({
  double p, 
  double t, 
  int minKeep, 
  int seed
});

typedef MirostatArgs = ({
  int nVocab, 
  int seed, 
  double tau, 
  double eta, 
  int m
});

typedef MirostatV2Args = ({
  int seed, 
  double tau, 
  double eta
});

typedef GrammarArgs = ({
  String str, 
  String root
});

typedef PenaltiesArgs = ({
  int lastN, 
  double repeat, 
  double frequency, 
  double present
});

typedef DrySamplerArgs = ({
  int nCtxTrain, 
  double multiplier, 
  double dryBase, 
  int allowedLength, 
  int penaltyLastN, 
  List<String> sequenceBreakers
});

class SamplingParams {
  bool greedy;
  bool infill;
  int? seed;
  int? topK;
  PArgs? topP;
  PArgs? minP;
  PArgs? typicalP;
  TemperatureArgs? temperature;
  XtcArgs? xtc;
  MirostatArgs? mirostat;
  MirostatV2Args? mirostatV2;
  GrammarArgs? grammar;
  PenaltiesArgs? penalties;
  DrySamplerArgs? drySampler;

  SamplingParams({
    this.greedy = false,
    this.infill = false,
    this.seed,
    this.topK,
    this.topP,
    this.minP,
    this.typicalP,
    this.temperature,
    this.xtc,
    this.mirostat,
    this.mirostatV2,
    this.grammar,
    this.penalties,
    this.drySampler
  });

  ffi.Pointer<llama_sampler> toNative(ffi.Pointer<llama_vocab> vocab) {
    final sampler = LlamaCPP.lib.llama_sampler_chain_init(LlamaCPP.lib.llama_sampler_chain_default_params());

    if (greedy) {
      LlamaCPP.lib.llama_sampler_chain_add(sampler, LlamaCPP.lib.llama_sampler_init_greedy());
    }

    if (infill) {
      LlamaCPP.lib.llama_sampler_chain_add(sampler, LlamaCPP.lib.llama_sampler_init_infill(vocab));
    }

    if (seed != null) {
      LlamaCPP.lib.llama_sampler_chain_add(sampler, LlamaCPP.lib.llama_sampler_init_dist(seed!));
    }

    if (topK != null) {
      LlamaCPP.lib.llama_sampler_chain_add(sampler, LlamaCPP.lib.llama_sampler_init_top_k(topK!));
    }

    if (topP != null) {
      LlamaCPP.lib.llama_sampler_chain_add(sampler, LlamaCPP.lib.llama_sampler_init_top_p(topP!.p, topP!.minKeep));
    }

    if (minP != null) {
      LlamaCPP.lib.llama_sampler_chain_add(sampler, LlamaCPP.lib.llama_sampler_init_min_p(minP!.p, minP!.minKeep));
    }

    if (typicalP != null) {
      LlamaCPP.lib.llama_sampler_chain_add(sampler, LlamaCPP.lib.llama_sampler_init_typical(typicalP!.p, typicalP!.minKeep));
    }

    if (temperature != null) {
      if (temperature!.delta == null && temperature!.exponent == null) {
        LlamaCPP.lib.llama_sampler_chain_add(
          sampler, 
          LlamaCPP.lib.llama_sampler_init_temp(temperature!.temperature)
        );
      } 
      else {
        LlamaCPP.lib.llama_sampler_chain_add(
          sampler, 
          LlamaCPP.lib.llama_sampler_init_temp_ext(temperature!.temperature, temperature!.delta!, temperature!.exponent!)
        );
      }
    }

    if (xtc != null) {
      LlamaCPP.lib.llama_sampler_chain_add(
        sampler, 
        LlamaCPP.lib.llama_sampler_init_xtc(xtc!.p, xtc!.t, xtc!.minKeep, xtc!.seed)
      );
    }

    if (mirostat != null) {
      LlamaCPP.lib.llama_sampler_chain_add(
        sampler, 
        LlamaCPP.lib.llama_sampler_init_mirostat(mirostat!.nVocab, mirostat!.seed, mirostat!.tau, mirostat!.eta, mirostat!.m)
      );
    }

    if (mirostatV2 != null) {
      LlamaCPP.lib.llama_sampler_chain_add(
        sampler, 
        LlamaCPP.lib.llama_sampler_init_mirostat_v2(mirostatV2!.seed, mirostatV2!.tau, mirostatV2!.eta)
      );
    }

    if (grammar != null) {
      LlamaCPP.lib.llama_sampler_chain_add(
        sampler, 
        LlamaCPP.lib.llama_sampler_init_grammar(
          vocab, 
          grammar!.str.toNativeUtf8().cast<ffi.Char>(), 
          grammar!.root.toNativeUtf8().cast<ffi.Char>()
        )
      );
    }

    if (penalties != null) {
      LlamaCPP.lib.llama_sampler_chain_add(
        sampler, 
        LlamaCPP.lib.llama_sampler_init_penalties(
          penalties!.lastN, 
          penalties!.repeat, 
          penalties!.frequency, 
          penalties!.present
        )
      );
    }

    if (drySampler != null) {
      final sequenceBreakers = calloc<ffi.Pointer<ffi.Char>>(drySampler!.sequenceBreakers.length);
      for (var i = 0; i < drySampler!.sequenceBreakers.length; i++) {
        sequenceBreakers[i] = drySampler!.sequenceBreakers[i].toNativeUtf8().cast<ffi.Char>();
      }

      LlamaCPP.lib.llama_sampler_chain_add(
        sampler, 
        LlamaCPP.lib.llama_sampler_init_dry(
          vocab,
          drySampler!.nCtxTrain, 
          drySampler!.multiplier, 
          drySampler!.dryBase, 
          drySampler!.allowedLength, 
          drySampler!.penaltyLastN, 
          sequenceBreakers,
          drySampler!.sequenceBreakers.length
        )
      );
    }

    return sampler;
  }
}