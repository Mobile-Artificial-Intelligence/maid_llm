#include "maid_llm.h"
#include "llama.h"
#include "ggml.h"
#include "common.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>
#include <thread>
#include <atomic>
#include <mutex>

static std::atomic_bool stop_generation(false);
static std::mutex continue_mutex;

static llama_model * model;
static llama_context * ctx;
static llama_context * ctx_guidance;
static llama_sampling_context * ctx_sampling;

static std::vector<llama_token> embd;
static std::vector<llama_token> embd_inp;
static std::vector<llama_token> embd_guidance;
static std::vector<llama_token> guidance_inp;

static int guidance_offset;
static int original_prompt_len;
static int n_past_guidance;

static int n_remain;
static int n_past;
static int n_consumed;

static bool add_bos;

static gpt_params params;
static llama_context_params lparams;

static dart_logger *dart_logger_callback;

static llama_sampling_params from_c_sampling_params(struct sampling_params c_params) {
    llama_sampling_params cpp_params;

    cpp_params.n_prev = c_params.n_prev;
    cpp_params.n_probs = c_params.n_probs;
    cpp_params.min_keep = c_params.min_keep;
    cpp_params.top_k = c_params.top_k;
    cpp_params.top_p = c_params.top_p;
    cpp_params.min_p = c_params.min_p;
    cpp_params.tfs_z = c_params.tfs_z;
    cpp_params.typical_p = c_params.typical_p;
    cpp_params.temp = c_params.temp;
    cpp_params.dynatemp_range = c_params.dynatemp_range;
    cpp_params.dynatemp_exponent = c_params.dynatemp_exponent;
    cpp_params.penalty_last_n = c_params.penalty_last_n;
    cpp_params.penalty_repeat = c_params.penalty_repeat;
    cpp_params.penalty_freq = c_params.penalty_freq;
    cpp_params.penalty_present = c_params.penalty_present;
    cpp_params.mirostat = c_params.mirostat;
    cpp_params.mirostat_tau = c_params.mirostat_tau;
    cpp_params.mirostat_eta = c_params.mirostat_eta;
    cpp_params.penalize_nl = c_params.penalize_nl;
    cpp_params.grammar = c_params.grammar;
    cpp_params.cfg_negative_prompt = c_params.cfg_negative_prompt;
    cpp_params.cfg_scale = c_params.cfg_scale;

    return cpp_params;
}

static gpt_params from_c_params(struct gpt_c_params c_params) {
    gpt_params cpp_params;

    cpp_params.seed                     = c_params.seed;
    cpp_params.n_threads                = c_params.n_threads;
    cpp_params.n_threads_draft          = c_params.n_threads_draft;
    cpp_params.n_threads_batch          = c_params.n_threads_batch;
    cpp_params.n_threads_batch_draft    = c_params.n_threads_batch_draft;
    cpp_params.n_predict                = c_params.n_predict;
    cpp_params.n_ctx                    = c_params.n_ctx;
    cpp_params.n_batch                  = c_params.n_batch;
    cpp_params.n_keep                   = c_params.n_keep;
    cpp_params.n_draft                  = c_params.n_draft;
    cpp_params.n_chunks                 = c_params.n_chunks;
    cpp_params.n_parallel               = c_params.n_parallel;
    cpp_params.n_sequences              = c_params.n_sequences;
    cpp_params.p_accept                 = c_params.p_accept;
    cpp_params.p_split                  = c_params.p_split;
    cpp_params.n_gpu_layers             = c_params.n_gpu_layers;
    cpp_params.n_gpu_layers_draft       = c_params.n_gpu_layers_draft;

    switch (c_params.split_mode) {
        case 0:
            cpp_params.split_mode = LLAMA_SPLIT_MODE_NONE;
            break;
        case 1:
            cpp_params.split_mode = LLAMA_SPLIT_MODE_LAYER;
            break;
        case 2:
            cpp_params.split_mode = LLAMA_SPLIT_MODE_ROW;
            break;
    };

    cpp_params.main_gpu                 = c_params.main_gpu;
    cpp_params.n_beams                  = c_params.n_beams;
    cpp_params.grp_attn_n               = c_params.grp_attn_n;
    cpp_params.grp_attn_w               = c_params.grp_attn_w;
    cpp_params.n_print                  = c_params.n_print;
    cpp_params.rope_freq_base           = c_params.rope_freq_base;
    cpp_params.rope_freq_scale          = c_params.rope_freq_scale;
    cpp_params.yarn_ext_factor          = c_params.yarn_ext_factor;
    cpp_params.yarn_attn_factor         = c_params.yarn_attn_factor;
    cpp_params.yarn_beta_fast           = c_params.yarn_beta_fast;
    cpp_params.yarn_beta_slow           = c_params.yarn_beta_slow;
    cpp_params.yarn_orig_ctx            = c_params.yarn_orig_ctx;
    cpp_params.defrag_thold             = c_params.defrag_thold;
    cpp_params.rope_scaling_type        = c_params.rope_scaling_type;

    switch (c_params.numa) {
        case 0:
            cpp_params.numa = GGML_NUMA_STRATEGY_DISABLED;
            break;
        case 1:
            cpp_params.numa = GGML_NUMA_STRATEGY_DISTRIBUTE;
            break;
        case 2:
            cpp_params.numa = GGML_NUMA_STRATEGY_ISOLATE;
            break;
        case 3:
            cpp_params.numa = GGML_NUMA_STRATEGY_NUMACTL;
            break;
        case 4:
            cpp_params.numa = GGML_NUMA_STRATEGY_MIRROR;
            break;
        case 5:
            cpp_params.numa = GGML_NUMA_STRATEGY_COUNT;
            break;
    };

    cpp_params.sparams = from_c_sampling_params(c_params.sparams);

    cpp_params.model                    = c_params.model;
    cpp_params.model_draft              = c_params.model_draft;
    cpp_params.model_alias              = c_params.model_alias;
    cpp_params.prompt                   = c_params.prompt;
    cpp_params.prompt_file              = c_params.prompt_file;
    cpp_params.path_prompt_cache        = c_params.path_prompt_cache;
    cpp_params.input_prefix             = c_params.input_prefix;
    cpp_params.input_suffix             = c_params.input_suffix;
    cpp_params.antiprompt.push_back(c_params.antiprompt);
    cpp_params.logdir                   = c_params.logdir;
    cpp_params.logits_file              = c_params.logits_file;

    cpp_params.lora_base                = c_params.lora_base;

    cpp_params.ppl_stride               = c_params.ppl_stride;
    cpp_params.ppl_output_type          = c_params.ppl_output_type;

    cpp_params.hellaswag                = c_params.hellaswag;
    cpp_params.hellaswag_tasks          = c_params.hellaswag_tasks;

    cpp_params.winogrande               = c_params.winogrande;
    cpp_params.winogrande_tasks         = c_params.winogrande_tasks;

    cpp_params.multiple_choice          = c_params.multiple_choice;
    cpp_params.multiple_choice_tasks    = c_params.multiple_choice_tasks;

    cpp_params.kl_divergence            = c_params.kl_divergence;

    cpp_params.random_prompt            = c_params.random_prompt;
    cpp_params.use_color                = c_params.use_color;
    cpp_params.interactive              = c_params.interactive;
    cpp_params.chatml                   = c_params.chatml;
    cpp_params.prompt_cache_all         = c_params.prompt_cache_all;
    cpp_params.prompt_cache_ro          = c_params.prompt_cache_ro;

    cpp_params.embedding                = c_params.embedding;
    cpp_params.escape                   = c_params.escape;
    cpp_params.interactive_first        = c_params.interactive_first;
    cpp_params.multiline_input          = c_params.multiline_input;
    cpp_params.simple_io                = c_params.simple_io;
    cpp_params.cont_batching            = c_params.cont_batching;

    cpp_params.input_prefix_bos         = c_params.input_prefix_bos;
    cpp_params.ignore_eos               = c_params.ignore_eos;
    cpp_params.instruct                 = c_params.instruct;
    cpp_params.logits_all               = c_params.logits_all;
    cpp_params.use_mmap                 = c_params.use_mmap;
    cpp_params.use_mlock                = c_params.use_mlock;
    cpp_params.verbose_prompt           = c_params.verbose_prompt;
    cpp_params.display_prompt           = c_params.display_prompt;
    cpp_params.infill                   = c_params.infill;
    cpp_params.dump_kv_cache            = c_params.dump_kv_cache;
    cpp_params.no_kv_offload            = c_params.no_kv_offload;

    cpp_params.cache_type_k             = c_params.cache_type_k;
    cpp_params.cache_type_v             = c_params.cache_type_v;

    cpp_params.mmproj                   = c_params.mmproj;
    cpp_params.image                    = c_params.image;

    return cpp_params;
}

static void dart_log_callback(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    dart_logger_callback(text);
}

int maid_llm_init(struct gpt_c_params *c_params, dart_logger *log_output) {
    dart_logger_callback = log_output;

    llama_log_set(dart_log_callback, NULL);

    params = from_c_params(*c_params);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    guidance_offset = 0;
    original_prompt_len = 0;
    n_past_guidance = 0;
    n_past       = 0;
    n_consumed   = 0;
    n_remain = params.n_predict;

    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    if (model == NULL) {
        return 1;
    } else if (ctx == NULL) {
        llama_free_model(model);
        return 1;
    }

    lparams = llama_context_params_from_gpt_params(params);

    ctx_sampling = llama_sampling_init(params.sparams);

    if (params.sparams.cfg_scale > 1.f) {
        ctx_guidance = llama_new_context_with_model(model, lparams);
    }

    add_bos = llama_should_add_bos_token(model);

    // tokenize the prompt
    embd_inp = ::llama_tokenize(model, params.prompt, add_bos, true);

    // Should not run without any tokens
    if (embd_inp.empty()) {
        embd_inp.push_back(llama_token_bos(model));
        LOG("embd_inp was considered empty and bos was added: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp).c_str());
    }

    // Tokenize negative prompt
    if (ctx_guidance) {
        guidance_inp = ::llama_tokenize(ctx_guidance, params.sparams.cfg_negative_prompt, add_bos, true);

        std::vector<llama_token> original_inp = ::llama_tokenize(ctx, params.prompt, add_bos, true);

        original_prompt_len = original_inp.size();
        guidance_offset = (int)guidance_inp.size() - original_prompt_len;
    }

    if ((int) embd_inp.size() > lparams.n_ctx - 4) {
        //Truncate the prompt if it's too long
        embd_inp.erase(embd_inp.begin(), embd_inp.begin() + (embd_inp.size() - (lparams.n_ctx - 4)));
    }

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size() || params.instruct || params.chatml) {
        params.n_keep = (int)embd_inp.size();
    } else {
        params.n_keep += add_bos; // always keep the BOS token
    }

    return 0;
}

int maid_llm_prompt(const char *input, dart_output *output) {   
    std::string buffer(input);

    bool is_antiprompt = false;
    bool is_interacting = false;
    bool suffix_found = false;

    const int ga_n = params.grp_attn_n;
    const int ga_w = params.grp_attn_w;
    int ga_i = 0;
    const int n_ctx = llama_n_ctx(ctx);

    std::vector<llama_token> embd_cache;

    std::lock_guard<std::mutex> lock(continue_mutex);
    stop_generation.store(false);

    // prefix & suffix for instruct mode
    const auto inp_pfx = ::llama_tokenize(ctx, "\n\n### Instruction:\n\n", add_bos, true);
    const auto inp_sfx = ::llama_tokenize(ctx, "\n\n### Response:\n\n",    false,   true);

    // chatml prefix & suffix
    const auto cml_pfx = ::llama_tokenize(ctx, "\n<|im_start|>user\n", add_bos, true);
    const auto cml_sfx = ::llama_tokenize(ctx, "<|im_end|>\n<|im_start|>assistant\n", false, true);


    // Add tokens to embd only if the input buffer is non-empty
    // Entering a empty line lets the user pass control back
    if (buffer.length() > 1) {
        const size_t original_size = embd_inp.size();

        // instruct mode: insert instruction prefix
        if (params.instruct && !is_antiprompt) {
            n_consumed = embd_inp.size();
            embd_inp.insert(embd_inp.end(), inp_pfx.begin(), inp_pfx.end());
        }
        // chatml mode: insert user chat prefix
        if (params.chatml && !is_antiprompt) {
            n_consumed = embd_inp.size();
            embd_inp.insert(embd_inp.end(), cml_pfx.begin(), cml_pfx.end());
        }
        if (params.escape) {
            process_escapes(buffer);
        }

        const auto line_pfx = ::llama_tokenize(ctx, params.input_prefix, false, true);
        const auto line_inp = ::llama_tokenize(ctx, buffer,              false, false);
        const auto line_sfx = ::llama_tokenize(ctx, params.input_suffix, false, true);

        embd_inp.insert(embd_inp.end(), line_pfx.begin(), line_pfx.end());
        embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
        embd_inp.insert(embd_inp.end(), line_sfx.begin(), line_sfx.end());

        // instruct mode: insert response suffix
        if (params.instruct) {
            embd_inp.insert(embd_inp.end(), inp_sfx.begin(), inp_sfx.end());
        }
        // chatml mode: insert assistant chat suffix
        if (params.chatml) {
            embd_inp.insert(embd_inp.end(), cml_sfx.begin(), cml_sfx.end());
        }

        n_remain -= line_inp.size();
    }

    while ((n_remain != 0 && !is_antiprompt) || params.interactive) {
        if (stop_generation.load()) {
            stop_generation.store(false);  // reset for future use
            output("", true);
            return 0;  // or any other cleanup you want to do
        }

        // predict
        if (!embd.empty()) {
            // Note: lparams.n_ctx - 4 here is to match the logic for commandline prompt handling via
            // --prompt or --file which uses the same value.
            int max_embd_size = lparams.n_ctx - 4;

            // Ensure the input doesn't exceed the context size by truncating embd if necessary.
            if ((int) embd.size() > max_embd_size) {
                const int skipped_tokens = (int) embd.size() - max_embd_size;
                embd.resize(max_embd_size);
            }

            if (ga_n == 1) {
                // infinite text generation via context shifting
                // if we run out of context:
                // - take the n_keep first tokens from the original prompt (via n_past)
                // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
                if (n_past + (int) embd.size() + std::max<int>(0, guidance_offset) > n_ctx) {
                    if (params.n_predict == -2) {
                        break;
                    }

                    const int n_left    = n_past - params.n_keep;
                    const int n_discard = n_left/2;


                    llama_kv_cache_seq_rm (ctx, 0, params.n_keep, params.n_keep + n_discard);
                    llama_kv_cache_seq_add(ctx, 0, params.n_keep + n_discard, n_past, -n_discard);

                    n_past -= n_discard;

                    if (ctx_guidance) {
                        n_past_guidance -= n_discard;
                    }
                }
            } else {
                // context extension via Self-Extend
                while (n_past >= ga_i + ga_w) {
                    const int ib = (ga_n*ga_i)/ga_w;
                    const int bd = (ga_w/ga_n)*(ga_n - 1);
                    const int dd = (ga_w/ga_n) - ib*bd - ga_w;

                    llama_kv_cache_seq_add(ctx, 0, ga_i,                n_past,              ib*bd);
                    llama_kv_cache_seq_div(ctx, 0, ga_i + ib*bd,        ga_i + ib*bd + ga_w, ga_n);
                    llama_kv_cache_seq_add(ctx, 0, ga_i + ib*bd + ga_w, n_past + ib*bd,      dd);

                    n_past -= bd;

                    ga_i += ga_w/ga_n;
                }
            }

            // evaluate tokens in batches
            // embd is typically prepared beforehand to fit within a batch, but not always
            if (ctx_guidance) {
                int input_size = 0;
                llama_token * input_buf = NULL;

                if (n_past_guidance < (int) guidance_inp.size()) {
                    // Guidance context should have the same data with these modifications:
                    //
                    // * Replace the initial prompt
                    // * Shift everything by guidance_offset
                    embd_guidance = guidance_inp;
                    if (embd.begin() + original_prompt_len < embd.end()) {
                        embd_guidance.insert(
                            embd_guidance.end(),
                            embd.begin() + original_prompt_len,
                            embd.end()
                        );
                    }

                    input_buf  = embd_guidance.data();
                    input_size = embd_guidance.size();
                } else {
                    input_buf  = embd.data();
                    input_size = embd.size();
                }

                for (int i = 0; i < input_size; i += params.n_batch) {
                    int n_eval = std::min(input_size - i, params.n_batch);
                    if (llama_decode(ctx_guidance, llama_batch_get_one(input_buf + i, n_eval, n_past_guidance, 0))) {
                        output("", true);
                        return 1;
                    }

                    n_past_guidance += n_eval;
                }
            }

            for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
                int n_eval = (int) embd.size() - i;
                if (n_eval > params.n_batch) {
                    n_eval = params.n_batch;
                }

                if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval, n_past, 0))) {
                    output("", true);
                    return 1;
                }

                n_past += n_eval;
            }
        }

        embd.clear();
        embd_guidance.clear();

        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
            const llama_token id = llama_sampling_sample(ctx_sampling, ctx, ctx_guidance);

            llama_sampling_accept(ctx_sampling, ctx, id, true);

            embd.push_back(id);

            // decrement remaining sampling budget
            --n_remain;
        } else {
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);

                // push the prompt in the sampling context in order to apply repetition penalties later
                // for the prompt, we don't apply grammar rules
                llama_sampling_accept(ctx_sampling, ctx, embd_inp[n_consumed], false);

                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        // display text
        for (auto id : embd) {
            output(llama_token_to_piece(ctx, id).c_str(), false);
        }

        // if not currently processing queued inputs;
        if ((int) embd_inp.size() <= n_consumed) {
            // check for reverse prompt in the last n_prev tokens
            if (!params.antiprompt.empty()) {
                const int n_prev = 32;
                const std::string last_output = llama_sampling_prev_str(ctx_sampling, ctx, n_prev);

                is_antiprompt = false;
                // Check if each of the reverse prompts appears at the end of the output.
                // If we're not running interactively, the reverse prompt might be tokenized with some following characters
                // so we'll compensate for that by widening the search window a bit.
                for (std::string & antiprompt : params.antiprompt) {
                    size_t extra_padding = params.interactive ? 0 : 2;
                    size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                        ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                        : 0;

                    if (last_output.find(antiprompt, search_start_pos) != std::string::npos) {
                        if (params.interactive) {
                            is_interacting = true;
                        }
                        is_antiprompt = true;
                        break;
                    }
                }

                if (is_antiprompt) {
                    LOG("found antiprompt: %s\n", last_output.c_str());
                }
            }

            // deal with end of text token in interactive mode
            if (llama_sampling_last(ctx_sampling) == llama_token_eos(model)) {
                if (params.interactive) {
                    if (!params.antiprompt.empty()) {
                        // tokenize and inject first reverse prompt
                        const auto first_antiprompt = ::llama_tokenize(ctx, params.antiprompt.front(), false, true);
                        embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                        is_antiprompt = true;
                    }

                    is_interacting = true;
                } else if (params.instruct || params.chatml) {
                    is_interacting = true;
                }
            }

            if (n_past > 0) {
                if (is_interacting) {
                    llama_sampling_reset(ctx_sampling);
                }
                is_interacting = false;
            }
        }

        // end of text token
        if (!embd.empty() && embd.back() == llama_token_eos(model) && !(params.instruct || params.interactive || params.chatml)) {
            break;
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        // We skip this logic when n_predict == -1 (infinite) or -2 (stop at context size).
        if (params.interactive && n_remain <= 0 && params.n_predict >= 0) {
            n_remain = params.n_predict;
            is_interacting = true;
        }
    }

    output("", true);
    return 0;
}

void maid_llm_stop(void) {
    stop_generation.store(true);
}

void maid_llm_cleanup(void) {
    stop_generation.store(true);
    llama_print_timings(ctx);
    llama_free(ctx);
    llama_free(ctx_guidance);
    llama_free_model(model);
    llama_sampling_free(ctx_sampling);
    llama_backend_free();
}