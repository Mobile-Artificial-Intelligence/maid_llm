#include "maid_llm.hpp"
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

static std::vector<llama_token> last_n_tokens;
static std::vector<llama_token> embd;
static std::vector<llama_token> embd_inp;

static int n_remain;
static int n_past;
static int n_consumed;
static signed int prior;

static gpt_params params;
static llama_context_params lparams;

static dart_logger *dart_logger_callback;

static gpt_params from_c_params(struct gpt_c_params *c_params) {
    gpt_params cpp_params;

    cpp_params.seed                     = c_params->seed;
    cpp_params.n_threads                = c_params->n_threads;
    cpp_params.n_threads_draft          = c_params->n_threads_draft;
    cpp_params.n_threads_batch          = c_params->n_threads_batch;
    cpp_params.n_threads_batch_draft    = c_params->n_threads_batch_draft;
    cpp_params.n_predict                = c_params->n_predict;
    cpp_params.n_ctx                    = c_params->n_ctx;
    cpp_params.n_batch                  = c_params->n_batch;
    cpp_params.n_keep                   = c_params->n_keep;
    cpp_params.n_draft                  = c_params->n_draft;
    cpp_params.n_chunks                 = c_params->n_chunks;
    cpp_params.n_parallel               = c_params->n_parallel;
    cpp_params.n_sequences              = c_params->n_sequences;
    cpp_params.p_accept                 = c_params->p_accept;
    cpp_params.p_split                  = c_params->p_split;
    cpp_params.n_gpu_layers             = c_params->n_gpu_layers;
    cpp_params.n_gpu_layers_draft       = c_params->n_gpu_layers_draft;

    switch (c_params->split_mode) {
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

    cpp_params.main_gpu                 = c_params->main_gpu;
    cpp_params.n_beams                  = c_params->n_beams;
    cpp_params.grp_attn_n               = c_params->grp_attn_n;
    cpp_params.grp_attn_w               = c_params->grp_attn_w;
    cpp_params.n_print                  = c_params->n_print;
    cpp_params.rope_freq_base           = c_params->rope_freq_base;
    cpp_params.rope_freq_scale          = c_params->rope_freq_scale;
    cpp_params.yarn_ext_factor          = c_params->yarn_ext_factor;
    cpp_params.yarn_attn_factor         = c_params->yarn_attn_factor;
    cpp_params.yarn_beta_fast           = c_params->yarn_beta_fast;
    cpp_params.yarn_beta_slow           = c_params->yarn_beta_slow;
    cpp_params.yarn_orig_ctx            = c_params->yarn_orig_ctx;
    cpp_params.defrag_thold             = c_params->defrag_thold;
    cpp_params.rope_scaling_type        = c_params->rope_scaling_type;

    switch (c_params->numa) {
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

    cpp_params.model                    = c_params->model;
    cpp_params.model_draft              = c_params->model_draft;
    cpp_params.model_alias              = c_params->model_alias;
    cpp_params.prompt                   = c_params->prompt;
    cpp_params.prompt_file              = c_params->prompt_file;
    cpp_params.path_prompt_cache        = c_params->path_prompt_cache;
    cpp_params.input_prefix             = c_params->input_prefix;
    cpp_params.input_suffix             = c_params->input_suffix;
    cpp_params.antiprompt.push_back(c_params->antiprompt);
    cpp_params.logdir                   = c_params->logdir;
    cpp_params.logits_file              = c_params->logits_file;

    cpp_params.lora_base                = c_params->lora_base;

    cpp_params.ppl_stride               = c_params->ppl_stride;
    cpp_params.ppl_output_type          = c_params->ppl_output_type;

    cpp_params.hellaswag                = c_params->hellaswag;
    cpp_params.hellaswag_tasks          = c_params->hellaswag_tasks;

    cpp_params.winogrande               = c_params->winogrande;
    cpp_params.winogrande_tasks         = c_params->winogrande_tasks;

    cpp_params.multiple_choice          = c_params->multiple_choice;
    cpp_params.multiple_choice_tasks    = c_params->multiple_choice_tasks;

    cpp_params.kl_divergence            = c_params->kl_divergence;

    cpp_params.mul_mat_q                = c_params->mul_mat_q;
    cpp_params.random_prompt            = c_params->random_prompt;
    cpp_params.use_color                = c_params->use_color;
    cpp_params.interactive              = c_params->interactive;
    cpp_params.chatml                   = c_params->chatml;
    cpp_params.prompt_cache_all         = c_params->prompt_cache_all;
    cpp_params.prompt_cache_ro          = c_params->prompt_cache_ro;

    cpp_params.embedding                = c_params->embedding;
    cpp_params.escape                   = c_params->escape;
    cpp_params.interactive_first        = c_params->interactive_first;
    cpp_params.multiline_input          = c_params->multiline_input;
    cpp_params.simple_io                = c_params->simple_io;
    cpp_params.cont_batching            = c_params->cont_batching;

    cpp_params.input_prefix_bos         = c_params->input_prefix_bos;
    cpp_params.ignore_eos               = c_params->ignore_eos;
    cpp_params.instruct                 = c_params->instruct;
    cpp_params.logits_all               = c_params->logits_all;
    cpp_params.use_mmap                 = c_params->use_mmap;
    cpp_params.use_mlock                = c_params->use_mlock;
    cpp_params.verbose_prompt           = c_params->verbose_prompt;
    cpp_params.display_prompt           = c_params->display_prompt;
    cpp_params.infill                   = c_params->infill;
    cpp_params.dump_kv_cache            = c_params->dump_kv_cache;
    cpp_params.no_kv_offload            = c_params->no_kv_offload;

    cpp_params.cache_type_k             = c_params->cache_type_k;
    cpp_params.cache_type_v             = c_params->cache_type_v;

    cpp_params.mmproj                   = c_params->mmproj;
    cpp_params.image                    = c_params->image;

    return cpp_params;
}

static void dart_log_callback(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    dart_logger_callback(text);
}

int maid_llm_init(struct gpt_c_params *c_params, dart_logger *log_output) {
    llama_backend_init();

    dart_logger_callback = log_output;

    llama_log_set(dart_log_callback, NULL);

    n_past       = 0;
    n_consumed   = 0;

    params = from_c_params(c_params);


    n_remain = params.n_predict;

    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model);
        return 1;
    } else if (ctx == NULL) {
        fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, params.model);
        llama_free_model(model);
        return 1;
    }

    lparams = llama_context_params_from_gpt_params(params);

    ctx_sampling = llama_sampling_init(params.sparams);

    if (params.sparams.cfg_scale > 1.f) {
        ctx_guidance = llama_new_context_with_model(model, lparams);
    }

    const bool add_bos = llama_should_add_bos_token(model);

    // tokenize the prompt
    embd_inp = ::llama_tokenize(model, params.prompt, add_bos, true);

    if ((int) embd_inp.size() > lparams.n_ctx - 4) {
        //Truncate the prompt if it's too long
        embd_inp.erase(embd_inp.begin(), embd_inp.begin() + (embd_inp.size() - (lparams.n_ctx - 4)));
    }

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size()) {
        params.n_keep = (int)embd_inp.size();
    }

    last_n_tokens = std::vector<llama_token>(lparams.n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    prior = embd_inp.size();

    return 0;
}

int maid_llm_prompt(const char *input, dart_output *output) {   
    std::string buffer(input);

    bool is_interacting = false;
    bool suffix_found = false;

    int n_pfx = 0;
    int n_sfx = 0;

    std::vector<llama_token> embd_cache;

    std::lock_guard<std::mutex> lock(continue_mutex);
    stop_generation.store(false);

    // Add tokens to embd only if the input buffer is non-empty
    // Entering a empty line lets the user pass control back
    if (buffer.length() > 1) {
        const auto inp_text = ::llama_tokenize(model, buffer, false, false);
        embd_inp.insert(embd_inp.end(), inp_text.begin(), inp_text.end());
        n_remain -= inp_text.size();
    }

    while (true) {
        if (stop_generation.load()) {
            stop_generation.store(false);  // reset for future use
            output("", true);
            return 0;  // or any other cleanup you want to do
        }

        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
            const llama_token id = llama_sampling_sample(ctx_sampling, ctx, NULL);

            llama_sampling_accept(ctx_sampling, ctx, id, true);

            embd.push_back(id);

            // decrement remaining sampling budget
            --n_remain;
        } else {
            // some user input remains from prompt or interaction, forward it to processing
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

            // infinite text generation via context swapping
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
            if (n_past + (int) embd.size() > lparams.n_ctx) {
                if (params.n_predict == -2) {
                    LOG_TEE("\n\n%s: context full and n_predict == -%d => stopping\n", __func__, params.n_predict);
                    break;
                }

                const int n_left    = n_past - params.n_keep - 1;
                const int n_discard = n_left/2;

                LOG("context full, swapping: n_past = %d, n_left = %d, lparams.n_ctx = %d, n_keep = %d, n_discard = %d\n",
                    n_past, n_left, lparams.n_ctx, params.n_keep, n_discard);

                llama_kv_cache_seq_rm   (ctx, 0, params.n_keep + 1            , params.n_keep + n_discard + 1);

                n_past -= n_discard;

                LOG("embd: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd).c_str());
            }

            for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
                int n_eval = (int) embd.size() - i;
                if (n_eval > params.n_batch) {
                    n_eval = params.n_batch;
                }

                LOG("eval: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd).c_str());

                if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval, n_past, 0))) {
                    LOG_TEE("%s : failed to eval\n", __func__);
                    return 1;
                }

                n_past += n_eval;

                LOG("n_past = %d\n", n_past);
            }
        }

        embd.clear();

        // if not currently processing queued inputs;
        if ((int) embd_inp.size() <= n_consumed) {
            // check for reverse prompt
            if (!params.antiprompt.empty()) {
                const int n_prev = 32;
                const std::string last_output = llama_sampling_prev_str(ctx_sampling, ctx, n_prev);

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
                        output("", true);
                        return 0;
                    }
                }
            }

             // deal with end of text token in interactive mode
            if (llama_sampling_last(ctx_sampling) == llama_token_eos(model)) {
                LOG("found EOS token\n");

                if (params.interactive) {
                    if (!params.antiprompt.empty()) {
                        // tokenize and inject first reverse prompt
                        const auto first_antiprompt = ::llama_tokenize(ctx, params.antiprompt.front(), false, true);
                        embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                        output("", true);
                        return 0;
                    }

                    is_interacting = true;
                    printf("\n");
                } else if (params.instruct) {
                    is_interacting = true;
                }
            }

            if (n_past > 0 && is_interacting) {
                output("", true);
                return 0;
            }

            if (n_past > 0) {
                if (is_interacting) {
                    llama_sampling_reset(ctx_sampling);
                }
                is_interacting = false;
            }
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