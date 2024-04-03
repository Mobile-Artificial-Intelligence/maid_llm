#include "maid_llm.hpp"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <chrono>
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
static gpt_params params;

static llama_context * ctx;
static llama_context * ctx_guidance;

static int n_past;
static int n_past_guidance;


EXPORT int maid_llm_model_init(struct gpt_c_params *c_params, dart_logger *log_output) {
    auto init_start_time = std::chrono::high_resolution_clock::now();

    params = from_c_params(*c_params);

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model_params mparams = llama_model_params_from_gpt_params(params);
    model = llama_load_model_from_file(params.model.c_str(), mparams);
    if (model == NULL) {
        return 1;
    }

    auto init_end_time = std::chrono::high_resolution_clock::now();
    log_output(("Model init in " + get_elapsed_seconds(init_end_time - init_start_time)).c_str());

    return 0;
}

EXPORT int maid_llm_context_init(struct gpt_c_params *c_params, dart_logger *log_output) {
    auto init_start_time = std::chrono::high_resolution_clock::now();

    llama_context_params lparams = llama_context_params_from_gpt_params(params);

    ctx = llama_new_context_with_model(model, lparams);

    if (params.sparams.cfg_scale > 1.f) {
        ctx_guidance = llama_new_context_with_model(model, lparams);
    }

    n_past = 0;
    n_past_guidance = 0;

    auto init_end_time = std::chrono::high_resolution_clock::now();
    log_output(("Context init in " + get_elapsed_seconds(init_end_time - init_start_time)).c_str());

    return 0;
}

EXPORT int maid_llm_prompt(int msg_count, struct chat_message* messages[], dart_output *output, dart_logger *log_output) {
    auto prompt_start_time = std::chrono::high_resolution_clock::now();

    std::lock_guard<std::mutex> lock(continue_mutex);
    stop_generation.store(false);

    llama_sampling_context * ctx_sampling = llama_sampling_init(params.sparams);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    const int n_ctx = llama_n_ctx(ctx);
    
    bool is_antiprompt = false;
    bool is_interacting = false;
    bool add_bos = llama_should_add_bos_token(model);

    int guidance_offset = 0;
    int original_prompt_len = 0; 
    int n_consumed = 0;
    int n_remain = params.n_predict;
    int ga_i = 0;
    int n_prior = n_past;
    int terminator_length = 0;

    const int ga_n = params.grp_attn_n;
    const int ga_w = params.grp_attn_w;

    std::vector<llama_token> embd;
    std::vector<llama_token> embd_inp;
    std::vector<llama_token> embd_cache;
    std::vector<llama_token> embd_out;
    std::vector<llama_token> embd_guidance;
    std::vector<llama_token> guidance_inp;  

    auto passed_lock_time = std::chrono::high_resolution_clock::now();
    log_output(("Passed lock in " + get_elapsed_seconds(passed_lock_time - prompt_start_time)).c_str());

    // parse messages
    embd_inp = parse_messages(msg_count, messages, ctx);

    auto finished_message_parsing_time = std::chrono::high_resolution_clock::now();
    log_output(("Parsed messages in " + get_elapsed_seconds(finished_message_parsing_time - passed_lock_time)).c_str()); 
    
    // Should not run without any tokens
    if (embd_inp.empty()) {
        embd_inp.push_back(llama_token_bos(model));
        log_output(("embd_inp was considered empty and bos was added: " + LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp)).c_str());
    }

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size() || params.instruct || params.chatml) {
        params.n_keep = (int)embd_inp.size();
    } else {
        params.n_keep += add_bos; // always keep the BOS token
    }

    if (params.instruct) {
        // instruct mode: insert instruction prefix to antiprompts
        params.antiprompt.push_back("### Instruction:");
        params.antiprompt.push_back("\n\n\n\n\n");
        terminator_length = llama_tokenize(ctx, "### Instruction:", false, true).size();
    }

    if (params.chatml) {
        // chatml mode: insert user chat prefix to antiprompts

        params.antiprompt.push_back("<|im_end|>");
        params.antiprompt.push_back("\n\n\n\n\n");
        terminator_length = llama_tokenize(ctx, "<|im_end|>", false, true).size();
    }

    // Tokenize negative prompt
    if (params.sparams.cfg_scale > 1.f) {
        guidance_inp = ::llama_tokenize(ctx_guidance, params.sparams.cfg_negative_prompt, add_bos, true);

        std::vector<llama_token> original_inp = ::llama_tokenize(ctx, params.prompt, add_bos, true);

        original_prompt_len = original_inp.size();
        guidance_offset = (int)guidance_inp.size() - original_prompt_len;
    }

    bool has_output = false;

    while ((n_remain != 0 && !is_antiprompt) || params.interactive) {
        // predict
        if (!embd.empty()) {
            auto prediction_start_time = std::chrono::high_resolution_clock::now();

            // Note: n_ctx - 4 here is to match the logic for commandline prompt handling via
            // --prompt or --file which uses the same value.
            int max_embd_size = n_ctx - 4;

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

                    if (ctx_guidance != NULL) {
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
            if (ctx_guidance != NULL) {
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

            auto prediction_end_time = std::chrono::high_resolution_clock::now();
            log_output(("Predicted in " + get_elapsed_seconds(prediction_end_time - prediction_start_time)).c_str());
        }

        embd.clear();
        embd_guidance.clear();

        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
            auto sample_start_time = std::chrono::high_resolution_clock::now();

            const llama_token id = llama_sampling_sample(ctx_sampling, ctx, ctx_guidance);

            llama_sampling_accept(ctx_sampling, ctx, id, true);

            embd.push_back(id);

            auto sample_end_time = std::chrono::high_resolution_clock::now();
            log_output(("Sampled token in " + get_elapsed_seconds(sample_end_time - sample_start_time)).c_str());

            // decrement remaining sampling budget
            --n_remain;
        } else {
            auto push_start_time = std::chrono::high_resolution_clock::now();

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

            auto push_end_time = std::chrono::high_resolution_clock::now();
            log_output(("Pushed tokens in " + get_elapsed_seconds(push_end_time - push_start_time)).c_str());
        }

        // Cache the last terminator_length tokens for display
        // This is done to ensure the antiprompt is detected correctly
        embd_cache.insert(embd_cache.end(), embd.begin(), embd.end());
        if ((int) embd_cache.size() > terminator_length) {
            // display text
            while ((int) embd_cache.size() > terminator_length) {
                auto id = embd_cache.front();
                embd_out.push_back(id);
                embd_cache.erase(embd_cache.begin());
            }
        }

        if ((n_past - terminator_length + 1 >= (int) embd_inp.size() + n_prior && embd_out.size() > 0) || !(params.instruct || params.interactive || params.chatml)) {
            for (auto id : embd_out) {
                output(llama_token_to_piece(ctx, id).c_str(), false);
            }

            if (!has_output) {
                auto first_output_time = std::chrono::high_resolution_clock::now();
                log_output(("First output in " + get_elapsed_seconds(first_output_time - prompt_start_time)).c_str());
            }

            has_output = true;
        }

        embd_out.clear();

        if (stop_generation.load()) {
            stop_generation.store(false);  // reset for future use
            break;
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

    llama_sampling_free(ctx_sampling);
    output("", true);
    return 0;
}

EXPORT void maid_llm_stop(void) {
    stop_generation.store(true);
}

EXPORT void maid_llm_cleanup(void) {
    stop_generation.store(true);
    llama_free(ctx);
    llama_free(ctx_guidance);
    llama_free_model(model);
    llama_backend_free();
}

// ===========================================================================================
//                                      Utility functions
// ===========================================================================================

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

    switch (c_params.rope_scaling_type) {
        case -1:
            cpp_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
            break;
        case 0:
            cpp_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE;
            break;
        case 1:
            cpp_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
            break;
        case 2:
            cpp_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
            break;
        case 3:
            cpp_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_MAX_VALUE;
            assert(false);
            break;
    }

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
            assert(false);
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
    
    if (strcmp(c_params.antiprompt, "") != 0) {
        cpp_params.antiprompt.push_back(c_params.antiprompt);
    }

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

std::vector<llama_token> parse_messages(int msg_count, chat_message* messages[], llama_context * ctx) {
    std::vector<llama_token> embd_inp;

    std::vector<llama_token> inp_pfx;
    std::vector<llama_token> res_pfx;
    std::vector<llama_token> sys_pfx;
    std::vector<llama_token> sfx;

    bool add_bos = llama_should_add_bos_token(model);

    if (params.instruct) {
        // prefixes & suffix for instruct mode
        inp_pfx = ::llama_tokenize(ctx, "### Instruction:\n\n",   add_bos, true);
        res_pfx = ::llama_tokenize(ctx, "### Response:\n\n",        false, true);
        sys_pfx = ::llama_tokenize(ctx, "### System:\n\n",          false, true);
        sfx     = ::llama_tokenize(ctx, "\n\n",                     false, true);
    } else if (params.chatml) {
        // prefixes & suffix for chatml mode
        inp_pfx = ::llama_tokenize(ctx, "<|im_start|>user\n",     add_bos, true);
        res_pfx = ::llama_tokenize(ctx, "<|im_start|>assistant\n",  false, true);
        sys_pfx = ::llama_tokenize(ctx, "<|im_start|>system\n",     false, true);
        sfx     = ::llama_tokenize(ctx, "<|im_end|>\n",             false, true);
    }

    const auto line_pfx = ::llama_tokenize(ctx, params.input_prefix, false, true);
    const auto line_sfx = ::llama_tokenize(ctx, params.input_suffix, false, true);

    for (int i = 0; i < msg_count; i++) {
        chat_message* message = messages[i];
        std::string buffer(message->content);

        printf("role: %d, content: %s\n", message->role, message->content);

        // Add tokens to embd only if the input buffer is non-empty
        // Entering a empty line lets the user pass control back
        if (buffer.length() > 1) {
            switch (message->role) {
                case ROLE_USER:
                    // insert user chat prefix
                    if (params.instruct || params.chatml) {
                        embd_inp.insert(embd_inp.end(), inp_pfx.begin(), inp_pfx.end());
                    }

                    embd_inp.insert(embd_inp.end(), line_pfx.begin(), line_pfx.end());
                    break;
                case ROLE_ASSISTANT:
                    // insert assistant chat prefix
                    if (params.instruct || params.chatml) {
                        embd_inp.insert(embd_inp.end(), res_pfx.begin(), res_pfx.end());
                    }
                    break;
                case ROLE_SYSTEM:
                    // insert system chat prefix
                    if (params.instruct || params.chatml) {
                        embd_inp.insert(embd_inp.end(), sys_pfx.begin(), sys_pfx.end());
                    }
                    break;
            }

            if (params.escape) process_escapes(buffer);            
            const auto line_inp = ::llama_tokenize(ctx, buffer, false, false);
            embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());

            if (message->role == ROLE_USER) {
                embd_inp.insert(embd_inp.end(), line_sfx.begin(), line_sfx.end());
            }

            if (params.instruct || params.chatml) {
                embd_inp.insert(embd_inp.end(), sfx.begin(), sfx.end());
            }
        }
    }

    // insert response prefix
    if (params.instruct || params.chatml) {
        embd_inp.insert(embd_inp.end(), res_pfx.begin(), res_pfx.end());
    }

    return embd_inp;
}

std::string get_elapsed_seconds(const std::chrono::nanoseconds &__d) {
    return std::to_string(std::chrono::duration<double>(__d).count()) + " seconds";
}