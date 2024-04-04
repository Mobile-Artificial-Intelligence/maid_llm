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

    if (params.instruct) {
        // instruct mode: insert instruction prefix to antiprompts
        params.antiprompt.push_back("### Instruction:");
    }

    if (params.chatml) {
        // chatml mode: insert user chat prefix to antiprompts
        params.antiprompt.push_back("<|im_end|>");
    }

    params.antiprompt.push_back("\n\n\n\n\n");

    auto init_end_time = std::chrono::high_resolution_clock::now();
    log_output(("Model init in " + get_elapsed_seconds(init_end_time - init_start_time)).c_str());

    return 0;
}

EXPORT int maid_llm_context_init(struct gpt_c_params *c_params, dart_logger *log_output) {
    auto init_start_time = std::chrono::high_resolution_clock::now();

    llama_context_params lparams = llama_context_params_from_gpt_params(params);

    ctx = llama_new_context_with_model(model, lparams);

    auto init_end_time = std::chrono::high_resolution_clock::now();
    log_output(("Context init in " + get_elapsed_seconds(init_end_time - init_start_time)).c_str());

    return 0;
}

EXPORT int maid_llm_prompt(int msg_count, struct chat_message* messages[], dart_output *output, dart_logger *log_output) {
    auto prompt_start_time = std::chrono::high_resolution_clock::now();

    std::lock_guard<std::mutex> lock(continue_mutex);
    stop_generation.store(false);

    llama_sampling_context * ctx_sampling = llama_sampling_init(params.sparams);

    llama_batch batch = llama_batch_init(params.n_batch, 0, params.n_sequences);

    const int n_ctx = llama_n_ctx(ctx);
    
    bool add_bos = llama_should_add_bos_token(model);

    int n_past = 0;
    int n_sample = n_ctx;

    if (n_sample <= 0) {
        n_sample = n_ctx;
    }

    std::vector<llama_token> embd_inp;      // input buffer

    // parse messages
    embd_inp = parse_messages(msg_count, messages, ctx);

    //Truncate the prompt if it's too long
    if ((int) embd_inp.size() > n_ctx - 4) {
        // truncate the input
        embd_inp.erase(embd_inp.begin(), embd_inp.begin() + (embd_inp.size() - (n_ctx - 4)));

        // log the truncation
        log_output(("embd_inp was truncated: " + LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp)).c_str());
    } 
    
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

    // evaluate the initial prompt
    for (size_t i = 0; i < embd_inp.size(); i++) {
        llama_batch_add(batch, embd_inp[i], i, { 0 }, false);
    }

    if (llama_decode(ctx, batch) != 0) {
        log_output("Initial decode failed");
        return 1;
    }

    llama_token output_token = llama_token_bos(model);

    while (n_past < n_sample) {
        if (stop_generation.load()) {
            stop_generation.store(false);  // reset for future use
            break;
        }

        if (output_token != llama_token_bos(model)) {
            output(llama_token_to_piece(ctx, output_token).c_str(), false);
        }

        // sample the most likely token
        output_token = llama_sampling_sample(ctx_sampling, ctx, NULL, NULL);

        // is it an end of stream?
        if (output_token == llama_token_eos(model)) {
            break;
        }

        // prepare the next batch
        llama_batch_clear(batch);

        // push this new token for next evaluation
        llama_batch_add(batch, output_token, n_past, { 0 }, true);
        
        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            log_output("Decode failed in main loop");
            return 1;
        }

        n_past++;
    }

    log_output(("Prompt force stopped in " + get_elapsed_seconds(std::chrono::high_resolution_clock::now() - prompt_start_time)).c_str());
    output("", true);
    return 0;
}

EXPORT void maid_llm_stop(void) {
    stop_generation.store(true);
}

EXPORT void maid_llm_cleanup(void) {
    stop_generation.store(true);
    llama_free(ctx);
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