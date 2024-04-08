#include "utils.hpp"

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
#include <algorithm>
#include <unordered_set>
#include <thread>
#include <atomic>
#include <mutex>

static std::atomic_bool stop_generation(false);
static std::mutex continue_mutex;

static llama_model * model;
static gpt_params params;

static llama_context * ctx;

static std::vector<std::vector<llama_token>> terminator_sequences;

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
        terminator_sequences.push_back(llama_tokenize(model, "### Instruction:", false, true));
        terminator_sequences.push_back(llama_tokenize(model, "### Response:", false, true));
        terminator_sequences.push_back(llama_tokenize(model, "### System:", false, true));
    }

    if (params.chatml) {
        terminator_sequences.push_back(llama_tokenize(model, "<|im_start|>", false, true));
        terminator_sequences.push_back(llama_tokenize(model, "<|im_end|>", false, true));
    }

    terminator_sequences.push_back(llama_tokenize(model, "\n\n\n\n\n", false, true));

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

    std::vector<llama_token> input_tokens = parse_messages(msg_count, messages, ctx, model, params);

    std::vector<llama_token> cache_tokens;

    int n_past = 0;
    int n_ctx = llama_n_ctx(ctx);
    int n_predict = params.n_predict;

    if (n_predict <= 0 || n_predict > n_ctx) {
        n_predict = n_ctx;
    }

    int terminator_max = 0;

    for (auto &terminator : terminator_sequences) {
        terminator_max = std::max(terminator_max, (int) terminator.size() + 1);
    }

    log_output(("n_ctx: " + std::to_string(n_ctx)).c_str());
    log_output(("n_predict: " + std::to_string(n_predict)).c_str());

    //Truncate the prompt if it's too long
    if ((int) input_tokens.size() >= n_ctx - n_predict) {
        // truncate the input
        input_tokens.erase(input_tokens.begin(), input_tokens.begin() + (input_tokens.size() - n_ctx) + n_predict);

        // log the truncation
        log_output(("input_tokens was truncated: " + LOG_TOKENS_TOSTR_PRETTY(ctx, input_tokens)).c_str());
    }
    
    // Should not run without any tokens
    if (input_tokens.empty()) {
        input_tokens.push_back(llama_token_bos(model));
        log_output(("input_tokens was considered empty and bos was added: " + LOG_TOKENS_TOSTR_PRETTY(ctx, input_tokens)).c_str());
    }

    eval_tokens(ctx, input_tokens, params.n_batch, &n_past);

    while (!stop_generation.load()) {
        // sample the most likely token
        llama_token id = llama_sampling_sample(ctx_sampling, ctx, NULL, NULL);

        // accept the token
        llama_sampling_accept(ctx_sampling, ctx, id, true);

        // is it an end of stream?
        if (id == llama_token_eos(model)) {
            output(tokens_to_string(ctx, cache_tokens).c_str(), false);
            log_output("Breaking due to eos");
            break;
        }

        // Add token to cache
        cache_tokens.push_back(id);

        if (cache_tokens.size() > terminator_max) {
            if (search_terminators(terminator_sequences, &cache_tokens)) {
                output(tokens_to_string(ctx, cache_tokens).c_str(), false);
                log_output("Breaking due to terminator");
                break;
            }

            // Pop the first token from the cache and send it to the output
            output(llama_token_to_piece(ctx, cache_tokens[0]).c_str(), false);
            cache_tokens.erase(cache_tokens.begin());
        }

        // evaluate the token
        if (!eval_id(ctx, id, &n_past)) {
            output(tokens_to_string(ctx, cache_tokens).c_str(), false);
            log_output("Breaking due to eval_id");
            break;
        }
    }

    log_output(("Prompt stopped in " + get_elapsed_seconds(std::chrono::high_resolution_clock::now() - prompt_start_time)).c_str());
    stop_generation.store(false);
    llama_free(ctx);
    llama_sampling_free(ctx_sampling);
    output("", true);
    return 0;
}

EXPORT void maid_llm_stop(void) {
    stop_generation.store(true);
}

EXPORT void maid_llm_cleanup(void) {
    stop_generation.store(true);
    llama_free_model(model);
    llama_backend_free();
}