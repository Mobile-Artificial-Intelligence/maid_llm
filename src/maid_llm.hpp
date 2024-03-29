#include "maid_llm.h"
#include "llama.h"
#include "ggml.h"
#include "common.h"

#include <chrono>

static llama_sampling_params from_c_sampling_params(struct sampling_params c_params);

static gpt_params from_c_params(struct gpt_c_params c_params);

std::vector<llama_token> parse_messages(maid_llm_chat *chat);

std::string get_elapsed_seconds(const std::chrono::nanoseconds &__d);