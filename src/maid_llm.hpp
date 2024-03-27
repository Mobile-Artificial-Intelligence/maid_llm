#include "maid_llm.h"
#include "llama.h"
#include "ggml.h"
#include "common.h"

static llama_sampling_params from_c_sampling_params(struct sampling_params c_params);

static gpt_params from_c_params(struct gpt_c_params c_params);

void parse_messages(int msg_count, chat_message* messages[]);

static void dart_log_callback(ggml_log_level level, const char * text, void * user_data);