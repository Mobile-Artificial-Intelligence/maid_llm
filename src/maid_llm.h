#ifndef _MAID_LLM_H
#define _MAID_LLM_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef WIN32
   #define EXPORT __declspec(dllexport)
#else
   #define EXPORT __attribute__((visibility("default"))) __attribute__((used))
#endif

#include <stdbool.h>

enum chat_role {
    ROLE_SYSTEM,
    ROLE_USER,
    ROLE_ASSISTANT
};

// sampling parameters
struct sampling_params {
    int         n_prev;                             // number of previous tokens to remember
    int         n_probs;                            // if greater than 0, output the probabilities of top n_probs tokens.
    int         min_keep;                           // 0 = disabled, otherwise samplers should return at least min_keep tokens
    int         top_k;                              // <= 0 to use vocab size
    float       top_p;                              // 1.0 = disabled
    float       min_p;                              // 0.0 = disabled
    float       tfs_z;                              // 1.0 = disabled
    float       typical_p;                          // 1.0 = disabled
    float       temp;                               // <= 0.0 to sample greedily, 0.0 to not output probabilities
    float       dynatemp_range;                     // 0.0 = disabled
    float       dynatemp_exponent;                  // controls how entropy maps to temperature in dynamic temperature sampler
    int         penalty_last_n;                     // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float       penalty_repeat;                     // 1.0 = disabled
    float       penalty_freq;                       // 0.0 = disabled
    float       penalty_present;                    // 0.0 = disabled
    int         mirostat;                           // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float       mirostat_tau;                       // target entropy
    float       mirostat_eta;                       // learning rate
    bool        penalize_nl;                        // consider newlines as a repeatable token
    char*       grammar;                            // optional BNF-like grammar to constrain sampling
    char*       cfg_negative_prompt;                // string to help guidance
    float       cfg_scale;                          // how strong is guidance
};

// llama.cpp gpt_c_params
struct gpt_c_params {
    signed int seed;                                // RNG seed
    int n_threads;              
    int n_threads_draft;                
    int n_threads_batch;                            // number of threads to use for batch processing (-1 = use n_threads)
    int n_threads_batch_draft;                  
    int n_predict;                                  // new tokens to predict
    int n_ctx;                                      // context size
    int n_batch;                                    // batch size for prompt processing (must be >=32 to use BLAS)
    int n_keep;                                     // number of tokens to keep from initial prompt
    int n_draft;                                    // number of tokens to draft during speculative decoding
    int n_chunks;                                   // max number of chunks to process (-1 = unlimited)
    int n_parallel;                                 // number of parallel sequences to decode
    int n_sequences;                                // number of sequences to decode
    float p_split;                                  // speculative decoding split probability
    int n_gpu_layers;                               // number of layers to store in VRAM (-1 - use default)
    int n_gpu_layers_draft;                         // number of layers to store in VRAM for the draft model (-1 - use default)
    char split_mode;                                // how to split the model across GPUs
    int main_gpu;                                   // the GPU that is used for scratch and small tensors
    int n_beams;                                    // if non-zero then use beam search of given width.
    int grp_attn_n;                                 // group-attention factor
    int grp_attn_w;                                 // group-attention width
    int n_print;                                    // print token count every n tokens (-1 = disabled)
    float rope_freq_base;                           // RoPE base frequency
    float rope_freq_scale;                          // RoPE frequency scaling factor
    float yarn_ext_factor;                          // YaRN extrapolation mix factor
    float yarn_attn_factor;                         // YaRN magnitude scaling factor
    float yarn_beta_fast;                           // YaRN low correction dim
    float yarn_beta_slow;                           // YaRN high correction dim
    int yarn_orig_ctx;                              // YaRN original context length
    float defrag_thold;                             // KV cache defragmentation threshold
    int rope_scaling_type;
    char numa;                                      // NUMA policy for memory allocation

    struct sampling_params sparams;                 // sampling parameters

    char *model;                                    // model path
    char *model_draft;                              // draft model for speculative decoding
    char *model_alias;                              // model alias
    char *prompt;   
    char *prompt_file;                              // store the external prompt file name
    char *path_prompt_cache;                        // path to file for saving/loading prompt eval state
    char *input_prefix;                             // string to prefix user inputs with
    char *input_suffix;                             // string to suffix user inputs with
    char *antiprompt;                               // string upon seeing which more user input is prompted
    char *logdir;                                   // directory in which to save YAML log files
    char *logits_file;                              // file for saving *all* logits
    char *lora_base;                                // base model path for the lora adapter

    int  ppl_stride;                                // stride for perplexity calculations. If left at 0, the pre-existing approach will be used.
    int  ppl_output_type;                           // = 0 -> ppl output is as usual, = 1 -> ppl output is num_tokens, ppl, one per line

    bool   hellaswag;                               // compute HellaSwag score over random tasks from datafile supplied in prompt
    unsigned long hellaswag_tasks;                  // number of tasks to use when computing the HellaSwag score

    bool   winogrande;                              // compute Winogrande score over random tasks from datafile supplied in prompt
    unsigned long winogrande_tasks;                 // number of tasks to use when computing the Winogrande score. If 0, all tasks will be computed

    bool   multiple_choice;                         // compute TruthfulQA score over random tasks from datafile supplied in prompt
    unsigned long multiple_choice_tasks;            // number of tasks to use when computing the TruthfulQA score. If 0, all tasks will be computed

    bool   kl_divergence;                           // compute KL-divergence

    bool random_prompt;                             // do not randomize prompt if none provided
    bool use_color;                                 // use color to distinguish generations and inputs
    bool interactive;                               // interactive mode
    bool chatml;                                    // chatml mode (used for models trained on chatml syntax)
    bool prompt_cache_all;                          // save user input and generations to prompt cache
    bool prompt_cache_ro;                           // open the prompt cache read-only and do not update it

    bool embedding;                                 // get only sentence embedding
    bool escape;                                    // escape "\n", "\r", "\t", "\'", "\"", and "\\"
    bool interactive_first;                         // wait for user input immediately
    bool multiline_input;                           // reverse the usage of `\`
    bool simple_io;                                 // improves compatibility with subprocesses and limited consoles
    bool cont_batching;                             // insert new sequences for decoding on-the-fly

    bool input_prefix_bos;                          // prefix BOS to user inputs, preceding input_prefix
    bool ignore_eos;                                // ignore generated EOS tokens
    bool instruct;                                  // instruction mode (used for Alpaca models)
    bool logits_all;                                // return logits for all tokens in the batch
    bool use_mmap;                                  // use mmap for faster loads
    bool use_mlock;                                 // use mlock to keep model in memory
    bool verbose_prompt;                            // print prompt tokens before generation
    bool display_prompt;                            // print prompt before generation
    bool infill;                                    // use infill mode
    bool dump_kv_cache;                             // dump the KV cache contents for debugging purposes
    bool no_kv_offload;                             // disable KV offloading

    char *cache_type_k;                             // KV cache data type for the K
    char *cache_type_v;                             // KV cache data type for the V

    // multimodal models (see examples/llava)
    char *mmproj;                                   // path to multimodal projector
    char *image;                                    // path to an image file
};

struct chat_message {
    enum chat_role role;
    char *content;
};

typedef void dart_logger(const char *buffer);

typedef void dart_output(const char *buffer, bool stop);

EXPORT int maid_llm_model_init(struct gpt_c_params *c_params, dart_logger *log_output);

EXPORT int maid_llm_context_init(struct gpt_c_params *c_params, dart_logger *log_output);

EXPORT int maid_llm_prompt(int msg_count, struct chat_message* messages[], dart_output *output, dart_logger *log_output);

EXPORT void maid_llm_stop(void);

EXPORT void maid_llm_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif