#ifndef _MAID_LLM_H
#define _MAID_LLM_H

#include <stdbool.h>

struct gpt_c_params {
    signed int seed                 = -1;       // RNG seed

    int n_threads                   = 8;
    int n_threads_draft             = -1;
    int n_threads_batch             = -1;       // number of threads to use for batch processing (-1 = use n_threads)
    int n_threads_batch_draft       = -1;
    int n_predict                   = -1;       // new tokens to predict
    int n_ctx                       = 512;      // context size
    int n_batch                     = 512;      // batch size for prompt processing (must be >=32 to use BLAS)
    int n_keep                      = 0;        // number of tokens to keep from initial prompt
    int n_draft                     = 8;        // number of tokens to draft during speculative decoding
    int n_chunks                    = -1;       // max number of chunks to process (-1 = unlimited)
    int n_parallel                  = 1;        // number of parallel sequences to decode
    int n_sequences                 = 1;        // number of sequences to decode
    float p_accept                  = 0.5f;     // speculative decoding accept probability
    float p_split                   = 0.1f;     // speculative decoding split probability
    int n_gpu_layers                = -1;       // number of layers to store in VRAM (-1 - use default)
    int n_gpu_layers_draft          = -1;       // number of layers to store in VRAM for the draft model (-1 - use default)
    char split_mode                 = 1;        // how to split the model across GPUs
    int main_gpu                    = 0;        // the GPU that is used for scratch and small tensors
    float tensor_split[128]         = {0};      // how split tensors should be distributed across GPUs
    int n_beams                     = 0;        // if non-zero then use beam search of given width.
    int grp_attn_n                  = 1;        // group-attention factor
    int grp_attn_w                  = 512;      // group-attention width
    int n_print                     = -1;       // print token count every n tokens (-1 = disabled)
    float rope_freq_base            = 0.0f;     // RoPE base frequency
    float rope_freq_scale           = 0.0f;     // RoPE frequency scaling factor
    float yarn_ext_factor           = -1.0f;    // YaRN extrapolation mix factor
    float yarn_attn_factor          = 1.0f;     // YaRN magnitude scaling factor
    float yarn_beta_fast            = 32.0f;    // YaRN low correction dim
    float yarn_beta_slow            = 1.0f;     // YaRN high correction dim
    int yarn_orig_ctx               = 0;        // YaRN original context length
    float defrag_thold              = -1.0f;    // KV cache defragmentation threshold
    int rope_scaling_type           = -1;
    char numa                       = 0;

    char *model;                                // model path
    char *model_draft;                          // draft model for speculative decoding
    char *model_alias;                          // model alias
    char *prompt;
    char *prompt_file;                          // store the external prompt file name
    char *path_prompt_cache;                    // path to file for saving/loading prompt eval state
    char *input_prefix;                         // string to prefix user inputs with
    char *input_suffix;                         // string to suffix user inputs with
    char *antiprompt;                           // string upon seeing which more user input is prompted
    char *logdir;                               // directory in which to save YAML log files
    char *logits_file;                          // file for saving *all* logits
    char *lora_base;                            // base model path for the lora adapter

    int  ppl_stride                 = 0;        // stride for perplexity calculations. If left at 0, the pre-existing approach will be used.
    int  ppl_output_type            = 0;        // = 0 -> ppl output is as usual, = 1 -> ppl output is num_tokens, ppl, one per line

    bool   hellaswag                = false;    // compute HellaSwag score over random tasks from datafile supplied in prompt
    unsigned long hellaswag_tasks   = 400;      // number of tasks to use when computing the HellaSwag score

    bool   winogrande               = false;    // compute Winogrande score over random tasks from datafile supplied in prompt
    unsigned long winogrande_tasks  = 0;        // number of tasks to use when computing the Winogrande score. If 0, all tasks will be computed

    bool   multiple_choice = false;             // compute TruthfulQA score over random tasks from datafile supplied in prompt
    unsigned long multiple_choice_tasks = 0;    // number of tasks to use when computing the TruthfulQA score. If 0, all tasks will be computed

    bool   kl_divergence            = false;    // compute KL-divergence

    bool mul_mat_q                  = true;     // if true, use mul_mat_q kernels instead of cuBLAS
    bool random_prompt              = false;    // do not randomize prompt if none provided
    bool use_color                  = false;    // use color to distinguish generations and inputs
    bool interactive                = false;    // interactive mode
    bool chatml                     = false;    // chatml mode (used for models trained on chatml syntax)
    bool prompt_cache_all           = false;    // save user input and generations to prompt cache
    bool prompt_cache_ro            = false;    // open the prompt cache read-only and do not update it

    bool embedding                  = false;    // get only sentence embedding
    bool escape                     = false;    // escape "\n", "\r", "\t", "\'", "\"", and "\\"
    bool interactive_first          = false;    // wait for user input immediately
    bool multiline_input            = false;    // reverse the usage of `\`
    bool simple_io                  = false;    // improves compatibility with subprocesses and limited consoles
    bool cont_batching              = false;    // insert new sequences for decoding on-the-fly

    bool input_prefix_bos           = false;    // prefix BOS to user inputs, preceding input_prefix
    bool ignore_eos                 = false;    // ignore generated EOS tokens
    bool instruct                   = false;    // instruction mode (used for Alpaca models)
    bool logits_all                 = false;    // return logits for all tokens in the batch
    bool use_mmap                   = true;     // use mmap for faster loads
    bool use_mlock                  = false;    // use mlock to keep model in memory
    bool verbose_prompt             = false;    // print prompt tokens before generation
    bool display_prompt             = true;     // print prompt before generation
    bool infill                     = false;    // use infill mode
    bool dump_kv_cache              = false;    // dump the KV cache contents for debugging purposes
    bool no_kv_offload              = false;    // disable KV offloading

    char *cache_type_k;                         // KV cache data type for the K
    char *cache_type_v;                         // KV cache data type for the V

    // multimodal models (see examples/llava)
    char *mmproj;                               // path to multimodal projector
    char *image;                                // path to an image file
};

typedef void dart_logger(const char *buffer);

typedef void dart_output(const char *buffer, bool stop);

int maid_llm_init(struct gpt_c_params *c_params, dart_logger *log_output);

int maid_llm_prompt(const char *input, dart_output *output);

void maid_llm_stop(void);

void maid_llm_cleanup(void);

#endif