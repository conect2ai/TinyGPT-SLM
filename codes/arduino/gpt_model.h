// Automatically generated file - gpt_model.h
#ifndef GPT_MODEL_H
#define GPT_MODEL_H

#include <Arduino.h>
#include <math.h>
#include <string.h>

// Architecture detection
#if defined(__AVR__)
  #include <avr/pgmspace.h>
  #define MODEL_DATA_STORAGE PROGMEM
  #define READ_MODEL_FLOAT(x) pgm_read_float_near(&x)
#else
  #define MODEL_DATA_STORAGE
  #define READ_MODEL_FLOAT(x) x
#endif

// Model Settings
    #define VOCAB_SIZE 445
    #define EMB_DIM 16
    #define CONTEXT_LENGTH 22
    #define N_LAYERS 4
    #define N_HEADS 8
    #define HEAD_DIM (EMB_DIM / N_HEADS)

    // Special Tokens
    #define SOS_TOKEN_ID 0
    #define EOS_TOKEN_ID 1
    #define PAD_TOKEN_ID 2
    #define UNK_TOKEN_ID 3
    #define SEP_TOKEN_ID 4

    // Constants for buffers
    #define MAX_TOKEN_STRING_LENGTH 16
    #define MAX_PROMPT_TOKENS_BUFFER_SIZE 44
    #define MAX_GENERATED_TEXT_LENGTH (CONTEXT_LENGTH * 5 + 100)
    


extern const float MODEL_DATA_STORAGE tok_emb_weight[VOCAB_SIZE][EMB_DIM];
extern const float MODEL_DATA_STORAGE pos_emb_weight[CONTEXT_LENGTH][EMB_DIM];
#if N_LAYERS > 0
extern const float MODEL_DATA_STORAGE trf_att_W_query_weight[N_LAYERS][EMB_DIM][EMB_DIM];
extern const float MODEL_DATA_STORAGE trf_att_W_key_weight[N_LAYERS][EMB_DIM][EMB_DIM];
extern const float MODEL_DATA_STORAGE trf_att_W_value_weight[N_LAYERS][EMB_DIM][EMB_DIM];
extern const float MODEL_DATA_STORAGE trf_att_out_proj_weight[N_LAYERS][EMB_DIM][EMB_DIM];
extern const float MODEL_DATA_STORAGE trf_att_out_proj_bias[N_LAYERS][EMB_DIM];
extern const float MODEL_DATA_STORAGE trf_ff_lin1_weight[N_LAYERS][4*EMB_DIM][EMB_DIM];
extern const float MODEL_DATA_STORAGE trf_ff_lin1_bias[N_LAYERS][4*EMB_DIM];
extern const float MODEL_DATA_STORAGE trf_ff_lin2_weight[N_LAYERS][EMB_DIM][4*EMB_DIM];
extern const float MODEL_DATA_STORAGE trf_ff_lin2_bias[N_LAYERS][EMB_DIM];
extern const float MODEL_DATA_STORAGE trf_norm1_scale[N_LAYERS][EMB_DIM];
extern const float MODEL_DATA_STORAGE trf_norm1_shift[N_LAYERS][EMB_DIM];
extern const float MODEL_DATA_STORAGE trf_norm2_scale[N_LAYERS][EMB_DIM];
extern const float MODEL_DATA_STORAGE trf_norm2_shift[N_LAYERS][EMB_DIM];
#endif
extern const float MODEL_DATA_STORAGE final_norm_scale[EMB_DIM];
extern const float MODEL_DATA_STORAGE final_norm_shift[EMB_DIM];
extern const float MODEL_DATA_STORAGE out_head_weight[VOCAB_SIZE][EMB_DIM];



// Buffers globais
extern float g_static_attn_out_buffer[CONTEXT_LENGTH * EMB_DIM];
extern float g_static_norm_out_buffer[CONTEXT_LENGTH * EMB_DIM];
extern float g_static_ff_out_buffer[CONTEXT_LENGTH * EMB_DIM];
extern float g_static_lin1_out_buffer[4 * EMB_DIM];
extern float g_static_q_buffer[EMB_DIM];
extern float g_static_k_buffer[EMB_DIM];
extern float g_static_v_buffer[EMB_DIM];
extern float g_static_attn_scores_buffer[CONTEXT_LENGTH];
extern float g_static_attention_results_buffer[CONTEXT_LENGTH * EMB_DIM];
extern float g_static_gpt_forward_x_buffer[CONTEXT_LENGTH * EMB_DIM];
extern float g_static_gpt_forward_norm_out_buffer[CONTEXT_LENGTH * EMB_DIM];
extern float g_static_logits_buffer[VOCAB_SIZE];
extern int g_static_tokens_buffer[CONTEXT_LENGTH]; 
extern int g_static_prompt_tokens_temp_buffer[MAX_PROMPT_TOKENS_BUFFER_SIZE];



// Function Declarations
void gpt_forward(float* logits_output, int* input_ids, int seq_len);
int generate_next_token(int* current_sequence_token_ids, int current_seq_len);
void generate_qna_answer(char* output_buffer, int buffer_size, const String& question_prompt, int max_new_tokens = 20);

// Tokenization Functions
char _to_lower_char(char c);
bool _is_word_char(char c);
int find_token_id(const char* word_to_find);
int tokenize_prompt(const String& prompt_str, int* out_token_ids, int max_tokens_to_fill);
const char* decode_token_str(int token_id);


#endif // GPT_MODEL_H
