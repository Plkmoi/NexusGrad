#pragma once
#include <cstdint>
#include <cstddef>

extern "C" {

// Opaque Rust model handle.
typedef void* TokenusModel;

// Load tokenizer from file.
TokenusModel tokenus_new(const char* path);

// Free tokenizer model.
void tokenus_free(TokenusModel model);

// Train tokenizer (your Rust implementation remains unchanged).
void tokenus_train(TokenusModel model,
                   const char* const* texts,
                   size_t text_count,
                   uint32_t vocab_size);

// Encode string → token IDs.
uint32_t* tokenus_encode(TokenusModel model,
                         const char* text,
                         size_t* out_len);

// Free returned token buffer.
void tokenus_free_ids(uint32_t* ptr);

} // extern "C"
