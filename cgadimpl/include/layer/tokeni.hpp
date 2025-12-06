#ifndef TOKENI_HPP
#define TOKENI_HPP

#include <fstream>
#include <vector>
#include <string>
#include <iostream>

// This is the ONLY FFI header we use.
#include "tokenize/exptok.hpp"

namespace ag::layer {

// ----------------------------------------------------
// Read entire file into std::string
// ----------------------------------------------------
std::string load_text_file(const std::string& path);

// ----------------------------------------------------
// Byte-level tokenize: each char → unsigned value 0..255
// ----------------------------------------------------
std::vector<int> tokenize_bytes(const std::string& text);

// ----------------------------------------------------
// C++ Wrapper Class for the Rust Tokenizer
// ----------------------------------------------------
class TokeniTokenizer {
public:
    TokeniTokenizer(const std::string& model_path);

    ~TokeniTokenizer();

    void train(const std::vector<std::string>& texts,
               uint32_t vocab_size);

    std::vector<int> encode(const std::string& input);

private:
    TokenusModel model = nullptr;
};


} // namespace ag::layer

#endif // TOKENI_HPP
