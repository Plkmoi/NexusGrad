#include <fstream>
#include <vector>
#include <string>
#include <iostream>

#include "layer/tokeni.hpp"
#include "layer/tokenize/exptok.hpp"

namespace ag::layer {

// ----------------------------------------------------
// Read entire file into std::string
// ----------------------------------------------------
std::string load_text_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Error: cannot open file " + path);
    }

    std::string data;
    in.seekg(0, std::ios::end);
    size_t size = in.tellg();
    in.seekg(0);

    data.resize(size);
    in.read(&data[0], size);

    return data;
}

// ----------------------------------------------------
// Byte-level tokenize: each char → unsigned value 0..255
// ----------------------------------------------------
std::vector<int> tokenize_bytes(const std::string& text) {
    std::vector<int> tokens;
    tokens.reserve(text.size());

    for (unsigned char c : text) {
        if (c >= 32 && c <= 126)
            tokens.push_back((int)c);
        else
            tokens.push_back((int)' ');
    }

    return tokens;
}


// ----------------------------------------------------
// TokeniTokenizer — Train
// ----------------------------------------------------
TokeniTokenizer::TokeniTokenizer(const std::string& model_path) {
    model = tokenus_new(model_path.c_str());
    if (!model)
        throw std::runtime_error("Failed to load tokenus model");
}

TokeniTokenizer::~TokeniTokenizer() {
    if (model)
        tokenus_free(model);
}

// -----------------------------------------------------
// Train through Rust FFI
// -----------------------------------------------------
void TokeniTokenizer::train(const std::vector<std::string>& texts,
                            uint32_t vocab_size)
{
    std::vector<const char*> raw;
    raw.reserve(texts.size());

    for (auto& s : texts)
        raw.push_back(s.c_str());

    tokenus_train(model, raw.data(), raw.size(), vocab_size);
}

// -----------------------------------------------------
// Encode via Rust tokenizer
// -----------------------------------------------------
std::vector<int> TokeniTokenizer::encode(const std::string& input)
{
    size_t out_len = 0;

    uint32_t* raw = tokenus_encode(model, input.c_str(), &out_len);
    if (!raw)
        throw std::runtime_error("tokenus_encode() failed");

    std::vector<int> result;
    result.reserve(out_len);

    for (size_t i = 0; i < out_len; ++i)
        result.push_back(static_cast<int>(raw[i]));

    tokenus_free(raw);

    return result;
}


} // namespace ag::layer
