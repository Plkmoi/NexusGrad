#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <map>
#include "layer/archlist.hpp"

namespace flow {
// ----------------------------------------------------
// Read entire file into std::string
// ----------------------------------------------------
std::string load_text_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);   // binary to keep raw bytes
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


// 2025 "Naked" CSV Loader for 1.9GB Datasets
std::vector<int> load_and_encode_csv(const std::string& path, flow::Tokenizer& tok) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("CSV not found lol");

    std::vector<int> all_tokens;
    all_tokens.reserve(500000000); // 500M slots is the target! 🎯

    std::string story;
    char ch;
    bool in_quotes = false;
    int last_mil = 0;

    std::cout << "Starting the REAL 1.9GB CSV Feast... 🥩⛽" << std::endl;

    while (file.get(ch)) {
        if (ch == '\"') {
            in_quotes = !in_quotes; // Toggle quote state 👗✨
            continue;
        }

        if (in_quotes) {
            story += ch; // Keep adding everything, even newlines! ✅
        } else if (ch == '\n' && !story.empty()) {
            // We reached the actual end of a CSV row! 🏁
            std::vector<uint32_t> ids = tok.encode(story);
            for (auto id : ids) all_tokens.push_back((int)id);
            story.clear();

            // "Brrr" Progress Update 🏎️💨
            int current_mil = all_tokens.size() / 1000000;
            if (current_mil > last_mil) {
                std::cout << "Tokens encoded: " << current_mil << " Million... 🏎️💨" << std::endl;
                last_mil = current_mil;
            }
        }
    }
    std::cout << "Feast complete! Total: " << all_tokens.size() << " 🏆✨" << std::endl;
    return all_tokens;
}

// 2025 "Turbo C++" Partial Loader lol
std::string load_text_file_partial(const std::string& path, size_t bytes_to_read) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open file lol");

    // 1. --- The "Silly" Check ---
    // If the file is smaller than our 50MB limit, just read the whole thing lol
    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    size_t actual_read = std::min(file_size, bytes_to_read);

    // 2. --- The "Naked" Resize ---
    std::string data;
    data.resize(actual_read);

    // 3. --- The "Brrr" Read ---
    in.read(&data[0], actual_read);

    std::cout << "Read " << actual_read / (1024 * 1024) << "MB for training! 🧠✅" << std::endl;
    return data;
}



std::string shape_to_json(const std::vector<int64_t> dims) {
    std::string s = "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        s += std::to_string(dims[i]) + (i == dims.size() - 1 ? "" : ", ");
    }
    return s + "]";
}


void save_safetensors(const std::string& filename, const std::map<std::string, Tensor>& weights) {
    // 1. Build the JSON header string
    std::string header = "{";
    size_t current_offset = 0;
    
    for (auto const& [name, tensor] : weights) {
        size_t byte_size = tensor.numel() * sizeof(float);
        header += "\"" + name + "\": {";
        header += "\"dtype\": \"F32\", ";
        header += "\"shape\": " + shape_to_json(tensor.shape().dims) + ", ";
        header += "\"data_offsets\": [" + std::to_string(current_offset) + ", " 
               + std::to_string(current_offset + byte_size) + "]},";
        current_offset += byte_size;
    }
    header.pop_back(); // Remove trailing comma
    header += "}";

    // 2. Write to File
    std::ofstream file(filename, std::ios::binary);
    
    // Write Header Length (8 bytes, Little Endian)
    u_int64_t header_len = header.length();
    file.write(reinterpret_cast<const char*>(&header_len), 8);
    
    // Write JSON Header
    file.write(header.c_str(), header_len);
    
    // Write Raw Tensor Data
    for (auto const& [name, tensor] : weights) {
        file.write(reinterpret_cast<const char*>(tensor.data<float>()), 
                   tensor.numel() * sizeof(float));
    }
    file.close();
}


// ----------------------------------------------------
// Byte-level tokenize: each char → unsigned value 0..255
// ----------------------------------------------------
std::vector<int> tokenize_bytes( const std::string& text ) {
    std::vector<int> tokens;
    tokens.reserve(text.size());

    for (unsigned char c : text) {
        if (c >= 32 && c <= 126)
            tokens.push_back((int)c);      // keep visible ASCII bytes
        else
            tokens.push_back((int)' ');    // replace all non-ascii with space
    }

    return tokens;
}
// ----------------------------------------------------
// Example usage
// ----------------------------------------------------
// int main() {
//     try {
//         std::string path = "corpus.txt";     // change to your file
//         std::string text = load_text_file(path);

//         std::vector<int> tokens = tokenize_bytes(text);

//         std::cout << "Loaded " << text.size() << " bytes.\n";
//         std::cout << "Tokenized into " << tokens.size() << " tokens.\n";

//         std::cout << "First 20 tokens: ";
//         for (int i = 0; i < 20 && i < (int)tokens.size(); i++)
//             std::cout << tokens[i] << " ";
//         std::cout << "\n";

//     } catch (const std::exception& e) {
//         std::cerr << e.what() << "\n";
//     }

//     return 0;
// }
}