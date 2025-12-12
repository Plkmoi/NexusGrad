#include <fstream>
#include <vector>
#include <string>
#include <iostream>
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