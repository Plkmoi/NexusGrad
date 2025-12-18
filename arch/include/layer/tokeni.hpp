#include <fstream>
#include <vector>
#include <string>
#include <iostream>


namespace flow {
// ----------------------------------------------------
// Read entire file into std::string
// ----------------------------------------------------
std::string load_text_file(const std::string& path) ;

// ----------------------------------------------------
// Byte-level tokenize: each char → unsigned value 0..255
// ----------------------------------------------------
std::vector<int> tokenize_bytes(const std::string& text) ;

std::string shape_to_json(const std::vector<int64_t> dims);

void save_safetensors(const std::string& filename, const std::map<std::string, Tensor>& weights);

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