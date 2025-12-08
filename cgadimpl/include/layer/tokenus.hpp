#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <regex>
#include <omp.h>
#include <cassert>
#include <algorithm>
#include <iostream>
#include "layer/affine.hpp"

namespace ag::layer {

// -------------------------------------------------
//               BASIC TYPES
// -------------------------------------------------
using Pair = std::pair<uint32_t,uint32_t>;

// Hash for Pair
struct PairHash {
    size_t operator()(Pair const& p) const noexcept {
        return ((size_t)p.first << 32) ^ (size_t)p.second;
    }
};

// -------------------------------------------------
//               WORD STRUCT
// -------------------------------------------------
struct Word {
    std::vector<uint32_t> ids;

    Word() {}
    Word(const std::vector<uint32_t>& v) : ids(v) {}

    std::vector<Pair> pairs() const;
    std::vector<std::pair<Pair,int32_t>> merge_pair(const Pair& p, uint32_t new_id);
};

// -------------------------------------------------
//               TOKENIZER
// -------------------------------------------------
class Tokenizer {
public:
    std::unordered_map<Pair,uint32_t,PairHash> merges;
    std::string pattern;
    std::regex compiled;

    Tokenizer();  // builds regex

    // training
    void train(const std::vector<std::string>& data, uint32_t vocab_size);

    // encoding
    std::vector<uint32_t> encode(const std::string& text) const;

private:
    using CountMap = std::unordered_map<Pair,int32_t,PairHash>;
    using PosMap   = std::unordered_map<Pair,std::unordered_set<size_t>,PairHash>;

    std::pair<CountMap,PosMap>
    count_pairs(const std::vector<Word>& words,
                const std::vector<int32_t>& counts);

    void train_core(std::vector<Word>& words,
                    std::vector<int32_t>& counts,
                    uint32_t vocab_size);
};

} // namespace ag::layer
