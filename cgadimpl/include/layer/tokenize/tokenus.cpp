#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <regex>
#include <string>
#include <algorithm>
#include <cassert>

// Default GPT-4 style regex pattern for splitting text
const std::string GPT4_PATTERN = R"('(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+)";

using Pair = std::pair<uint32_t, uint32_t>;

struct Word {
    std::vector<uint32_t> ids;

    Word() = default;
    Word(const std::vector<uint32_t>& ids_) : ids(ids_) {}

    std::vector<Pair> pairs() const {
        std::vector<Pair> result;
        for (size_t i = 0; i + 1 < ids.size(); ++i) {
            result.emplace_back(ids[i], ids[i + 1]);
        }
        return result;
    }

    // Merge all non-overlapping occurrences of pair -> new_id.
    // Returns a vector of local pair-count deltas for THIS word only.
    std::vector<std::pair<Pair, int32_t>> merge_pair(Pair pair, uint32_t new_id) {
        uint32_t a = pair.first, b = pair.second;
        size_t n = ids.size();
        if (n < 2) return {};

        std::vector<uint32_t> out;
        out.reserve(n);
        std::vector<std::pair<Pair, int32_t>> deltas;
        deltas.reserve(6);

        size_t i = 0;
        while (i < n) {
            if (i + 1 < n && ids[i] == a && ids[i + 1] == b) {
                uint32_t left = out.empty() ? UINT32_MAX : out.back();
                uint32_t right = (i + 2 < n) ? ids[i + 2] : UINT32_MAX;

                // remove old pairs
                if (left != UINT32_MAX) {
                    deltas.emplace_back(Pair(left, a), -1);
                    deltas.emplace_back(Pair(left, new_id), 1);
                }
                deltas.emplace_back(Pair(a, b), -1);
                if (right != UINT32_MAX) {
                    deltas.emplace_back(Pair(b, right), -1);
                    deltas.emplace_back(Pair(new_id, right), 1);
                }

                out.push_back(new_id);
                i += 2;
            } else {
                out.push_back(ids[i]);
                i += 1;
            }
        }
        ids = std::move(out);
        return deltas;
    }
};

struct MergeJob {
    Pair pair;
    uint64_t count;
    std::unordered_set<size_t> pos;

    // For max-heap: higher count first, then ascending pair order
    bool operator<(const MergeJob& other) const {
        if (count != other.count) return count < other.count;
        return pair > other.pair; // ascending order on pair when counts tie
    }
};

class Tokenizer {
public:
    std::unordered_map<Pair, uint32_t> merges;
    std::string pattern;
    std::regex compiled_pattern;

    Tokenizer() : pattern(GPT4_PATTERN), compiled_pattern(GPT4_PATTERN, std::regex::optimize | std::regex::ECMAScript) {}

    // Core incremental BPE training given unique words and their counts.
    void train_core_incremental(std::vector<Word>& words, const std::vector<int32_t>& counts, uint32_t vocab_size) {
        assert(vocab_size >= 256);
        uint32_t num_merges = vocab_size - 256;
        std::cout << "Starting BPE training: " << num_merges << " merges to compute\n";
        merges.clear();

        // ---- Initial pair_counts and where_to_update ----
        std::cout << "Computing initial pair counts from " << words.size() << " unique sequences\n";
        auto [pair_counts, where_to_update] = count_pairs(words, counts);

        // ---- Build heap ----
        std::cout << "Building heap with " << pair_counts.size() << " unique pairs\n";
        std::priority_queue<MergeJob> heap;
        for (auto& [pair, pos] : where_to_update) {
            int c = pair_counts[pair];
            if (c > 0) {
                heap.push(MergeJob{pair, static_cast<uint64_t>(c), std::move(pos)});
            }
        }

        // ---- Merge loop ----
        std::cout << "Starting merge loop\n";
        uint32_t merges_done = 0;
        uint32_t last_log_percent = 0;

        while (merges_done < num_merges && !heap.empty()) {
            MergeJob top = heap.top(); heap.pop();

            // Lazy refresh
            int current = pair_counts[top.pair];
            if (top.count != static_cast<uint64_t>(current)) {
                top.count = current;
                if (top.count > 0) heap.push(top);
                continue;
            }
            if (top.count == 0) break;

            // Record merge
            uint32_t new_id = 256 + merges_done;
            merges[top.pair] = new_id;

            // Merge this pair in all words where it occurs
            std::unordered_map<Pair, std::unordered_set<size_t>> local_pos_updates;
            for (size_t word_idx : top.pos) {
                auto changes = words[word_idx].merge_pair(top.pair, new_id);
                for (auto& [pair, delta] : changes) {
                    int delta_total = delta * counts[word_idx];
                    if (delta_total != 0) {
                        pair_counts[pair] += delta_total;
                        if (delta > 0) local_pos_updates[pair].insert(word_idx);
                    }
                }
            }

            // Add the updated pair counts back to the heap
            for (auto& [pair, pos] : local_pos_updates) {
                int cnt = pair_counts[pair];
                if (cnt > 0) {
                    heap.push(MergeJob{pair, static_cast<uint64_t>(cnt), std::move(pos)});
                }
            }

            merges_done++;
            uint32_t current_percent = (merges_done * 100) / num_merges;
            if (current_percent > last_log_percent) {
                std::cout << "Progress: " << current_percent << "% (" << merges_done << "/" << num_merges
                          << " merges) - Last merge: (" << top.pair.first << "," << top.pair.second
                          << ") -> " << new_id << " (frequency: " << top.count << ")\n";
                last_log_percent = current_percent;
            }
        }
        std::cout << "Finished training: " << merges_done << " merges completed\n";
    }

    // Train from a vector of strings (no parallel ingestion for simplicity)
    void train_from_strings(const std::vector<std::string>& texts, uint32_t vocab_size, const std::string& pattern_override = "") {
        if (!pattern_override.empty()) {
            pattern = pattern_override;
            compiled_pattern = std::regex(pattern, std::regex::optimize | std::regex::ECMAScript);
        }
        std::unordered_map<std::string, int32_t> counts;
        for (const auto& s : texts) {
            for (auto it = std::sregex_iterator(s.begin(), s.end(), compiled_pattern);
                 it != std::sregex_iterator(); ++it) {
                std::string piece = it->str();
                counts[piece]++;
            }
        }
        std::cout << "Processed " << texts.size() << " sequences total, " << counts.size() << " unique\n";

        std::vector<Word> words;
        std::vector<int32_t> cvec;
        for (const auto& [chunk, c] : counts) {
            std::vector<uint32_t> ids(chunk.begin(), chunk.end());
            words.emplace_back(ids);
            cvec.push_back(c);
        }
        train_core_incremental(words, cvec, vocab_size);
    }

    // Return the regex pattern
    std::string get_pattern() const { return pattern; }

    // Return the mergeable ranks (token bytes -> token id / rank)
    std::vector<std::pair<std::vector<uint8_t>, uint32_t>> get_mergeable_ranks() const {
        std::vector<std::pair<std::vector<uint8_t>, uint32_t>> mergeable_ranks;
        std::vector<std::vector<uint8_t>> token_bytes(256);
        for (uint32_t i = 0; i < 256; ++i) token_bytes[i] = {static_cast<uint8_t>(i)};
        for (size_t i = 0; i < token_bytes.size(); ++i)
            mergeable_ranks.emplace_back(token_bytes[i], static_cast<uint32_t>(i));

        // Sort merges by token id
        std::vector<std::pair<Pair, uint32_t>> sorted_merges(merges.begin(), merges.end());
        std::sort(sorted_merges.begin(), sorted_merges.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });

        for (const auto& [pair, merged_id] : sorted_merges) {
            auto [left, right] = pair;
            std::vector<uint8_t> merged_bytes = token_bytes[left];
            merged_bytes.insert(merged_bytes.end(), token_bytes[right].begin(), token_bytes[right].end());
            if (token_bytes.size() <= merged_id) token_bytes.resize(merged_id + 1);
            token_bytes[merged_id] = merged_bytes;
            mergeable_ranks.emplace_back(merged_bytes, merged_id);
        }
        return mergeable_ranks;
    }

    // Encode a string into token IDs
    std::vector<uint32_t> encode(const std::string& text) const {
        std::vector<uint32_t> all_ids;
        for (auto it = std::sregex_iterator(text.begin(), text.end(), compiled_pattern);
             it != std::sregex_iterator(); ++it) {
            std::string chunk = it->str();
            std::vector<uint32_t> ids(chunk.begin(), chunk.end());
            // Apply merges iteratively
            while (ids.size() >= 2) {
                std::optional<std::tuple<size_t, Pair, uint32_t>> best_pair;
                for (size_t i = 0; i + 1 < ids.size(); ++i) {
                    Pair pair = {ids[i], ids[i + 1]};
                    auto it = merges.find(pair);
                    if (it != merges.end()) {
                        if (!best_pair || it->second < std::get<2>(*best_pair)) {
                            best_pair = std::make_tuple(i, pair, it->second);
                        }
                    }
                }
                if (best_pair) {
                    size_t idx = std::get<0>(*best_pair);
                    uint32_t new_id = std::get<2>(*best_pair);
                    ids[idx] = new_id;
                    ids.erase(ids.begin() + idx + 1);
                } else {
                    break;
                }
            }
            all_ids.insert(all_ids.end(), ids.begin(), ids.end());
        }
        return all_ids;
    }

private:
    // Helper: count pairs and where they occur
    static std::pair<
        std::unordered_map<Pair, int32_t>,
        std::unordered_map<Pair, std::unordered_set<size_t>>
    > count_pairs(const std::vector<Word>& words, const std::vector<int32_t>& counts) {
        std::unordered_map<Pair, int32_t> pair_counts;
        std::unordered_map<Pair, std::unordered_set<size_t>> where_to_update;
        for (size_t i = 0; i < words.size(); ++i) {
            if (words[i].ids.size() >= 2 && counts[i] != 0) {
                for (const auto& p : words[i].pairs()) {
                    pair_counts[p] += counts[i];
                    where_to_update[p].insert(i);
                }
            }
        }
        return {pair_counts, where_to_update};
    }
};
