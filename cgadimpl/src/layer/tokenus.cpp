#include "layer/tokenus.hpp"
#include <random>

namespace ag::layer {




int safe_sample_from_logits_tensor(const Tensor& logits_tensor_cpu, float temp)
{
    // -------------------------------
    // 1. Validate & extract shape
    // -------------------------------
    const auto& dims = logits_tensor_cpu.shape().dims;
    if (dims.empty()) return 0;

    int V = dims.back();             // vocab size
    if (V <= 0) return 0;

    const float* ptr = logits_tensor_cpu.data<float>();
    if (!ptr) return 0;

    // If shape is [B,1,V], or [1,1,V] → flatten automatically
    const float* logits = ptr;

    // -------------------------------
    // 2. Compute max for stability
    // -------------------------------
    float maxlog = -INFINITY;
    for (int i = 0; i < V; ++i)
        if (std::isfinite(ptr[i]) && ptr[i] > maxlog)
            maxlog = ptr[i];

    if (!std::isfinite(maxlog)) {
        // every logit was NaN/inf → fallback uniform
        static thread_local std::mt19937 rng(42);
        std::uniform_int_distribution<int> uni(0, V - 1);
        return uni(rng);
    }

    // -------------------------------
    // 3. Softmax with temperature
    // -------------------------------
    std::vector<double> probs(V);
    double sum = 0.0;

    for (int i = 0; i < V; ++i) {
        float l = ptr[i];
        if (!std::isfinite(l)) {
            probs[i] = 0.0;
        } else {
            double ex = std::exp(double(l - maxlog) / double(temp));
            probs[i] = ex;
            sum += ex;
        }
    }

    // all logits zeroed? fallback
    if (sum == 0.0 || std::isnan(sum)) {
        std::fill(probs.begin(), probs.end(), 1.0 / double(V));
    } else {
        for (int i = 0; i < V; ++i)
            probs[i] /= sum;
    }

    // -------------------------------
    // 4. Sample
    // -------------------------------
    static thread_local std::mt19937 rng(42);
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    int t = dist(rng);

    return std::clamp(t, 0, V - 1);
}



// -------------------------------------------------
// WORD::pairs
// -------------------------------------------------
std::vector<Pair> Word::pairs() const {
    std::vector<Pair> out;
    if (ids.size() < 2) return out;

    out.reserve(ids.size());
    for (size_t i = 0; i + 1 < ids.size(); ++i)
        out.emplace_back(ids[i], ids[i+1]);

    return out;
}

// -------------------------------------------------
// WORD::merge_pair
// -------------------------------------------------
std::vector<std::pair<Pair,int32_t>>
Word::merge_pair(const Pair& p, uint32_t new_id)
{
    const auto [a,b] = p;
    size_t n = ids.size();

    std::vector<std::pair<Pair,int32_t>> deltas;
    if (n < 2) return deltas;

    std::vector<uint32_t> out;
    out.reserve(n);

    deltas.reserve(6);
    size_t i = 0;

    while (i < n) {
        if (i + 1 < n && ids[i] == a && ids[i+1] == b) {

            uint32_t left  = out.empty() ? UINT32_MAX : out.back();
            uint32_t right = (i + 2 < n ? ids[i+2] : UINT32_MAX);

            if (left != UINT32_MAX) {
                deltas.push_back({{left,a}, -1});
                deltas.push_back({{left,new_id}, +1});
            }
            deltas.push_back({{a,b}, -1});

            if (right != UINT32_MAX) {
                deltas.push_back({{b,right}, -1});
                deltas.push_back({{new_id,right}, +1});
            }

            out.push_back(new_id);
            i += 2;
        } else {
            out.push_back(ids[i]);
            i++;
        }
    }

    ids.swap(out);
    return deltas;
}

// -------------------------------------------------
// TOKENIZER::Tokenizer
// -------------------------------------------------
Tokenizer::Tokenizer()
{
    pattern =
       R"([A-Za-z]+|\d+|[^\sA-Za-z0-9]+|\s+)";
    compiled = std::regex(pattern, std::regex::optimize);
}

// -------------------------------------------------
// count_pairs (parallel)
// -------------------------------------------------
std::pair<Tokenizer::CountMap, Tokenizer::PosMap>
Tokenizer::count_pairs(const std::vector<Word>& words,
                       const std::vector<int32_t>& counts)
{
    int N = words.size();
    int T = omp_get_max_threads();

    std::vector<CountMap> local_counts(T);
    std::vector<PosMap>   local_pos(T);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        int tid = omp_get_thread_num();
        if (counts[i] == 0) continue;

        const auto& w = words[i];
        if (w.ids.size() < 2) continue;

        for (auto& p : w.pairs()) {
            local_counts[tid][p] += counts[i];
            local_pos[tid][p].insert(i);
        }
    }

    CountMap out;
    PosMap   posout;

    for (int t = 0; t < T; ++t) {
        for (auto& kv : local_counts[t])
            out[kv.first] += kv.second;

        for (auto& kv : local_pos[t])
            posout[kv.first].insert(kv.second.begin(), kv.second.end());
    }

    return {out, posout};
}

// -------------------------------------------------
// train_core (main BPE loop)
// -------------------------------------------------
void Tokenizer::train_core(std::vector<Word>& words,
                           std::vector<int32_t>& counts,
                           uint32_t vocab_size)
{
    merges.clear();
    uint32_t total_merges = vocab_size - 256;

    auto [pair_counts, where] = count_pairs(words, counts);

    struct Item {
        Pair p;
        uint64_t c;
        std::unordered_set<size_t> pos;
    };

    auto cmp = [](const Item& a, const Item& b) {
        return a.c < b.c; // max-heap
    };

    std::vector<Item> heap;
    heap.reserve(pair_counts.size());

    for (auto& kv : where) {
        int32_t c = pair_counts[kv.first];
        if (c > 0)
            heap.push_back({kv.first, (uint64_t)c, kv.second});
    }

    std::make_heap(heap.begin(), heap.end(), cmp);

    uint32_t done = 0;

    while (done < total_merges && !heap.empty()) {
        std::pop_heap(heap.begin(), heap.end(), cmp);
        Item top = heap.back();
        heap.pop_back();

        uint64_t current = pair_counts[top.p];
        if (current != top.c) {
            if (current > 0) {
                top.c = current;
                heap.push_back(std::move(top));
                std::push_heap(heap.begin(), heap.end(), cmp);
            }
            continue;
        }

        if (current == 0)
            break;

        uint32_t new_id = 256 + done;
        merges[top.p] = new_id;

        int T = omp_get_max_threads();
        std::vector<std::vector<std::pair<Pair,int32_t>>> local_deltas(T);
        std::vector<std::vector<size_t>> local_idxs(T);

        std::vector<size_t> posvec(top.pos.begin(), top.pos.end());
        int M = posvec.size();

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < M; ++i) {
            int tid = omp_get_thread_num();
            size_t idx = posvec[i];

            auto changes = words[idx].merge_pair(top.p, new_id);
            local_idxs[tid].push_back(idx);

            for (auto& ch : changes)
                local_deltas[tid].push_back(ch);
        }

        std::unordered_map<Pair,std::unordered_set<size_t>,PairHash>
        local_pos_updates;

        for (int t = 0; t < T; ++t) {
            for (auto& pr : local_deltas[t]) {
                auto [pair, delta] = pr;

                for (auto idx : local_idxs[t]) {
                    int32_t d = delta * counts[idx];
                    pair_counts[pair] += d;

                    if (delta > 0)
                        local_pos_updates[pair].insert(idx);
                }
            }
        }

        for (auto& kv : local_pos_updates) {
            Pair p = kv.first;
            int32_t c = pair_counts[p];

            if (c > 0) {
                heap.push_back({p, (uint64_t)c, kv.second});
                std::push_heap(heap.begin(), heap.end(), cmp);
            }
        }

        done++;
    }
}

// -------------------------------------------------
// train (public)
// -------------------------------------------------
void Tokenizer::train(const std::vector<std::string>& data,
                      uint32_t vocab_size)
{
    std::unordered_map<std::string,int32_t> freq;

    for (auto& s : data) {
        std::sregex_iterator it(s.begin(), s.end(), compiled);
        std::sregex_iterator end;

        while (it != end) {
            freq[(*it).str()]++;
            ++it;
        }
    }

    std::vector<Word> words;
    std::vector<int32_t> counts;

    words.reserve(freq.size());
    counts.reserve(freq.size());

    for (auto& kv : freq) {
        std::vector<uint32_t> vec;
        vec.reserve(kv.first.size());

        for (unsigned char b : kv.first)
            vec.push_back((uint32_t)b);

        words.emplace_back(vec);
        counts.push_back(kv.second);
    }

    train_core(words, counts, vocab_size);
}

// -------------------------------------------------
// encode
// -------------------------------------------------
std::vector<uint32_t> Tokenizer::encode(const std::string& text) const
{
    std::vector<uint32_t> out;

    std::sregex_iterator it(text.begin(), text.end(), compiled);
    std::sregex_iterator end;

    while (it != end) {
        std::string c = (*it).str();

        std::vector<uint32_t> ids;
        ids.reserve(c.size());
        for (unsigned char b : c)
            ids.push_back((uint32_t)b);

        bool changed = true;
        while (changed && ids.size() >= 2) {
            changed = false;

            size_t best_i = 0;
            uint32_t best_new = UINT32_MAX;

            for (size_t i = 0; i + 1 < ids.size(); ++i) {
                Pair p(ids[i], ids[i+1]);
                auto it2 = merges.find(p);
                if (it2 != merges.end()) {
                    uint32_t newid = it2->second;
                    if (newid < best_new) {
                        best_new = newid;
                        best_i = i;
                        changed = true;
                    }
                }
            }

            if (changed) {
                ids[best_i] = best_new;
                ids.erase(ids.begin() + best_i + 1);
            }
        }

        out.insert(out.end(), ids.begin(), ids.end());
        ++it;
    }

    return out;
}

std::string ag::layer::Tokenizer::decode(const std::vector<uint32_t>& ids) const {
    // Reverse mapping: new_id -> (left,right)
    std::unordered_map<uint32_t, Pair> rev;
    rev.reserve(merges.size());
    for (const auto& kv : merges) {
        rev[kv.second] = kv.first;  // new_id -> (id1,id2)
    }

    // Convert each token ID into its byte sequence
    std::vector<uint8_t> bytes;

    std::function<void(uint32_t)> decode_id = [&](uint32_t id) {
        // If id < 256: it's a byte
        if (id < 256) {
            bytes.push_back((uint8_t)id);
            return;
        }

        // Otherwise a merged token → split recursively
        auto it = rev.find(id);
        if (it == rev.end()) {
            // Unknown merge, fallback: treat as byte '?'
            bytes.push_back((uint8_t)'?');
            return;
        }

        const Pair& p = it->second;
        decode_id(p.first);
        decode_id(p.second);
    };

    // Process all IDs
    for (uint32_t id : ids) {
        decode_id(id);
    }

    // Convert bytes to string
    return std::string(bytes.begin(), bytes.end());
}


} // namespace ag::layer
