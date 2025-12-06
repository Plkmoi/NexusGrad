// tokenus.rs — C-ABI compatible BPE tokenizer library (no PyO3)

// ----------------------
// Imports
// ----------------------
use fancy_regex::Regex;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use std::collections::{BinaryHeap, HashMap as StdHashMap};
use std::ffi::{CStr, CString};
use std::hash::BuildHasherDefault;
use std::os::raw::{c_char, c_uint};
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ----------------------
// Types
// ----------------------
type Pair = (String, String);

#[derive(Clone, Debug)]
struct Word {
    tokens: Vec<String>,
}

impl Word {
    fn from_str(text: &str, re: &Regex) -> Self {
        let mut tokens = vec![];
        let mut last = 0;

        for m in re.find_iter(text) {
            let m = m.unwrap();
            let (s, e) = (m.start(), m.end());
            if s > last {
                tokens.push(text[last..s].to_owned());
            }
            tokens.push(text[s..e].to_owned());
            last = e;
        }

        if last < text.len() {
            tokens.push(text[last..].to_owned());
        }

        Self { tokens }
    }

    fn apply_merge(&mut self, target: &Pair, merged: &str) {
        let mut new_tokens = Vec::new();
        let mut i = 0;

        while i < self.tokens.len() {
            if i + 1 < self.tokens.len()
                && self.tokens[i] == target.0
                && self.tokens[i + 1] == target.1
            {
                new_tokens.push(merged.to_owned());
                i += 2;
            } else {
                new_tokens.push(self.tokens[i].clone());
                i += 1;
            }
        }

        self.tokens = new_tokens;
    }

    fn num_pairs(&self) -> usize {
        if self.tokens.len() < 2 {
            0
        } else {
            self.tokens.len() - 1
        }
    }
}

// ----------------------
// Tokenizer struct (same as your logic, minus PyO3)
// ----------------------
pub struct Tokenizer {
    merges: StdHashMap<Pair, u32>,
    pattern: String,
    compiled_pattern: Regex,
}

impl Tokenizer {
    pub fn new() -> Self {
        let pattern = String::from(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]");
        let compiled_pattern = Regex::new(&pattern).unwrap();

        Self {
            merges: StdHashMap::new(),
            pattern,
            compiled_pattern,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        let mut start = 0;
while start < text.len() {
    // Slice the text starting at `start`
    let slice = &text[start..];

    // Search inside slice
    let m = match self.compiled_pattern.find(slice).unwrap() {
        Some(m) => m,
        None => {
            // No more matches: push the rest and break
            tokens.push(slice.to_string());
            break;
        }
    };

    let s = m.start();
    let e = m.end();

    if s > 0 {
        // Push unmatched text before the match
        tokens.push(slice[..s].to_string());
    }

    // Push matched token
    tokens.push(slice[s..e].to_string());

    // Move global index forward
    start += e;
}

        if start < text.len() {
            tokens.push(text[start..].to_string());
        }

        let mut merged: Vec<String> = tokens.clone();
        let mut pairs: Vec<(Pair, usize)> = Vec::new();

        for (i, token) in tokens.iter().enumerate() {
            if i > 0 {
                let prev = &tokens[i - 1];
                let pair = (prev.clone(), token.clone());
                pairs.push((pair, merged.len() - tokens.len() + i));
            }

            let mut j = i + 1;
            while j < tokens.len() {
                let prev = &tokens[j - 1];
                let token_j = &tokens[j];
                let pair = (prev.clone(), token_j.clone());
                pairs.push((pair, merged.len() - tokens.len() + j));
                j += 1;
            }
        }

        let mut pair_positions = FxHashMap::default();
        for (pair, pos) in pairs {
            pair_positions.entry(pair).or_insert_with(Vec::new).push(pos);
        }

        let mut sorted_merges: Vec<_> = self.merges.iter().collect();
        sorted_merges.sort_by_key(|(_, &v)| v);

        for (pair, _) in sorted_merges {
            if let Some(pos_list) = pair_positions.get(pair) {
                let merged_pair = format!("{}{}", pair.0, pair.1);
                for &pos in pos_list {
                    if pos < merged.len() {
                        merged[pos] = merged_pair.clone();
                    }
                }
            }
        }

        let mut ids = Vec::new();
        for token in merged {
            if let Some(id) = self.get_token_id(&token) {
                ids.push(id);
            } else {
                ids.push(self.merges.len() as u32 + 1);
            }
        }

        ids
    }

    pub fn get_token_id(&self, token: &str) -> Option<u32> {
        for ((a, b), id) in &self.merges {
            if format!("{}{}", a, b) == token {
                return Some(*id);
            }
        }
        None
    }

    pub fn prepare_training_from_strings(
        &self,
        texts: &[String],
    ) -> (Vec<Word>, Vec<i32>) {
        let mut words = Vec::new();
        let mut counts = Vec::new();
        let mut map = FxHashMap::default();

        for text in texts {
            let word = Word::from_str(text, &self.compiled_pattern);
            let key = word.tokens.join("");
            *map.entry(key).or_insert(0) += 1;
        }

        for (key, count) in map {
            words.push(Word::from_str(&key, &self.compiled_pattern));
            counts.push(count);
        }

        (words, counts)
    }

    pub fn train_core_incremental(
        &mut self,
        mut words: Vec<Word>,
        counts: Vec<i32>,
        vocab_size: u32,
    ) {
        if vocab_size <= 256 {
            return;
        }

        let merges_needed = vocab_size - 256;
        let merges: Arc<Mutex<StdHashMap<Pair, u32>>> =
            Arc::new(Mutex::new(StdHashMap::new()));
        let mut next_id: u32 = 256;

        for _ in 0..merges_needed {
            let pair_freqs = Self::parallel_count_pairs(&words, &counts);

            if let Some(((pair_first, pair_second), freq)) =
                Self::select_best_pair(&pair_freqs)
            {
                let best_pair = (pair_first.clone(), pair_second.clone());
                let merged_symbol = format!("{}{}", pair_first, pair_second);

                let mut local = merges.lock().unwrap();
                local.insert(best_pair.clone(), next_id);
                next_id += 1;
                drop(local);

                words
                    .par_iter_mut()
                    .zip(counts.par_iter())
                    .for_each(|(word, _)| {
                        word.apply_merge(&best_pair, &merged_symbol);
                    });
            } else {
                break;
            }
        }

        self.merges = Arc::try_unwrap(merges).unwrap().into_inner().unwrap();
    }

    fn parallel_count_pairs(
        words: &[Word],
        counts: &[i32],
    ) -> FxHashMap<Pair, i32> {
        let merged: FxHashMap<Pair, i32> = words
            .par_iter()
            .zip(counts.par_iter())
            .fold(
                || FxHashMap::<Pair, i32>::default(),
                |mut acc, (word, &count)| {
                    for pair in Self::find_symbol_pairs(word) {
                        *acc.entry(pair).or_insert(0) += count;
                    }
                    acc
                },
            )
            .reduce(
                || FxHashMap::default(),
                |mut acc, map| {
                    for (k, v) in map {
                        *acc.entry(k).or_insert(0) += v;
                    }
                    acc
                },
            );
        merged
    }

    fn find_symbol_pairs(word: &Word) -> Vec<Pair> {
        let tokens = &word.tokens;
        if tokens.len() < 2 {
            return vec![];
        }

        let mut result = Vec::with_capacity(tokens.len() - 1);
        for i in 0..tokens.len() - 1 {
            result.push((tokens[i].clone(), tokens[i + 1].clone()));
        }
        result
    }

    fn select_best_pair(pair_freqs: &FxHashMap<Pair, i32>) -> Option<(Pair, i32)> {
        pair_freqs
            .iter()
            .max_by_key(|(_, freq)| *freq)
            .map(|(k, v)| (k.clone(), *v))
    }
}

// ----------------------
// FFI SECTION (C ABI)
// ----------------------

#[no_mangle]
pub extern "C" fn tokenus_new() -> *mut Tokenizer {
    Box::into_raw(Box::new(Tokenizer::new()))
}

#[no_mangle]
pub extern "C" fn tokenus_free(ptr: *mut Tokenizer) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

#[no_mangle]
pub extern "C" fn tokenus_train(
    ptr: *mut Tokenizer,
    texts: *const *const c_char,
    count: usize,
    vocab_size: c_uint,
) {
    let tokenizer = unsafe { &mut *ptr };

    let mut rust_texts = Vec::new();

    for i in 0..count {
        let cstr = unsafe { CStr::from_ptr(*texts.add(i)) };
        rust_texts.push(cstr.to_str().unwrap().to_owned());
    }

    let (words, counts) = tokenizer.prepare_training_from_strings(&rust_texts);
    tokenizer.train_core_incremental(words, counts, vocab_size);
}

#[no_mangle]
pub extern "C" fn tokenus_encode(
    ptr: *mut Tokenizer,
    text: *const c_char,
    out_len: *mut usize,
) -> *mut u32 {
    let tokenizer = unsafe { &mut *ptr };
    let cstr = unsafe { CStr::from_ptr(text) }.to_str().unwrap();
    let ids = tokenizer.encode(cstr);

    let len = ids.len();
    unsafe { *out_len = len };

    let boxed = ids.into_boxed_slice();
    Box::into_raw(boxed) as *mut u32
}

#[no_mangle]
pub extern "C" fn tokenus_free_ids(ptr: *mut u32, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(std::slice::from_raw_parts_mut(ptr, len)));
    }
}
