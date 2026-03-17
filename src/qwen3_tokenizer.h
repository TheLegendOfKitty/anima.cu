#pragma once

#include <string>
#include <vector>

// Qwen3 BPE tokenizer - loads from HuggingFace tokenizer.json
// Ported from flux2.c/flux_qwen3_tokenizer.c

constexpr int QWEN3_TOK_PAD_ID = 151643;      // <|endoftext|>

class Qwen3Tokenizer {
public:
    ~Qwen3Tokenizer();

    // Load vocabulary and merges from tokenizer.json
    bool load(const std::string& path);

    // Tokenize text (no special tokens added, matching add_special_tokens=False)
    std::vector<int> tokenize(const std::string& text, int max_len = 0) const;

    int vocab_size() const { return vocab_size_; }

private:
    static constexpr int HASH_SIZE = 300007;

    struct VocabEntry { char* token; int id; };
    struct BPEMerge { char* left; char* right; int rank; };

    char** vocab_ = nullptr;
    int vocab_size_ = 0;

    VocabEntry* vocab_hash_ = nullptr;
    BPEMerge* merges_ = nullptr;
    int num_merges_ = 0;
    int* merge_ranks_ = nullptr;  // hash table

    int vocab_lookup(const char* token) const;
    int merge_rank(const char* left, const char* right) const;

    friend struct TokenNode* bpe_encode_word(const Qwen3Tokenizer* tok, const char* word);
};
