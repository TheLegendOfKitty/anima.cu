#pragma once

#include <string>
#include <vector>
#include <sentencepiece_processor.h>

// T5 SentencePiece tokenizer wrapper
class T5Tokenizer {
public:
    bool load(const std::string& model_path);
    std::vector<int> tokenize(const std::string& text) const;

private:
    sentencepiece::SentencePieceProcessor sp_;
    static constexpr int EOS_ID = 1;
};
