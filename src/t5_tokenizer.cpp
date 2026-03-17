#include "t5_tokenizer.h"
#include <cstdio>

bool T5Tokenizer::load(const std::string& model_path) {
    auto status = sp_.Load(model_path);
    if (!status.ok()) {
        fprintf(stderr, "[t5_tok] failed to load %s: %s\n",
                model_path.c_str(), status.ToString().c_str());
        return false;
    }
    fprintf(stderr, "[t5_tok] loaded: %d vocab\n", sp_.GetPieceSize());
    return true;
}

std::vector<int> T5Tokenizer::tokenize(const std::string& text) const {
    std::vector<int> ids;
    sp_.Encode(text, &ids);

    // Ensure EOS token at end
    if (ids.empty() || ids.back() != EOS_ID)
        ids.push_back(EOS_ID);

    return ids;
}
