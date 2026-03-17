/*
 * Qwen3 BPE Tokenizer - C++ port from flux2.c/flux_qwen3_tokenizer.c
 * Loads HuggingFace tokenizer.json, implements byte-level BPE.
 */

#include "qwen3_tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>

// ========================= Byte-level encoding =========================

static int byte_to_unicode[256];
static int unicode_to_byte_table[512];
static bool byte_encoder_init = false;

static void init_byte_encoder() {
    if (byte_encoder_init) return;
    memset(byte_to_unicode, 0, sizeof(byte_to_unicode));
    memset(unicode_to_byte_table, 0, sizeof(unicode_to_byte_table));

    for (int i = 33; i <= 126; i++) { byte_to_unicode[i] = i; unicode_to_byte_table[i] = i; }
    for (int i = 161; i <= 172; i++) { byte_to_unicode[i] = i; unicode_to_byte_table[i] = i; }
    for (int i = 174; i <= 255; i++) { byte_to_unicode[i] = i; unicode_to_byte_table[i] = i; }

    int offset = 256;
    for (int i = 0; i < 256; i++) {
        if (byte_to_unicode[i] == 0 && i != 33) {
            byte_to_unicode[i] = offset;
            unicode_to_byte_table[offset] = i;
            offset++;
        }
    }
    byte_to_unicode[0] = 256;
    unicode_to_byte_table[256] = 0;
    byte_encoder_init = true;
}

static int encode_byte_utf8(unsigned char b, char* out) {
    init_byte_encoder();
    int cp = byte_to_unicode[b];
    if (cp < 128) { out[0] = (char)cp; return 1; }
    if (cp < 2048) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    out[0] = '?'; return 1;
}

static char* text_to_bytes(const char* text) {
    init_byte_encoder();
    int len = strlen(text);
    char* result = (char*)malloc(len * 2 + 1);
    int j = 0;
    for (int i = 0; i < len; i++)
        j += encode_byte_utf8((unsigned char)text[i], result + j);
    result[j] = '\0';
    return result;
}

// ========================= Hash functions =========================

static unsigned int fnv_hash(const char* str) {
    unsigned int h = 2166136261u;
    while (*str) { h ^= (unsigned char)*str++; h *= 16777619u; }
    return h;
}

// ========================= JSON parsing helpers =========================

static const char* skip_ws(const char* p) {
    while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

static char* parse_json_string(const char** pp) {
    const char* p = *pp;
    if (*p != '"') return nullptr;
    p++;

    const char* start = p;
    int len = 0;
    while (*p && *p != '"') {
        if (*p == '\\' && p[1]) { p += 2; len++; }
        else { p++; len++; }
    }

    char* result = (char*)malloc(len * 3 + 1);  // extra for unicode escapes
    p = start;
    int i = 0;
    while (*p && *p != '"') {
        if (*p == '\\' && p[1]) {
            p++;
            switch (*p) {
                case 'n': result[i++] = '\n'; break;
                case 'r': result[i++] = '\r'; break;
                case 't': result[i++] = '\t'; break;
                case '\\': result[i++] = '\\'; break;
                case '"': result[i++] = '"'; break;
                case 'u': {
                    if (p[1] && p[2] && p[3] && p[4]) {
                        char hex[5] = {p[1], p[2], p[3], p[4], 0};
                        int cp = (int)strtol(hex, nullptr, 16);
                        p += 4;
                        if (cp < 0x80) result[i++] = (char)cp;
                        else if (cp < 0x800) {
                            result[i++] = (char)(0xC0 | (cp >> 6));
                            result[i++] = (char)(0x80 | (cp & 0x3F));
                        } else {
                            result[i++] = (char)(0xE0 | (cp >> 12));
                            result[i++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                            result[i++] = (char)(0x80 | (cp & 0x3F));
                        }
                    }
                    break;
                }
                default: result[i++] = *p; break;
            }
            p++;
        } else {
            result[i++] = *p++;
        }
    }
    result[i] = '\0';
    if (*p == '"') p++;
    *pp = p;
    return result;
}

static int parse_json_int(const char** pp) {
    const char* p = *pp;
    int neg = 0;
    if (*p == '-') { neg = 1; p++; }
    int val = 0;
    while (*p >= '0' && *p <= '9') { val = val * 10 + (*p - '0'); p++; }
    *pp = p;
    return neg ? -val : val;
}

static const char* skip_json_value(const char* p) {
    p = skip_ws(p);
    if (*p == '"') {
        p++;
        while (*p && *p != '"') { if (*p == '\\' && p[1]) p += 2; else p++; }
        if (*p == '"') p++;
    } else if (*p == '{' || *p == '[') {
        char open = *p, close = (*p == '{') ? '}' : ']';
        int depth = 1; p++;
        while (*p && depth > 0) {
            if (*p == open) depth++;
            else if (*p == close) depth--;
            else if (*p == '"') { p++; while (*p && *p != '"') { if (*p == '\\' && p[1]) p += 2; else p++; } }
            p++;
        }
    } else {
        while (*p && *p != ',' && *p != '}' && *p != ']' && *p != ' ' && *p != '\n') p++;
    }
    return p;
}

// ========================= Pre-tokenization =========================

struct Chunk { const char* start; int len; };

static std::vector<Chunk> pretokenize(const char* text) {
    std::vector<Chunk> chunks;
    const char* p = text;
    while (*p) {
        const char* start = p;
        if (*p == '\'' && p[1]) {
            char lower = tolower(p[1]);
            if (lower == 's' || lower == 't' || lower == 'm' || lower == 'd') p += 2;
            else if ((lower == 'r' || lower == 'v' || lower == 'l') && p[2] && (tolower(p[2]) == 'e' || tolower(p[2]) == 'l')) p += 3;
            else p++;
        } else if (isalpha((unsigned char)*p) || (unsigned char)*p >= 128) {
            while (*p && (isalpha((unsigned char)*p) || (unsigned char)*p >= 128)) {
                if ((unsigned char)*p >= 128) {
                    if (((unsigned char)*p & 0xE0) == 0xC0) p += 2;
                    else if (((unsigned char)*p & 0xF0) == 0xE0) p += 3;
                    else if (((unsigned char)*p & 0xF8) == 0xF0) p += 4;
                    else p++;
                } else p++;
            }
        } else if (*p >= '0' && *p <= '9') {
            while (*p >= '0' && *p <= '9') p++;
        } else if (*p == ' ' && p[1] && (isalpha(p[1]) || (unsigned char)p[1] >= 128)) {
            p++;
            while (*p && (isalpha((unsigned char)*p) || (unsigned char)*p >= 128)) {
                if ((unsigned char)*p >= 128) {
                    if (((unsigned char)*p & 0xE0) == 0xC0) p += 2;
                    else if (((unsigned char)*p & 0xF0) == 0xE0) p += 3;
                    else if (((unsigned char)*p & 0xF8) == 0xF0) p += 4;
                    else p++;
                } else p++;
            }
        } else if (*p == ' ' && p[1] >= '0' && p[1] <= '9') {
            p++;
            while (*p >= '0' && *p <= '9') p++;
        } else if (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') {
            while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
        } else {
            p++;
        }
        if (p > start)
            chunks.push_back({start, (int)(p - start)});
    }
    return chunks;
}

// ========================= BPE encoding =========================

struct TokenNode { char* text; TokenNode* next; };

static TokenNode* make_node(const char* text) {
    auto* n = (TokenNode*)malloc(sizeof(TokenNode));
    n->text = strdup(text);
    n->next = nullptr;
    return n;
}

static void free_list(TokenNode* head) {
    while (head) {
        auto* next = head->next;
        free(head->text);
        free(head);
        head = next;
    }
}

TokenNode* bpe_encode_word(const Qwen3Tokenizer* tok, const char* word);

// ========================= Qwen3Tokenizer implementation =========================

Qwen3Tokenizer::~Qwen3Tokenizer() {
    if (vocab_) {
        for (int i = 0; i < vocab_size_; i++) free(vocab_[i]);
        free(vocab_);
    }
    if (vocab_hash_) {
        for (int i = 0; i < HASH_SIZE; i++) free(vocab_hash_[i].token);
        free(vocab_hash_);
    }
    if (merges_) {
        for (int i = 0; i < num_merges_; i++) { free(merges_[i].left); free(merges_[i].right); }
        free(merges_);
    }
    free(merge_ranks_);
}

int Qwen3Tokenizer::vocab_lookup(const char* token) const {
    unsigned int h = fnv_hash(token) % HASH_SIZE;
    int probes = 0;
    while (vocab_hash_[h].token && probes < HASH_SIZE) {
        if (strcmp(vocab_hash_[h].token, token) == 0) return vocab_hash_[h].id;
        h = (h + 1) % HASH_SIZE;
        probes++;
    }
    return -1;
}

int Qwen3Tokenizer::merge_rank(const char* left, const char* right) const {
    int len1 = strlen(left), len2 = strlen(right);
    char* key = (char*)malloc(len1 + len2 + 2);
    memcpy(key, left, len1);
    key[len1] = ' ';
    memcpy(key + len1 + 1, right, len2);
    key[len1 + len2 + 1] = '\0';

    unsigned int h = fnv_hash(key) % HASH_SIZE;
    int probes = 0;
    while (merge_ranks_[h] != -1 && probes < HASH_SIZE) {
        int rank = merge_ranks_[h];
        if (rank >= 0 && rank < num_merges_ &&
            strcmp(merges_[rank].left, left) == 0 &&
            strcmp(merges_[rank].right, right) == 0) {
            free(key);
            return rank;
        }
        h = (h + 1) % HASH_SIZE;
        probes++;
    }
    free(key);
    return -1;
}

bool Qwen3Tokenizer::load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "[qwen3_tok] cannot open %s\n", path.c_str()); return false; }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* json = (char*)malloc(size + 1);
    fread(json, 1, size, f);
    json[size] = '\0';
    fclose(f);

    vocab_hash_ = (VocabEntry*)calloc(HASH_SIZE, sizeof(VocabEntry));

    auto hash_insert = [&](const char* token, int id) {
        unsigned int h = fnv_hash(token) % HASH_SIZE;
        int probes = 0;
        while (vocab_hash_[h].token && probes < HASH_SIZE) {
            if (strcmp(vocab_hash_[h].token, token) == 0) return;
            h = (h + 1) % HASH_SIZE;
            probes++;
        }
        if (probes < HASH_SIZE) {
            vocab_hash_[h].token = strdup(token);
            vocab_hash_[h].id = id;
        }
    };

    // Parse vocab
    const char* p = strstr(json, "\"model\"");
    if (!p) { fprintf(stderr, "[qwen3_tok] no model section\n"); free(json); return false; }
    p = strstr(p, "\"vocab\"");
    if (!p) { fprintf(stderr, "[qwen3_tok] no vocab\n"); free(json); return false; }
    p = strchr(p, '{');
    if (!p) { free(json); return false; }
    p++;

    // Count entries
    int count = 0;
    const char* cp = p;
    int depth = 1;
    while (*cp && depth > 0) {
        if (*cp == '{') depth++;
        else if (*cp == '}') depth--;
        else if (*cp == '"' && depth == 1) {
            count++;
            cp++;
            while (*cp && *cp != '"') { if (*cp == '\\' && cp[1]) cp += 2; else cp++; }
        }
        cp++;
    }

    vocab_ = (char**)calloc(count + 1000, sizeof(char*));
    int max_id = 0;

    p = skip_ws(p);
    while (*p && *p != '}') {
        if (*p == '"') {
            char* token = parse_json_string(&p);
            p = skip_ws(p);
            if (*p == ':') p++;
            p = skip_ws(p);
            int id = parse_json_int(&p);
            if (token && id >= 0 && id < count + 1000) {
                vocab_[id] = token;
                hash_insert(token, id);
                if (id > max_id) max_id = id;
            } else free(token);
            p = skip_ws(p);
            if (*p == ',') p++;
            p = skip_ws(p);
        } else p++;
    }

    // Parse merges
    p = strstr(json, "\"merges\"");
    if (!p) { fprintf(stderr, "[qwen3_tok] no merges\n"); free(json); return false; }
    p = strchr(p, '[');
    if (!p) { free(json); return false; }
    p++;

    int mc = 0;
    cp = p; depth = 1;
    while (*cp && depth > 0) {
        if (*cp == '[') { if (depth == 1) mc++; depth++; }
        else if (*cp == ']') depth--;
        else if (*cp == '"') { cp++; while (*cp && *cp != '"') { if (*cp == '\\' && cp[1]) cp += 2; else cp++; } }
        if (*cp) cp++;
    }

    num_merges_ = mc;
    merges_ = (BPEMerge*)calloc(mc, sizeof(BPEMerge));
    merge_ranks_ = (int*)malloc(HASH_SIZE * sizeof(int));
    for (int i = 0; i < HASH_SIZE; i++) merge_ranks_[i] = -1;

    p = skip_ws(p);
    int mi = 0;
    while (*p && *p != ']' && mi < mc) {
        if (*p == '[') {
            p++;
            p = skip_ws(p);
            char* left = (*p == '"') ? parse_json_string(&p) : nullptr;
            p = skip_ws(p);
            if (*p == ',') p++;
            p = skip_ws(p);
            char* right = (*p == '"') ? parse_json_string(&p) : nullptr;
            while (*p && *p != ']') p++;
            if (*p == ']') p++;

            if (left && right) {
                merges_[mi] = {left, right, mi};
                int l1 = strlen(left), l2 = strlen(right);
                char* key = (char*)malloc(l1 + l2 + 2);
                memcpy(key, left, l1); key[l1] = ' ';
                memcpy(key + l1 + 1, right, l2); key[l1 + l2 + 1] = '\0';
                unsigned int h = fnv_hash(key) % HASH_SIZE;
                int probes = 0;
                while (merge_ranks_[h] != -1 && probes < HASH_SIZE) { h = (h + 1) % HASH_SIZE; probes++; }
                if (probes < HASH_SIZE) merge_ranks_[h] = mi;
                free(key);
            } else { free(left); free(right); }
            mi++;
            p = skip_ws(p);
            if (*p == ',') p++;
            p = skip_ws(p);
        } else p++;
    }

    // Parse added_tokens
    p = strstr(json, "\"added_tokens\"");
    if (p) {
        p = strchr(p, '[');
        if (p) {
            p++;
            while (*p && *p != ']') {
                if (*p == '{') {
                    p++;
                    char* content = nullptr;
                    int id = -1;
                    while (*p && *p != '}') {
                        p = skip_ws(p);
                        if (*p == '"') {
                            char* key = parse_json_string(&p);
                            p = skip_ws(p);
                            if (*p == ':') p++;
                            p = skip_ws(p);
                            if (key && strcmp(key, "content") == 0 && *p == '"')
                                content = parse_json_string(&p);
                            else if (key && strcmp(key, "id") == 0)
                                id = parse_json_int(&p);
                            else p = skip_json_value(p);
                            free(key);
                        }
                        p = skip_ws(p);
                        if (*p == ',') p++;
                    }
                    if (content && id >= 0 && id < count + 1000 && !vocab_[id]) {
                        vocab_[id] = content;
                        hash_insert(content, id);
                        if (id > max_id) max_id = id;
                        content = nullptr;
                    }
                    free(content);
                    if (*p == '}') p++;
                }
                p = skip_ws(p);
                if (*p == ',') p++;
            }
        }
    }

    vocab_size_ = max_id + 1;
    free(json);
    fprintf(stderr, "[qwen3_tok] loaded: %d vocab, %d merges\n", vocab_size_, num_merges_);
    return true;
}

TokenNode* bpe_encode_word(const Qwen3Tokenizer* tok, const char* word) {
    if (!*word) return nullptr;

    TokenNode* head = nullptr;
    TokenNode* tail = nullptr;
    const char* p = word;
    while (*p) {
        int clen = 1;
        unsigned char c = (unsigned char)*p;
        if ((c & 0xE0) == 0xC0) clen = 2;
        else if ((c & 0xF0) == 0xE0) clen = 3;
        else if ((c & 0xF8) == 0xF0) clen = 4;
        char buf[8];
        memcpy(buf, p, clen);
        buf[clen] = '\0';
        auto* node = make_node(buf);
        if (!head) head = node; else tail->next = node;
        tail = node;
        p += clen;
    }

    bool changed = true;
    while (changed) {
        changed = false;
        int best_rank = tok->num_merges_ + 1;
        TokenNode* best = nullptr;
        for (auto* n = head; n && n->next; n = n->next) {
            int r = tok->merge_rank(n->text, n->next->text);
            if (r >= 0 && r < best_rank) { best_rank = r; best = n; }
        }
        if (best) {
            int l1 = strlen(best->text), l2 = strlen(best->next->text);
            char* merged = (char*)malloc(l1 + l2 + 1);
            memcpy(merged, best->text, l1);
            memcpy(merged + l1, best->next->text, l2);
            merged[l1 + l2] = '\0';
            free(best->text);
            best->text = merged;
            auto* rm = best->next;
            best->next = rm->next;
            free(rm->text);
            free(rm);
            changed = true;
        }
    }
    return head;
}

std::vector<int> Qwen3Tokenizer::tokenize(const std::string& text, int max_len) const {
    if (max_len <= 0) max_len = 131072;
    std::vector<int> tokens;

    auto chunks = pretokenize(text.c_str());
    for (auto& chunk : chunks) {
        if ((int)tokens.size() >= max_len) break;
        char tmp[4096];
        int clen = chunk.len < 4095 ? chunk.len : 4095;
        memcpy(tmp, chunk.start, clen);
        tmp[clen] = '\0';

        char* byte_text = text_to_bytes(tmp);
        if (!byte_text) continue;

        auto* bpe = bpe_encode_word(this, byte_text);
        for (auto* n = bpe; n && (int)tokens.size() < max_len; n = n->next) {
            int id = vocab_lookup(n->text);
            if (id >= 0) tokens.push_back(id);
        }
        free_list(bpe);
        free(byte_text);
    }
    return tokens;
}
