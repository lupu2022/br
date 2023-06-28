#include <string>
#include <fstream>
#include <map>

#include <sentencepiece_processor.h>
#include <tokenizers_c.h>

#include "common.hpp"
#include "tokenizer.hpp"


inline std::string fileToString(const char* filename) {
    std::ifstream t(filename);
    std::string str;

    t.seekg(0, std::ios::end);
    str.reserve(t.tellg());
    t.seekg(0, std::ios::beg);

    str.assign((std::istreambuf_iterator<char>(t)),
        std::istreambuf_iterator<char>());

    return str;
}

namespace br {

struct BloomzTokenizer : public Tokenizer {
    TokenizerHandle rustObj;
    BloomzTokenizer( const char* file_name) {
        std::string json = br::fileToString(file_name);
        rustObj = tokenizers_new_from_str(json.c_str(), json.size());
    }
    ~BloomzTokenizer() {
        if ( rustObj != nullptr) {
            tokenizers_free(rustObj);
        }
        rustObj = nullptr;
    }

    virtual int token_unk() override {
        return 0;
    }

    virtual std::vector<int> encode(const std::string& text, bool bos) override {
        const uint32_t* ids;
        size_t ids_num;

        tokenizers_encode(rustObj, text.c_str(), text.size(), token_unk());
        tokenizers_get_encode_ids(rustObj, &ids, &ids_num);

        std::vector<int> res;
        for(size_t i = 0; i < ids_num; i++) {
            res.push_back( ids[i] );
        }
        return res;
    }

    virtual std::string decode(const int id) override {
        std::vector<int> ids;
        ids.push_back(id);
        return std::move( decode(ids) );
    }

    virtual std::string decode(const std::vector<int>& ids) override {
        const char* out;
        size_t out_len;

        tokenizers_decode(rustObj, (const uint32_t *)ids.data(), ids.size(), token_unk());
        tokenizers_get_decode_str(rustObj, &out, &out_len);

        std::string res;
        for (size_t i = 0; i < out_len; i++) {
            res = res + out[i];
        }
        return res;
    }
};

struct BaichuanTokenizer : public Tokenizer {
    sentencepiece::SentencePieceProcessor processor;

    BaichuanTokenizer( const char* file_name) {
        const auto status = processor.Load(file_name);
        if (!status.ok()) {
            br_panic("Can't crate BaichuanTokenizer from model file!");
        }
    }
    ~BaichuanTokenizer() {
    }

    virtual int token_unk() override {
        return 0;
    }

    virtual std::vector<int> encode(const std::string& text, bool bos) override {
        std::vector<int> res;
        processor.Encode(text, &res);
        return res;
    }

    virtual std::string decode(const int id) override {
        std::vector<int> ids = { id };
        std::string text;
        processor.Decode(ids, &text);
        return text;
    }

    virtual std::string decode(const std::vector<int>& ids) override {
        std::string text;
        processor.Decode(ids, &text);
        return text;
    }
};


inline bool ends_with(std::string const & value, std::string const & ending) {
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

Tokenizer* build_tokenizer(const char* file_name) {
    std::string fname = file_name;
    if ( ends_with(fname, ".json") ) {
        return new BloomzTokenizer(file_name);
    }
    if( ends_with(fname, ".model") ) {
        return new BaichuanTokenizer(file_name);
    }
    br_panic("Can't create tokenizer from file");
    return nullptr;
}

}
