#include <string>
#include <fstream>
#include <map>

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

    virtual std::vector<int> encode(const std::string& text, bool bos) override {
        const uint32_t* ids;
        size_t ids_num;

        tokenizers_encode(rustObj, text.c_str(), text.size(), 0);
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

        tokenizers_decode(rustObj, (const uint32_t *)ids.data(), ids.size(), 0);
        tokenizers_get_decode_str(rustObj, &out, &out_len);

        std::string res;
        for (size_t i = 0; i < out_len; i++) {
            res = res + out[i];
        }
        return res;
    }
};

Tokenizer* build_tokenizer(const char* file_name) {
    return new BloomzTokenizer(file_name);
}

}
