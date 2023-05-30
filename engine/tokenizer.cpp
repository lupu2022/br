#include <string>
#include <fstream>
#include <map>

#include "common.hpp"
#include "tokenizer.hpp"

namespace br {

struct BloomzTokensizer : public Tokensizer {
    std::map<std::string, int> token_to_id;
    std::map<int, std::string> id_to_token;

    BloomzTokensizer( std::ifstream& fin) {
        int32_t vocab_size;
        fin.read((char *)&vocab_size, sizeof(vocab_size));

        std::string word;
        for (int i = 0; i < vocab_size; i++) {
            uint32_t len;
            fin.read((char *)&len, sizeof(len));

            word.resize(len);
            fin.read((char *)word.data(), len);

            token_to_id[word] = i;
            id_to_token[i] = word;
        }
        fin.close();
    }
    ~BloomzTokensizer() {

    }

    virtual std::vector<int> encode(const std::string& text, bool bos) override {
        std::vector<int> res;

        if (bos) {
            res.push_back(1); // TODO: replace with vocab.bos
        }

        //find the longest token that matches the text
        int pos = 0;
        while (true) {
            int l = 0;
            int t = 0;
            for (const auto & kv : id_to_token) {
                if (kv.second.size() < l) continue;
                if (kv.second.size() > text.size() - pos) continue;
                if (text.substr(pos, kv.second.size()) == kv.second) {
                    l = kv.second.size();
                    t = kv.first;
                }
            }

            if (l == 0) {
                break;
            }

            res.push_back(t);
            pos += l;
        }

        return res;
    }

    virtual std::string decode(const int id) override {
        return id_to_token[id];
    }

    virtual std::string decode(const std::vector<int>& tokens) override {
        std::string res;
        for (auto id : tokens) {
            res.append( id_to_token[id] );
        }
        return res;
    }
};

Tokensizer* build_tokenizer(const char* file_name) {
    std::ifstream fin = std::ifstream(file_name, std::ios::binary);
    if (!fin) {
        br_panic("Can't open tokenizer vocab file.");
    }

    // verify magic
    uint32_t magic;
    fin.read((char *) &magic, sizeof(magic));
    if (magic == 0x67676d6c) {
        return new BloomzTokensizer(fin);
    }

    br_panic("Can't build tokenizer from vocab file.");
    return nullptr;
}

}
