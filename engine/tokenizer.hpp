#ifndef _BR_TOKEgnizer_HPP_
#define _BR_TOKEgnizer_HPP_

#include <string>
#include <map>
#include <utility>

namespace br {
struct Tokenizer {
    virtual ~Tokenizer() {}

    virtual std::vector<int> encode(const std::string& text, bool bos = false) = 0;
    virtual std::string decode(const int id) = 0;
    virtual std::string decode(const std::vector<int>& tokens) = 0;
    virtual int token_unk() = 0;
};

Tokenizer* build_tokenizer(const char* file_name);

}
#endif
