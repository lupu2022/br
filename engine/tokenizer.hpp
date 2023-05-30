#ifndef _BR_TOKENSIZER_HPP_
#define _BR_TOKENSIZER_HPP_

#include <string>
#include <map>
#include <utility>

namespace br {
struct Tokensizer {
    virtual ~Tokensizer() {}

    virtual std::vector<int> encode(const std::string& text, bool bos = false) = 0;
    virtual std::string decode(const int id) = 0;
    virtual std::string decode(const std::vector<int>& tokens) = 0;
};

Tokensizer* build_tokenizer(const char* file_name);

}
#endif
