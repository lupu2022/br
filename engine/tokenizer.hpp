#ifndef _BR_TOKENSIZER_HPP_
#define _BR_TOKENSIZER_HPP_

#include <string>
#include <map>

namespace br {
struct Tokensizer {
    using id = int32_t;
    using token = std::string;

    virtual std::vector<id> encode(const std::string& text, bool bos) = 0;
    virtual std::string decode(const std::vector<id>& tokens) = 0;
};

Tokensizer* build_tokenizer(const char* file_name);

}
#endif
