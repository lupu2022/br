#include <iostream>
#include <chrono>
#include <tuple>
#include <unistd.h>

#include <br_engine.hpp>

int main(int argc, const char* argv[]) {
    //auto tokenizer = br::build_tokenizer("../models/bloomz/tokenizer.json");
    auto tokenizer = br::build_tokenizer("../models/baichuan-7B/tokenizer.model");

    //std::string text = "Translate \"Hi, how are you?\" in 中文:";
    std::string text = argv[1];
    auto tokens = tokenizer->encode(text);
    for(auto& v : tokens) {
        std::cout << v << "\t |" << tokenizer->decode(v)  << "|" << std::endl;
    }
    std::vector<int> ids;
    for ( size_t i = 0; i < tokens.size(); i++) {
        ids.push_back( tokens[i] );
    }
    std::cout << tokenizer->decode( ids) << std::endl;

    delete tokenizer;
    return 0;
}
