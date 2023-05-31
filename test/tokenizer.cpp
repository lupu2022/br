#include <iostream>
#include <chrono>
#include <tuple>
#include <unistd.h>

#include <br_engine.hpp>

/*
153772 -> 'Translate'
 17959 -> ' "H'
    76 -> 'i'
 98257 -> ', '
 20263 -> 'how'
  1306 -> ' are'
  1152 -> ' you'
  2040 -> '?'
     5 -> '"'
   361 -> ' in'
104229 -> ' 中文'
    29 -> ':'
*/
int main(int argc, const char* argv[]) {
    auto tokenizer = br::build_tokenizer("../models/bloomz/tokenizer.json");

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
