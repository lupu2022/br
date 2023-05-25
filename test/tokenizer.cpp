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
int main() {
    auto tokenizer = br::build_tokenizer("../models/bloomz/bloomz.vocab");

    std::string text = "Translate \"Hi, how are you?\" in 中文:";
    auto tokens = tokenizer->encode(text);
    for(auto& v : tokens) {
        std::cout << v << std::endl;
    }
    std::cout << tokenizer->decode(tokens) << std::endl;

    delete tokenizer;
    return 0;
}
