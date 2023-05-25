#include <iostream>
#include <sstream>
#include <chrono>
#include <tuple>
#include <unistd.h>

#include <br_engine.hpp>

const size_t MEM_CTX_SIZE = 4 * 1024 * 1024 * 1024l;

inline std::string fileToString(const char* filename) {
    std::ifstream t(filename);
    std::string str;

    t.seekg(0, std::ios::end);
    str.reserve(t.tellg());
    t.seekg(0, std::ios::beg);

    str.assign((std::istreambuf_iterator<char>(t)),  std::istreambuf_iterator<char>());

    return str;
}

bool readline(const std::string& prop, std::string& code) {
    std::cout << prop << std::flush;
    if ( std::getline(std::cin, code) ) {
        return true;
    }
    return false;
}


int main(int argc, char* argv[] ) {
    std::string text;
    for (int i = 1; i < argc; i++) {
        auto code = fileToString( argv[i] );
        text = text + code + "\n";
    }

    br::CollectiveContext::boot();
    br::MemoryContext::boot( MEM_CTX_SIZE );
    br::ComputingContext::boot( 0 );
    br::Enviroment* env_ = new br::Enviroment(1);

    env_->execute(text);
    std::string code;
    while( readline(">> ", code ) ) {
        env_->execute( code );
    }
    delete env_;

    br::ComputingContext::shutdown();
    br::MemoryContext::shutdown();
    br::CollectiveContext::shutdown();
}

