#include <iostream>
#include <chrono>
#include <tuple>
#include <unistd.h>

#include <br_engine.hpp>

const size_t MEM_CTX_SIZE = 12 * 1024 * 1024 * 1024l;

struct ChatApplication {
    ChatApplication() {
        tokenizer_ = br::build_tokenizer("../models/bloomz/tokenizer.json");
    }
    ~ChatApplication() {
        delete tokenizer_;
    }
    void run() {

    }

private:
    br::Tokenizer* tokenizer_;
};

int main(int argc, char* argv[] ) {
    br::CollectiveContext::boot(argc, argv, 2);

    if ( br::CollectiveContext::mpi_rank == 0) {
        ChatApplication* app = new ChatApplication();

        app->run();

        delete app;
    } else if ( br::CollectiveContext::mpi_rank == 1) {
        br::MemoryContext::boot( MEM_CTX_SIZE );
        br::ComputingContext::boot( br::CollectiveContext::nccl_rank );

        br::Enviroment* env = new br::Enviroment(15);

        delete env;

        br::ComputingContext::shutdown();
        br::MemoryContext::shutdown();
    } else if ( br::CollectiveContext::mpi_rank == 2) {
        br::MemoryContext::boot( MEM_CTX_SIZE );
        br::ComputingContext::boot( br::CollectiveContext::nccl_rank );

        br::Enviroment* env = new br::Enviroment(15);

        delete env;

        br::ComputingContext::shutdown();
        br::MemoryContext::shutdown();
    }
    br::CollectiveContext::shutdown();
}

