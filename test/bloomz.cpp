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
        int dummy;
        MPI_Recv(&dummy, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&dummy, 1, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::string text;
        while( readline(">> ", text) ) {
            if ( text.size() > 0 ) {
                auto tokens = tokenizer_->encode(text);

                std::vector<int> ids;
                std::vector<int> masks;
                for(auto& v : tokens) {
                    ids.push_back(v);
                    masks.push_back(1);
                }
                int len = ids.size() + 4;
                len = len - (len % 4);
                len = len - (int)ids.size();

                for (int i = 0; i < len; i++) {
                    ids.push_back( tokenizer_->token_pad() );
                    masks.push_back(0);
                }

                len = (int)ids.size();
                MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(ids.data(), len, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(masks.data(), len, MPI_INT, 0, MPI_COMM_WORLD);
            }
        }

        int n = -1;
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    bool readline(const std::string& prop, std::string& code) {
        std::cout << prop << std::flush;
        if ( std::getline(std::cin, code) ) {
            return true;
        }
        return false;
    }

private:
    br::Tokenizer* tokenizer_;
};

void do_inference(br::Enviroment* env, const char* init_cmd, const char* main_cmd) {
    {
        std::string all_code = br::fileToString("../models/bloomz/common.words");
        all_code = all_code + br::fileToString("../models/bloomz/inference.words");

        br::DaG* init_bin = env->build(all_code);
        for(size_t i = 0; i < env->hashes_num(); i++) {
            env->change(i);
            env->run(init_bin);
        }
        delete init_bin;
    }

    env->execute(init_cmd);

    int ok = 1;
    MPI_Send(&ok, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

    br::DaG* target_cmd = env->build(main_cmd);
    for (;;) {
        int ids = -1;
        MPI_Bcast(&ids, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if ( ids <= 0 ) {
            break;
        }
        env->stack().push_number(ids);
        env->run( target_cmd );
    }

    delete target_cmd;
}

int main(int argc, char* argv[] ) {
    br::CollectiveContext::boot(argc, argv, 2);

    if ( br::CollectiveContext::mpi_rank == 0) {
        ChatApplication* app = new ChatApplication();

        app->run();

        delete app;
    } else if ( br::CollectiveContext::mpi_rank == 1) {
        br::MemoryContext::boot( MEM_CTX_SIZE );
        br::ComputingContext::boot( br::CollectiveContext::nccl_rank );
        br::Enviroment* env = new br::Enviroment(16);
        br::load_nn_words(*env);

        do_inference(env, "init_gpu_0", "main_gpu_0");

        delete env;
        br::ComputingContext::shutdown();
        br::MemoryContext::shutdown();
    } else if ( br::CollectiveContext::mpi_rank == 2) {
        br::MemoryContext::boot( MEM_CTX_SIZE );
        br::ComputingContext::boot( br::CollectiveContext::nccl_rank );
        br::Enviroment* env = new br::Enviroment(16);
        br::load_nn_words(*env);

        do_inference(env, "init_gpu_1", "main_gpu_1");

        delete env;
        br::ComputingContext::shutdown();
        br::MemoryContext::shutdown();
    }
    br::CollectiveContext::shutdown();
}

