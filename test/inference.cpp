#include <iostream>
#include <chrono>
#include <tuple>
#include <unistd.h>
#include <sys/wait.h>

#include <br_engine.hpp>

const size_t MEM_CTX_SIZE = 14 * 1024 * 1024 * 1024l;

struct ChatApplication {
    ChatApplication() {
        tokenizer_ = br::build_tokenizer("../models/baichuan-7B/tokenizer.model");
    }
    ~ChatApplication() {
        delete tokenizer_;
    }

    void write_all(const void* buf, size_t nbyte) {
        br_assert( br::CollectiveContext::pipe_write(1, buf, nbyte) > 0, "write_all error");
        br_assert( br::CollectiveContext::pipe_write(2, buf, nbyte) > 0, "write_all error");
    }

    void wait_all_ready() {
        int dummy = -1;
        br::CollectiveContext::pipe_read(&dummy, sizeof(int));
        br_assert(dummy == 1, "wait_all_ready error");

        dummy = -1;
        br::CollectiveContext::pipe_read(&dummy, sizeof(int));
        br_assert(dummy == 1, "wait_all_ready error");
    }

    void run() {
        wait_all_ready();

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
                    ids.push_back( tokenizer_->token_unk() );
                    masks.push_back(0);
                }
                len = (int)ids.size();

                for (int i = 0; i < len; i++) {
                    std::cout << ids[i] << " ";
                }
                std::cout << std::endl;

                write_all(&len, sizeof(int));
                write_all(ids.data(), len * sizeof(int));
                write_all(masks.data(), len * sizeof(int));

                /*
                MPI_Recv(ids.data(), len, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                std::cout << "===> " << tokenizer_->decode( ids) << std::endl;
                */
            }
        }

        int n = -1;
        write_all(&n, sizeof(int));
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
        std::string all_code = br::fileToString("../models/baichuan-7B/common.words");
        all_code = all_code + br::fileToString("../models/baichuan-7B/inference.words");

        br::DaG* init_bin = env->build(all_code);
        for(size_t i = 0; i < env->hashes_num(); i++) {
            env->change(i);
            env->run(init_bin);
        }
        delete init_bin;
    }
    env->execute(init_cmd);

    int ok = 1;
    br_assert( br::CollectiveContext::pipe_write(0, &ok, sizeof(int)) > 0, "pipe_write error");

    br::DaG* target_cmd = env->build(main_cmd);

    for (;;) {
        int ids = -1;
        br::CollectiveContext::pipe_read(&ids, sizeof(int));;
        if ( ids <= 0 ) {
            break;
        }
        env->stack().push_number(ids);
        env->run(target_cmd);
    }

    delete target_cmd;
}

int main(int argc, char* argv[] ) {
    br::CollectiveContext::boot_pipe(2);

    if ( br::CollectiveContext::pipe_rank == 0) {
        ChatApplication* app = new ChatApplication();
        app->run();
        delete app;

        // wait for all child processes finished
        {
            int status = 0;
            while ( wait(&status) != -1) {
            }
        }
    } else if ( br::CollectiveContext::pipe_rank == 1) {
        br::MemoryContext::boot( MEM_CTX_SIZE );
        br::ComputingContext::boot( br::CollectiveContext::nccl_rank );
        br::Enviroment* env = new br::Enviroment(17);

        br::load_nn_words(*env);
        do_inference(env, "init_gpu_0", "main_gpu_0");

        delete env;
        br::ComputingContext::shutdown();
        br::MemoryContext::shutdown();
    } else if ( br::CollectiveContext::pipe_rank == 2) {
        br::MemoryContext::boot( MEM_CTX_SIZE );
        br::ComputingContext::boot( br::CollectiveContext::nccl_rank );
        br::Enviroment* env = new br::Enviroment(17);

        br::load_nn_words(*env);
        do_inference(env, "init_gpu_1", "main_gpu_1");

        delete env;
        br::ComputingContext::shutdown();
        br::MemoryContext::shutdown();
    }
    br::CollectiveContext::shutdown();
}

