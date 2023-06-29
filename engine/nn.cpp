#include "tensortype.hpp"
#include "context.hpp"
#include "nn.hpp"

namespace br {

namespace nn {
    std::vector<size_t> fetch_shape(Stack& stack) {
        auto nums = stack.pop_number_list();
        std::vector<size_t> shape;
        for ( size_t i = 0; i < nums.size(); i++) {
            shape.push_back( nums[i] );
        }
        return shape;
    }
    struct Sync : public NativeWord {
        void run(Stack& stack) override {
            auto stream = br::ComputingContext::cuda_stream;
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        NWORD_CREATOR_DEFINE_LR(Sync)
    };
    struct Create : public NativeWord {
        void run(Stack& stack) override {
            auto dtype = stack.pop_string();
            auto device = stack.pop_string();
            auto shape = fetch_shape(stack);
            tensor_t t;
            if ( dtype == "float" && device == "cuda" ) {
                t = br::create_cuda_float(shape);
            } else if ( dtype == "float" && device == "cpu" ) {
                t = br::create_cpu_float(shape);
            } else if (dtype == "int" && device=="cuda") {
                t = br::create_cuda_int(shape);
            } else if (dtype == "int" && device=="cpu") {
                t = br::create_cpu_int(shape);
            } else if ( dtype == "fp16" && device == "cuda" ) {
                t = br::create_cuda_fp16(shape);
            } else if ( dtype == "fp16" && device == "cpu" ) {
                t = br::create_cpu_fp16(shape);
            } else {
                br_panic("Can' be here!");
            }
            stack.push_tensor(t);
        }
        NWORD_CREATOR_DEFINE_LR(Create)
    };

    struct Null : public NativeWord {
        void run(Stack& stack) override {
            tensor_t ret = nullptr;
            stack.push_tensor(ret);
        }

        NWORD_CREATOR_DEFINE_LR(Null);
    };

    struct Zero : public NativeWord {
        void run(Stack& stack) override {
            tensor_t t = stack.pop_tensor();
            t->op_zero(t);
        }
        NWORD_CREATOR_DEFINE_LR(Zero)
    };

    struct Fill : public NativeWord {
        void run(Stack& stack) override {
            double value = stack.pop_number();
            tensor_t t = stack.pop_tensor();
            t->op_fill(t, value);
        }
        NWORD_CREATOR_DEFINE_LR(Fill)
    };

    struct Alibi : public NativeWord {
        void run(Stack& stack) override {
            tensor_t t = stack.pop_tensor();
            t->op_alibi(t);
        }
        NWORD_CREATOR_DEFINE_LR(Alibi)
    };

    struct CausalMask : public NativeWord {
        void run(Stack& stack) override {
            auto out = stack.pop_tensor();
            auto self = stack.pop_tensor();
            self->op_causal_mask(self, out);
        }
        NWORD_CREATOR_DEFINE_LR(CausalMask)
    };

    struct View : public NativeWord {
        void run(Stack& stack) override {
            auto shape = fetch_shape(stack);
            size_t offset = stack.pop_number();
            tensor_t t = stack.pop_tensor();
            auto ret = t->op_view(t, offset, shape);
            stack.push_tensor( std::get<1>(ret) );
        }
        NWORD_CREATOR_DEFINE_LR(View)
    };

    struct Embed : public NativeWord {
        void run(Stack& stack) override {
            auto out = stack.pop_tensor();
            auto table = stack.pop_tensor();
            auto self = stack.pop_tensor();

            self->op_embed(self, table, out);
        }
        NWORD_CREATOR_DEFINE_LR(Embed)
    };

    struct Copy : public NativeWord {
        void run(Stack& stack) override {
            tensor_t src = stack.pop_tensor();
            tensor_t dst = stack.pop_tensor();
            dst->op_copy(dst, src);
        }
        NWORD_CREATOR_DEFINE_LR(Copy)
    };

    struct Linear : public NativeWord {
        void run(Stack& stack) override {
            tensor_t y = stack.pop_tensor();
            tensor_t b = stack.pop_tensor();
            tensor_t w = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();

            x->op_linear(x, w, b, y);
        }
        NWORD_CREATOR_DEFINE_LR(Linear)
    };

    struct Layernorm : public NativeWord {
        void run(Stack& stack) override {
            auto eps = stack.pop_number();
            tensor_t y = stack.pop_tensor();
            tensor_t bias = stack.pop_tensor();
            tensor_t scale = stack.pop_tensor();
            tensor_t var = stack.pop_tensor();
            tensor_t mean = stack.pop_tensor();

            tensor_t x = stack.pop_tensor();

            x->op_layernorm(x, mean, var, scale, bias, y, eps);
        }
        NWORD_CREATOR_DEFINE_LR(Layernorm)
    };

    struct RMSnorm : public NativeWord {
        void run(Stack& stack) override {
            auto eps = stack.pop_number();
            tensor_t y = stack.pop_tensor();
            tensor_t norm2 = stack.pop_tensor();
            tensor_t scale = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();

            x->op_rmsnorm(x, scale, norm2, y, eps);
        }
        NWORD_CREATOR_DEFINE_LR(RMSnorm)
    };

    struct Transpos0213 : public NativeWord {
        void run(Stack& stack) override {
            tensor_t y = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();

            x->op_transpos_0213(x,y);
        }
        NWORD_CREATOR_DEFINE_LR(Transpos0213)
    };

    struct QueryKey : public NativeWord {
        void run(Stack& stack) override {
            tensor_t qk = stack.pop_tensor();
            tensor_t k = stack.pop_tensor();
            tensor_t q = stack.pop_tensor();

            q->op_qk(q, k, qk);
        }
        NWORD_CREATOR_DEFINE_LR(QueryKey)
    };

    struct Scale : public NativeWord {
        void run(Stack& stack) override {
            float scale = stack.pop_number();
            tensor_t x = stack.pop_tensor();
            x->op_scale(x, scale);
        }
        NWORD_CREATOR_DEFINE_LR(Scale)
    };

    struct Add : public NativeWord {
        void run(Stack& stack) override {
            tensor_t c = stack.pop_tensor();
            tensor_t b = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            x->op_add(x, b, c);
        }
        NWORD_CREATOR_DEFINE_LR(Add)
    };

    struct Softmax : public NativeWord {
        void run(Stack& stack) override {
            tensor_t out = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            x->op_softmax(x, out);
        }
        NWORD_CREATOR_DEFINE_LR(Softmax)
    };

    struct Attn : public NativeWord {
        void run(Stack& stack) override {
            tensor_t out = stack.pop_tensor();
            tensor_t value = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            x->op_attn(x, value, out);
        }
        NWORD_CREATOR_DEFINE_LR(Attn)
    };

    struct Gelu : public NativeWord {
        void run(Stack& stack) override {
            tensor_t out = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            x->op_gelu(x, out);
        }
        NWORD_CREATOR_DEFINE_LR(Gelu)
    };

    struct LastLogits : public NativeWord {
        void run(Stack& stack) override {
            tensor_t out = stack.pop_tensor();
            tensor_t lm_head = stack.pop_tensor();
            tensor_t mask = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            x->op_last_logits(x, mask, lm_head, out);
        }
        NWORD_CREATOR_DEFINE_LR(LastLogits)
    };

    struct SamplingTopP : public NativeWord {
        void run(Stack& stack) override {
            float top_p = stack.pop_number();
            float temp = stack.pop_number();
            tensor_t ids = stack.pop_tensor();
            tensor_t mask = stack.pop_tensor();
            tensor_t logits = stack.pop_tensor();
            logits->op_sampling_top_p(logits, mask, ids, temp, top_p);
        }
        NWORD_CREATOR_DEFINE_LR(SamplingTopP);
    };

    struct LossBackward : public NativeWord {
        void run(Stack& stack) override {
            tensor_t lm_head_g = stack.pop_tensor();
            tensor_t x_g = stack.pop_tensor();
            tensor_t workspace = stack.pop_tensor();
            tensor_t lm_head = stack.pop_tensor();
            tensor_t mask = stack.pop_tensor();
            tensor_t ids = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            x->op_loss_backward(x, ids, mask, lm_head, workspace, x_g, lm_head_g);
        }
        NWORD_CREATOR_DEFINE_LR(LossBackward)
    };

    struct LayernormBackward : public NativeWord {
        void run(Stack& stack) override {
            auto eps = stack.pop_number();
            tensor_t din = stack.pop_tensor();
            tensor_t dbias = stack.pop_tensor();
            tensor_t dscale = stack.pop_tensor();
            tensor_t y = stack.pop_tensor();
            tensor_t var = stack.pop_tensor();
            tensor_t bias = stack.pop_tensor();
            tensor_t scale = stack.pop_tensor();
            tensor_t self = stack.pop_tensor();
            self->op_layernorm_backward(self, scale, bias, var, y, dscale, dbias, din, eps);
        }
        NWORD_CREATOR_DEFINE_LR(LayernormBackward)
    };
    struct LinearBackward : public NativeWord {
        void run(Stack& stack) override {
            tensor_t bias_g = stack.pop_tensor();
            tensor_t weight_g = stack.pop_tensor();
            tensor_t x_g = stack.pop_tensor();
            tensor_t bias = stack.pop_tensor();
            tensor_t weight = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            tensor_t self = stack.pop_tensor();
            self->op_linear_backward(self, x, weight, bias, x_g, weight_g, bias_g);
        }
        NWORD_CREATOR_DEFINE_LR(LinearBackward)
    };
    struct GeluBackward : public NativeWord {
        void run(Stack& stack) override {
            tensor_t x_g = stack.pop_tensor();
            tensor_t x = stack.pop_tensor();
            tensor_t self = stack.pop_tensor();
            self->op_gelu_backward(self, x, x_g);
        }
        NWORD_CREATOR_DEFINE_LR(GeluBackward)
    };
    struct AttnBackward : public NativeWord {
        void run(Stack& stack) override {
            tensor_t v_g = stack.pop_tensor();
            tensor_t attn_g = stack.pop_tensor();
            tensor_t v = stack.pop_tensor();
            tensor_t attn = stack.pop_tensor();
            tensor_t self = stack.pop_tensor();
            self->op_attn_backward(self, attn, v, attn_g, v_g);
        }
        NWORD_CREATOR_DEFINE_LR(AttnBackward)
    };
    struct SoftmaxBackward : public NativeWord {
        void run(Stack& stack) override {
            tensor_t x_g = stack.pop_tensor();
            tensor_t out = stack.pop_tensor();
            tensor_t self = stack.pop_tensor();
            self->op_softmax_backward(self, out, x_g);
        }
        NWORD_CREATOR_DEFINE_LR(SoftmaxBackward)
    };
    struct SoftmaxAttnBackward : public NativeWord {
        void run(Stack& stack) override {
            tensor_t v_g = stack.pop_tensor();
            tensor_t attn_g = stack.pop_tensor();
            tensor_t v = stack.pop_tensor();
            tensor_t attn = stack.pop_tensor();
            tensor_t self = stack.pop_tensor();
            self->op_softmax_attn_backward(self, attn, v, attn_g, v_g);
        }
        NWORD_CREATOR_DEFINE_LR(SoftmaxAttnBackward)
    };

    struct QueryKeyBackward : public NativeWord {
        void run(Stack& stack) override {
            tensor_t k_g = stack.pop_tensor();
            tensor_t q_g = stack.pop_tensor();
            tensor_t k = stack.pop_tensor();
            tensor_t q = stack.pop_tensor();
            tensor_t self = stack.pop_tensor();
            self->op_qk_backward(self, q, k, q_g, k_g);
        }
        NWORD_CREATOR_DEFINE_LR(QueryKeyBackward)
    };
}

namespace io {
    struct Dump : public NativeWord {
        void run(Stack& stack) override {
            tensor_t t = stack.pop_tensor();
            t->io_dump(t);
        }
        NWORD_CREATOR_DEFINE_LR(Dump)
    };

    struct Load : public NativeWord {
        void run(Stack& stack) override {
            std::string fileName = stack.pop_string();
            tensor_t x = stack.pop_tensor();
            x->io_load(x, fileName.c_str());
        }
        NWORD_CREATOR_DEFINE_LR(Load)
    };

    struct Save : public NativeWord {
        void run(Stack& stack) override {
            std::string fileName = stack.pop_string();
            tensor_t x = stack.pop_tensor();
            x->io_save(x, fileName.c_str());
        }
        NWORD_CREATOR_DEFINE_LR(Save)
    };

    struct MPIRank : public NativeWord {
        void run(Stack& stack) override {
            stack.push_number( CollectiveContext::mpi_rank );
        }
        NWORD_CREATOR_DEFINE_LR(MPIRank)
    };

    struct PipeRank : public NativeWord {
        void run(Stack& stack) override {
            stack.push_number( CollectiveContext::pipe_rank );
        }
        NWORD_CREATOR_DEFINE_LR(PipeRank)
    };

    struct NcclRank : public NativeWord {
        void run(Stack& stack) override {
            stack.push_number( CollectiveContext::nccl_rank );
        }
        NWORD_CREATOR_DEFINE_LR(NcclRank)
    };

    struct MPIRecv : public NativeWord {
        void run(Stack& stack) override {
            int source = stack.pop_number();
            tensor_t x = stack.pop_tensor();
            x->io_mpi_recv(x, source);
        }
        NWORD_CREATOR_DEFINE_LR(MPIRecv)
    };

    struct MPISend : public NativeWord {
        void run(Stack& stack) override {
            int dst = stack.pop_number();
            tensor_t x = stack.pop_tensor();
            x->io_mpi_send(x, dst);
        }
        NWORD_CREATOR_DEFINE_LR(MPISend)
    };

    struct MPIBcast : public NativeWord {
        void run(Stack& stack) override {
            int root = stack.pop_number();
            tensor_t x = stack.pop_tensor();
            x->io_mpi_bcast(x, root);
        }
        NWORD_CREATOR_DEFINE_LR(MPIBcast)
    };

    struct NcclRecv : public NativeWord {
        void run(Stack& stack) override {
            int source = stack.pop_number();
            tensor_t x = stack.pop_tensor();
            x->io_nccl_recv(x, source);
        }
        NWORD_CREATOR_DEFINE_LR(NcclRecv)
    };

    struct NcclSend : public NativeWord {
        void run(Stack& stack) override {
            int dst = stack.pop_number();
            tensor_t x = stack.pop_tensor();
            x->io_nccl_send(x, dst);
        }
        NWORD_CREATOR_DEFINE_LR(NcclSend)
    };

    struct PipeRead : public NativeWord {
        void run(Stack& stack) override {
            tensor_t x = stack.pop_tensor();
            x->io_pipe_read(x);
        }
        NWORD_CREATOR_DEFINE_LR(PipeRead)
    };

    struct PipeWrite : public NativeWord {
        void run(Stack& stack) override {
            int dst = stack.pop_number();
            tensor_t x = stack.pop_tensor();
            x->io_pipe_write(x, dst);
        }
        NWORD_CREATOR_DEFINE_LR(PipeWrite)
    };

}

void load_nn_words(Enviroment& env) {
    env.insert_native_word("io.dump", io::Dump::creator );
    env.insert_native_word("io.load", io::Load::creator );
    env.insert_native_word("io.save", io::Save::creator );

    env.insert_native_word("io.mpi_rank", io::MPIRank::creator );
    env.insert_native_word("io.pipe_rank", io::PipeRank::creator );
    env.insert_native_word("io.nccl_rank", io::NcclRank::creator );
    env.insert_native_word("io.mpi.bcast", io::MPIBcast::creator );
    env.insert_native_word("io.mpi.recv", io::MPIRecv::creator );
    env.insert_native_word("io.mpi.send", io::MPISend::creator );
    env.insert_native_word("io.nccl.send", io::NcclSend::creator );
    env.insert_native_word("io.nccl.recv", io::NcclRecv::creator );
    env.insert_native_word("io.pipe.read", io::PipeRead::creator );
    env.insert_native_word("io.pipe.write", io::PipeWrite::creator );

    env.insert_native_word("op.sync", nn::Sync::creator );
    env.insert_native_word("op.create", nn::Create::creator );
    env.insert_native_word("op.null", nn::Null::creator );
    env.insert_native_word("op.zero", nn::Zero::creator );
    env.insert_native_word("op.fill", nn::Fill::creator );
    env.insert_native_word("op.alibi", nn::Alibi::creator );
    env.insert_native_word("op.causal_mask", nn::CausalMask::creator );
    env.insert_native_word("op.scale", nn::Scale::creator );
    env.insert_native_word("op.view", nn::View::creator );
    env.insert_native_word("op.embed", nn::Embed::creator );
    env.insert_native_word("op.copy", nn::Copy::creator );
    env.insert_native_word("op.linear", nn::Linear::creator );
    env.insert_native_word("op.layernorm", nn::Layernorm::creator );
    env.insert_native_word("op.rmsnorm", nn::RMSnorm::creator );
    env.insert_native_word("op.transpos_0213", nn::Transpos0213::creator );
    env.insert_native_word("op.add", nn::Add::creator);
    env.insert_native_word("op.querykey", nn::QueryKey::creator);
    env.insert_native_word("op.softmax", nn::Softmax::creator);
    env.insert_native_word("op.attn", nn::Attn::creator);
    env.insert_native_word("op.gelu", nn::Gelu::creator);
    env.insert_native_word("op.last_logits", nn::LastLogits::creator);
    env.insert_native_word("op.sampling_top_p", nn::SamplingTopP::creator);
    env.insert_native_word("op.loss_backward", nn::LossBackward::creator);
    env.insert_native_word("op.layernorm_backward", nn::LayernormBackward::creator);
    env.insert_native_word("op.linear_backward", nn::LinearBackward::creator);
    env.insert_native_word("op.gelu_backward", nn::GeluBackward::creator);
    env.insert_native_word("op.attn_backward", nn::AttnBackward::creator);
    env.insert_native_word("op.softmax_backward", nn::SoftmaxBackward::creator);
    env.insert_native_word("op.softmax_attn_backward", nn::SoftmaxAttnBackward::creator);
    env.insert_native_word("op.qk_backward", nn::QueryKeyBackward::creator);
}

}// end of namespace br
