#include "common.hpp"
#include "tensortype.hpp"
#include "cpu_tensor.hpp"
#include "cuda_tensor.hpp"

namespace br {

ComputingReturn TensorType::op_zero(tensor_t self) {
    br_assert(self.get() == this, "can't be here!");
    marker_ = 0;
    auto ret = impl()->op_zero(self);
    op_check(ret, "zero");
}

ComputingReturn TensorType::op_fill(tensor_t self, float value) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_fill(self, value);
    op_check(ret, "fill");
}

ComputingReturn TensorType::op_alibi(tensor_t self) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_alibi(self);
    op_check(ret, "alibi");
}

ComputingReturn TensorType::op_rotary_cache(tensor_t self, float base, int dims) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_rotary_cache(self, base, dims);
    op_check(ret, "rotary_cache");
}

ComputingReturn TensorType::op_causal_mask(tensor_t self, tensor_t out) {
    br_assert(self.get() == this, "can't be here!");
    br_assert(self->dtype() == DataType::Int, " mask must be int !");
    auto ret = impl()->op_causal_mask(self, out);
    op_check(ret, "causal_mask");
}

ComputingReturn TensorType::op_copy(tensor_t self, tensor_t src) {
    br_assert(self.get() == this, "can't be here!");
    br_assert(items() == src->items(), "copy must has same size");
    br_assert(self->dtype() == src->dtype(), "Copy must has same data type");
    auto ret = impl()->op_copy(self, src);
    op_check(ret, "copy");
}

std::variant<ComputingReturn, tensor_t> TensorType::op_view(tensor_t self, size_t offset, const std::vector<size_t>& newShape_) {
    br_assert(self.get() == this, "can't be here!");
    int dynamic = -1;
    size_t sub_size = 1;
    for (size_t i = 0; i < newShape_.size(); i++) {
        if ( newShape_[i] == 0 ) {
            if ( dynamic == -1 ) {
                dynamic = i;
            } else {
                br_panic("dynamic view must has one zero shape");
            }
        } else {
            sub_size = sub_size * newShape_[i];
        }
    }

    std::vector<size_t> newShape = newShape_;
    if ( dynamic >= 0 ) {
        size_t d = (self->items() - offset) / sub_size;
        newShape[dynamic] = d;
    }

    ShapeType s(newShape);
    br_assert(offset + s.numel() <= items() , "view out of shape!");
    auto result = impl()->op_view(self, offset, newShape);
    if ( result.index() == 0) {
        ComputingReturn ret = std::get<0>(result);
        op_check(ret, "view");
    }
    return result;
}

ComputingReturn TensorType::op_embed(tensor_t self, tensor_t table, tensor_t out) {
    br_assert(self.get() == this, "can't be here!");
    br_assert(self->dtype() == DataType::Int, "token id must be Int");
    br_assert(table->dtype() == out->dtype(), " output and table must have same DataType" );
    br_assert(table->impl_index() == out->impl_index(), "table and outspace must have same device");
    br_assert(table->shape()[1] == out->shape()[2], "table and out must have same hidden size");
    auto ret = impl()->op_embed(self, table, out);
    op_check(ret, "loss_backward");
}

ComputingReturn TensorType::op_scale(tensor_t self, float scale) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_scale(self, scale);
    op_check(ret, "scale");
}

ComputingReturn TensorType::op_add(tensor_t self, tensor_t b, tensor_t c) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_add(self, b, c);
    op_check(ret, "add");
}

ComputingReturn TensorType::op_mul(tensor_t self, tensor_t b, tensor_t c) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_mul(self, b, c);
    op_check(ret, "add");
}

ComputingReturn TensorType::op_linear(tensor_t self, tensor_t w, tensor_t b, tensor_t y) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_linear(self, w, b, y);
    op_check(ret, "linear");
}

ComputingReturn TensorType::op_layernorm(tensor_t self, tensor_t mean, tensor_t var, tensor_t scale, tensor_t bias, tensor_t y, float eps) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_layernorm(self, mean, var, scale, bias, y, eps);
    op_check(ret, "layernorm");
}

ComputingReturn TensorType::op_rmsnorm(tensor_t self, tensor_t scale, tensor_t norm2, tensor_t y, float eps) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_rmsnorm(self, scale, norm2, y, eps);
    op_check(ret, "rmsnorm");
}

ComputingReturn TensorType::op_transpos_0213(tensor_t self, tensor_t y) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_transpos_0213(self, y);
    op_check(ret, "transpose_0213");
}

ComputingReturn TensorType::op_qk(tensor_t self, tensor_t k, tensor_t qk) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_qk(self, k, qk);
    op_check(ret, "qk");
}

ComputingReturn TensorType::op_softmax(tensor_t self, tensor_t out) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_softmax(self, out);
    op_check(ret, "softmax");
}

ComputingReturn TensorType::op_attn(tensor_t self, tensor_t v, tensor_t attn) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_attn(self, v, attn);
    op_check(ret, "attn");
}

ComputingReturn TensorType::op_gelu(tensor_t self, tensor_t dst) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_gelu(self, dst);
    op_check(ret, "attn");
}

ComputingReturn TensorType::op_last_logits(tensor_t self, tensor_t mask, tensor_t lm_head, tensor_t output) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_last_logits(self, mask, lm_head, output);
    op_check(ret, "last_logits");
}

ComputingReturn TensorType::op_sampling_top_p(tensor_t self, tensor_t mask, tensor_t ids, float temp, float top_p) {
    br_assert(self.get() == this, "can't be here!");
    br_assert(ids->dtype() == DataType::Int, "sampling output type Int");
    auto ret = impl()->op_sampling_top_p(self, mask, ids, temp, top_p);
    op_check(ret, "sampling_top_p");
}

std::variant<ComputingReturn, float> TensorType::op_loss_backward(tensor_t self, tensor_t ids, tensor_t mask, tensor_t lm_head, tensor_t all_logits, tensor_t x_g, tensor_t lm_head_g) {
    br_assert(self.get() == this, "can't be here!");

    auto result = impl()->op_loss_backward(self, ids, mask, lm_head, all_logits, x_g, lm_head_g);
    if ( result.index() == 0) {
        ComputingReturn ret = std::get<0>(result);
        op_check(ret, "loss_backward");
    }
    return result;
}

ComputingReturn TensorType::op_layernorm_backward(tensor_t self, tensor_t scale, tensor_t bias, tensor_t var, tensor_t y, tensor_t dscale, tensor_t dbias, tensor_t din, float eps) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_layernorm_backward(self, scale, bias, var, y, dscale, dbias, din, eps);
    op_check(ret, "layernorm_backward");
}

ComputingReturn TensorType::op_rmsnorm_backward(tensor_t self, tensor_t x, tensor_t scale, tensor_t norm2, tensor_t dscale, tensor_t dx, float eps) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_rmsnorm_backward(self, x, scale, norm2, dscale, dx, eps);
    op_check(ret, "rmsnorm_backward");
}

ComputingReturn TensorType::op_linear_backward(tensor_t self, tensor_t x, tensor_t weight, tensor_t bias, tensor_t x_g, tensor_t weight_g, tensor_t bias_g ) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_linear_backward(self, x, weight, bias, x_g, weight_g, bias_g);
    op_check(ret, "linear_backward");
}

ComputingReturn TensorType::op_gelu_backward(tensor_t self, tensor_t x, tensor_t x_g) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_gelu_backward(self, x, x_g);
    op_check(ret, "gelu_backward");
}

ComputingReturn TensorType::op_attn_backward(tensor_t self, tensor_t attn, tensor_t v, tensor_t attn_g, tensor_t v_g) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_attn_backward(self, attn, v, attn_g, v_g);
    op_check(ret, "gelu_attn_backward");
}

ComputingReturn TensorType::op_softmax_backward(tensor_t self, tensor_t out, tensor_t x_g) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_softmax_backward(self, out, x_g);
    op_check(ret, "gelu_softmax_backward");
}

ComputingReturn TensorType::op_softmax_attn_backward(tensor_t self, tensor_t attn, tensor_t v, tensor_t attn_g, tensor_t v_g) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_softmax_attn_backward(self, attn, v, attn_g, v_g);
    op_check(ret, "gelu_softmax_attn_backward");
}

ComputingReturn TensorType::op_qk_backward(tensor_t self, tensor_t q, tensor_t k, tensor_t q_g, tensor_t k_g) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->op_qk_backward(self, q, k, q_g, k_g);
    op_check(ret, "gelu_qk_backward");
}

ComputingReturn TensorType::io_load(tensor_t self, const char* fileName) {
    br_assert(this == self.get() , "can't be here!");
    auto ret = impl()->io_load(self, fileName);
    op_check(ret, "load");
}

ComputingReturn TensorType::io_save(tensor_t self, const char* fileName) {
    br_assert(this == self.get() , "can't be here!");
    auto ret = impl()->io_save(self, fileName);
    op_check(ret, "save");
}

ComputingReturn TensorType::io_dump(tensor_t self) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->io_dump(self);
    op_check(ret, "dump");
}

ComputingReturn TensorType::io_mpi_bcast(tensor_t self, int root) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->io_mpi_bcast(self, root);
    op_check(ret, "mpi_bcast");
}

ComputingReturn TensorType::io_mpi_recv(tensor_t self, int source) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->io_mpi_recv(self, source);
    op_check(ret, "mpi_recv");
}

ComputingReturn TensorType::io_mpi_send(tensor_t self, int dst) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->io_mpi_send(self, dst);
    op_check(ret, "mpi_send");
}

ComputingReturn TensorType::io_nccl_recv(tensor_t self, int source) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->io_nccl_recv(self, source);
    op_check(ret, "nccl_recv");
}

ComputingReturn TensorType::io_nccl_send(tensor_t self, int dst) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->io_nccl_send(self, dst);
    op_check(ret, "nccl_send");
}

ComputingReturn TensorType::io_pipe_read(tensor_t self) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->io_pipe_read(self);
    op_check(ret, "pipe_read");
}

ComputingReturn TensorType::io_pipe_write(tensor_t self, int n) {
    br_assert(self.get() == this, "can't be here!");
    auto ret = impl()->io_pipe_write(self, n);
    op_check(ret, "pipe_write");
}


TensorType::~TensorType() {
    if ( impl_index() == ImplType::CUDA_FLOAT ) {
        cuda_float_t* tensor = std::get<CUDA_FLOAT>(impl_);
        delete tensor;
    }
    if ( impl_index() == ImplType::CUDA_FP16 ) {
        cuda_fp16_t* tensor = std::get<CUDA_FP16>(impl_);
        delete tensor;
    }
    if ( impl_index() == ImplType::CPU_FLOAT ) {
        cpu_float_t* tensor = std::get<CPU_FLOAT>(impl_);
        delete tensor;
    }
}

TransformerComputing* TensorType::impl() {
    if ( impl_index() == ImplType::CUDA_FLOAT ) {
        cuda_float_t* tensor = std::get<CUDA_FLOAT>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::CUDA_INT ) {
        cuda_int_t* tensor = std::get<CUDA_INT>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::CUDA_FP16 ) {
        cuda_fp16_t* tensor = std::get<CUDA_FP16>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::CPU_FLOAT ) {
        cpu_float_t* tensor = std::get<CPU_FLOAT>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::CPU_INT ) {
        cpu_int_t* tensor = std::get<CPU_INT>(impl_);
        return tensor;
    }
    if ( impl_index() == ImplType::CPU_FP16 ) {
        cpu_fp16_t* tensor = std::get<CPU_FP16>(impl_);
        return tensor;
    }
    br_panic("Can't be here!");
    return nullptr;
}



}

