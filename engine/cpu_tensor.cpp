#include <chrono>

#include "common.hpp"
#include "context.hpp"
#include "cpu_tensor.hpp"
#include "cuda_tensor.hpp"

namespace br {
template <DataType _DTYPE_>
ComputingReturn CPUTensor<_DTYPE_>::op_zero(tensor_t self) {
    memset( mem_, 0, DataType_size(_DTYPE_)  * self->items() );
    return OP_OK;
}

template <DataType _DTYPE_>
ComputingReturn CPUTensor<_DTYPE_>::op_copy(tensor_t self, tensor_t src) {
    if ( src->is_cpu() ) {
        memcpy(data(), src->cpu_float()->data(), self->items() * DataType_size(_DTYPE_) );
        return OP_OK;
    }
    auto stream = ComputingContext::cuda_stream;
    CUDA_CHECK(cudaMemcpyAsync(data(), src->cuda_float()->data(), self->items() * DataType_size(_DTYPE_) , cudaMemcpyDeviceToHost, stream));
    return OP_OK;
}

template <DataType _DTYPE_>
ComputingReturn CPUTensor<_DTYPE_>::op_fill(tensor_t self, float value) {
    if ( _DTYPE_ == DataType::Float ) {
        float* dst = (float *)data();
        for (size_t i = 0; i < self->items(); i++) {
            dst[i] = value;
        }
        return OP_OK;
    }
    if ( _DTYPE_ == DataType::Int ) {
        int32_t* dst = (int32_t *)data();
        int32_t v = (int32_t)value;
        for (size_t i = 0; i < self->items(); i++) {
            dst[i] = v;
        }
        return OP_OK;
    }
    if ( _DTYPE_ == DataType::FP16 ) {
        local_fp16* dst = (local_fp16 *)data();
        local_fp16 v = fp32_to_fp16(value);
        for (size_t i = 0; i < self->items(); i++) {
            dst[i] = v;
        }
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
std::variant<ComputingReturn, tensor_t> CPUTensor<_DTYPE_>::op_view(tensor_t self, size_t offset, const std::vector<size_t>& newShape_) {
    if ( _DTYPE_ == DataType::Float ) {
        ShapeType newShape(newShape_);
        float *newData = (float *)data() + offset;
        auto* newCpuTensor = new CPUTensor<DataType::Float>(newData);
        return std::make_shared<TensorType>(newCpuTensor, newShape);
    }
    if ( _DTYPE_ == DataType::Int ) {
        ShapeType newShape(newShape_);
        int *newData = (int *)data() + offset;
        auto* newCpuTensor = new CPUTensor<DataType::Int>(newData);
        return std::make_shared<TensorType>(newCpuTensor, newShape);
    }
    if ( _DTYPE_ == DataType::FP16 ) {
        ShapeType newShape(newShape_);
        local_fp16 *newData = (local_fp16 *)data() + offset;
        auto* newCpuTensor = new CPUTensor<DataType::FP16>(newData);
        return std::make_shared<TensorType>(newCpuTensor, newShape);
    }
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn CPUTensor<_DTYPE_>::op_embed(tensor_t self, tensor_t table, tensor_t outspace) {
    size_t batch = self->shape()[0];
    size_t len = self->shape()[1];
    size_t hidden = table->shape()[1];

    int* text = (int *)data();

    if ( table->dtype() == DataType::Float ) {
        float* from = (float *)table->cpu_float()->data();
        float* out = (float *)outspace->cpu_float()->data();
        for (size_t i = 0; i < batch*len; i++) {
            int id = text[i];
            memcpy(out, from + hidden * id, hidden * sizeof(float) );
            out += hidden;
        }
        return OP_OK;
    }
    if ( table->dtype() == DataType::FP16 ) {
        local_fp16* from = (local_fp16 *)table->cpu_fp16()->data();
        local_fp16* out = (local_fp16 *)outspace->cpu_fp16()->data();
        for (size_t i = 0; i < batch*len; i++) {
            int id = text[i];
            memcpy(out, from + hidden * id, hidden * sizeof(local_fp16) );
            out += hidden;
        }
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn CPUTensor<_DTYPE_>::io_dump(tensor_t self) {
    size_t first8 = std::min(self->shape().vec().back(), (size_t)8);
    if ( _DTYPE_ == DataType::Float ) {
        float* d = (float *)data();
        std::cout << "--------------------------" << std::endl;
        std::cout << "First " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << d[i] << " ";
        }
        std::cout << std::endl;
        d = (float *)data() + self->items() - first8;
        std::cout << "Last " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << d[i] << " ";
        }
        std::cout << std::endl;

        return OP_OK;
    }
    if ( _DTYPE_ == DataType::Int ) {
        int* d = (int *)data();
        std::cout << "--------------------------" << std::endl;
        std::cout << "First " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << d[i] << " ";
        }
        std::cout << std::endl;
        d = (int *)data() + self->items() - first8;
        std::cout << "Last " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << d[i] << " ";
        }
        std::cout << std::endl;

        return OP_OK;
    }
    if ( _DTYPE_ == DataType::FP16 ) {
        local_fp16* d = (local_fp16 *)data();
        std::cout << "--------------------------" << std::endl;
        std::cout << "First " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << fp16_to_fp32(d[i]) << " ";
        }
        std::cout << std::endl;
        d = (local_fp16 *)data() + self->items() - first8;
        std::cout << "Last " << first8 << " : ";
        for(size_t i = 0; i < first8; i++) {
            std::cout << fp16_to_fp32(d[i]) << " ";
        }
        std::cout << std::endl;

        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn CPUTensor<_DTYPE_>::io_save(tensor_t self, const char* fileName) {
    // TODO
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn CPUTensor<_DTYPE_>::io_load(tensor_t self, const char* fileName) {
    if ( _DTYPE_ == DataType::Float ) {
        const size_t count = 1024;
        std::ifstream inf(fileName, std::ios::binary);
        if ( ! inf.is_open() ) {
            std::cout << "Can't open " << fileName << std::endl;
            br_panic("Can't open file");
        }

        br_assert( self->items() % count == 0, "Only support block read");
        for(size_t i = 0; i < self->items() / count; i++) {
            float * src = (float *)data() + i * count;
            inf.read( (char *)src , sizeof(float) * count);
        }
        inf.close();
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn CPUTensor<_DTYPE_>::io_mpi_recv(tensor_t self, int source) {
    if ( _DTYPE_ == DataType::Float ) {
        MPI_Recv(data(), self->items(), MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        return OP_OK;
    }
    if ( _DTYPE_ == DataType::Int ) {
        MPI_Recv(data(), self->items(), MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

template <DataType _DTYPE_>
ComputingReturn CPUTensor<_DTYPE_>::io_mpi_bcast(tensor_t self, int root) {
    if ( _DTYPE_ == DataType::Float ) {
        MPI_Bcast(data(), self->items(), MPI_FLOAT, root, MPI_COMM_WORLD);
        return OP_OK;
    }
    if ( _DTYPE_ == DataType::Int ) {
        MPI_Bcast(data(), self->items(), MPI_INT, root, MPI_COMM_WORLD);
        return OP_OK;
    }
    return OP_TODO_ERROR;
}

tensor_t create_cpu_float(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    CPUTensor<DataType::Float>* tensor = new CPUTensor<DataType::Float>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_cpu_fp16(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    CPUTensor<DataType::FP16>* tensor = new CPUTensor<DataType::FP16>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

tensor_t create_cpu_int(std::vector<size_t>& shape_) {
    ShapeType shape(shape_);
    CPUTensor<DataType::Int>* tensor = new CPUTensor<DataType::Int>(shape);
    return std::make_shared<TensorType>(tensor, shape);
}

}
