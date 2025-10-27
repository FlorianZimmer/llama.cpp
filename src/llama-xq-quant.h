#pragma once

#include <cstddef>
#include <cstdint>

namespace llama::xquant {

struct block_spec {
    uint32_t bits;
    uint32_t group_size;
};

struct block_qparams {
    float   scale;
    int16_t zero_point;
};

size_t block_data_bytes(uint32_t bits, size_t n_values);

void quantize_block(
        const float * src,
        size_t        n_values,
        const block_spec & spec,
        uint8_t *     dst,
        block_qparams & qparams);

void dequantize_block(
        const uint8_t * src,
        size_t          n_values,
        const block_spec & spec,
        const block_qparams & qparams,
        float *         dst);

} // namespace llama::xquant
