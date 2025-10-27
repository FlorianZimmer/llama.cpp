#include "llama-xq-quant.h"

#include "llama-impl.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace llama::xquant {

namespace {

constexpr float XQ_EPS = 1.0e-12f;

[[nodiscard]] bool is_supported_bits(uint32_t bits) {
    switch (bits) {
        case 2:
        case 3:
        case 4:
        case 8:
            return true;
        default:
            return false;
    }
}

[[nodiscard]] uint32_t bit_mask(uint32_t bits) {
    return (bits >= 32) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
}

void validate_spec(const block_spec & spec) {
    if (!is_supported_bits(spec.bits)) {
        throw std::invalid_argument("XQuant: unsupported bit-width " + std::to_string(spec.bits));
    }
    if (spec.group_size == 0) {
        throw std::invalid_argument("XQuant: group size must be non-zero");
    }
}

void pack_bits(const uint8_t * src, size_t n_values, uint32_t bits, uint8_t * dst) {
    if (bits == 8) {
        std::memcpy(dst, src, n_values);
        return;
    }

    uint32_t acc      = 0;
    uint32_t acc_bits = 0;
    size_t   out_idx  = 0;
    const uint32_t mask = bit_mask(bits);
    const size_t dst_size = block_data_bytes(bits, n_values);
    std::fill(dst, dst + dst_size, 0);

    for (size_t i = 0; i < n_values; ++i) {
        acc |= (uint32_t(src[i]) & mask) << acc_bits;
        acc_bits += bits;
        while (acc_bits >= 8) {
            dst[out_idx++] = uint8_t(acc & 0xFFu);
            acc >>= 8;
            acc_bits -= 8;
        }
    }

    if (acc_bits > 0) {
        dst[out_idx++] = uint8_t(acc & 0xFFu);
    }
}

void unpack_bits(const uint8_t * src, size_t n_values, uint32_t bits, uint8_t * dst) {
    if (bits == 8) {
        std::memcpy(dst, src, n_values);
        return;
    }

    const uint32_t mask = bit_mask(bits);
    uint32_t acc      = 0;
    uint32_t acc_bits = 0;
    size_t   in_idx   = 0;

    for (size_t i = 0; i < n_values; ++i) {
        while (acc_bits < bits) {
            acc |= uint32_t(src[in_idx++]) << acc_bits;
            acc_bits += 8;
        }
        dst[i] = uint8_t(acc & mask);
        acc >>= bits;
        acc_bits -= bits;
    }
}

} // namespace

size_t block_data_bytes(uint32_t bits, size_t n_values) {
    if (!is_supported_bits(bits)) {
        throw std::invalid_argument("XQuant: unsupported bit-width " + std::to_string(bits));
    }
    return (bits * n_values + 7u) / 8u;
}

void quantize_block(
        const float * src,
        size_t        n_values,
        const block_spec & spec,
        uint8_t *     dst,
        block_qparams & qparams) {
    validate_spec(spec);

    if (n_values == 0) {
        qparams.scale = 1.0f;
        qparams.zero_point = 0;
        return;
    }

    float min_val = src[0];
    float max_val = src[0];
    for (size_t i = 1; i < n_values; ++i) {
        min_val = std::min(min_val, src[i]);
        max_val = std::max(max_val, src[i]);
    }

    const uint32_t qmax = bit_mask(spec.bits);
    const float range = max_val - min_val;
    float scale = range <= 0.0f ? 1.0f : range / float(qmax);
    if (!std::isfinite(scale) || scale < XQ_EPS) {
        scale = 1.0f;
    }

    const float zero_point_f = std::nearbyintf(-min_val / scale);
    const int32_t zero_point_i = std::clamp<int32_t>(
            static_cast<int32_t>(zero_point_f),
            0,
            static_cast<int32_t>(qmax));

    qparams.scale = scale;
    qparams.zero_point = static_cast<int16_t>(zero_point_i);

    std::vector<uint8_t> tmp(n_values);
    for (size_t i = 0; i < n_values; ++i) {
        const float qf = std::nearbyintf(src[i] / scale) + static_cast<float>(zero_point_i);
        const int32_t qi = std::clamp<int32_t>(
                static_cast<int32_t>(std::lrintf(qf)),
                0,
                static_cast<int32_t>(qmax));
        tmp[i] = static_cast<uint8_t>(qi);
    }

    pack_bits(tmp.data(), n_values, spec.bits, dst);
}

void dequantize_block(
        const uint8_t * src,
        size_t          n_values,
        const block_spec & spec,
        const block_qparams & qparams,
        float *         dst) {
    validate_spec(spec);

    if (n_values == 0) {
        return;
    }

    std::vector<uint8_t> tmp(n_values);
    unpack_bits(src, n_values, spec.bits, tmp.data());

    for (size_t i = 0; i < n_values; ++i) {
        const int32_t q = static_cast<int32_t>(tmp[i]);
        const float deq = (static_cast<float>(q) - static_cast<float>(qparams.zero_point)) * qparams.scale;
        dst[i] = deq;
    }
}

} // namespace llama::xquant
