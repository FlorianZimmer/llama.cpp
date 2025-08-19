#include "ggml.h"
#include "../ggml/src/ggml-quants.h"  // block_q4_0, quantize_row_q4_0_ref, dequantize_row_q4_0
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <cstdio>

int main() {
    const int d = 4096; // multiple of 32
    const int T = 7;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> U(-3.0f, 3.0f);

    std::vector<float> X((size_t)T * d);
    for (auto & v : X) v = U(rng);

    // Use typed storage: block_q4_0 per 32 elements
    const int blck = ggml_blck_size(GGML_TYPE_Q4_0); // typically 32
    assert(d % blck == 0);
    const int nblk_per_row = d / blck;

    // Sanity: bytes computed by ggml match the typed layout
    const int row_size_bytes = (int)ggml_row_size(GGML_TYPE_Q4_0, d);
    assert(row_size_bytes == (int)(sizeof(block_q4_0) * nblk_per_row));

    std::vector<block_q4_0> Q((size_t)T * nblk_per_row); // quantized rows (typed)
    std::vector<float>      Y((size_t)T * d);            // dequantized

    for (int t = 0; t < T; ++t) {
        block_q4_0 * qdst = Q.data() + (size_t)t * (size_t)nblk_per_row;

        // Row-wise quant/dequant (k == d)
        quantize_row_q4_0_ref(&X[(size_t)t * d], qdst, d);
        dequantize_row_q4_0(qdst, &Y[(size_t)t * d], d);
    }

    double se = 0.0, ve = 0.0;
    for (size_t i = 0; i < X.size(); ++i) {
        double e = (double)X[i] - (double)Y[i];
        se += e * e;
        ve += (double)X[i] * (double)X[i];
    }
    const double rmse  = std::sqrt(se / X.size());
    const double nrmse = rmse / std::sqrt(ve / X.size());

    std::printf("RMSE=%.6f NRMSE=%.6f\n", rmse, nrmse);
    // Loose sanity threshold for Q4_0
    assert(nrmse < 0.12);
    return 0;
}