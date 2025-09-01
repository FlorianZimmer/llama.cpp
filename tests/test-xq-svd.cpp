#include "../src/llama-memory-xquant.h"
#include "../src/llama-model.h"

#include <cassert>
#include <cstdio>
#include <fstream>

struct llama_model_stub {
    llm_type                 type = LLM_TYPE_UNKNOWN;
    llm_arch                 arch = LLM_ARCH_UNKNOWN;
    std::string              name;
    llama_hparams            hparams;
    std::vector<llama_layer> layers;
};

struct xq_svd_header {
    char     magic[6];
    uint32_t version;
    uint32_t n_layer;
    uint32_t d_model;
};

int main() {
    llama_model_stub stub;
    stub.hparams.n_layer = 1;
    llama_memory_xquant mem(*reinterpret_cast<llama_model *>(&stub));

    const char    path[] = "tmp.xqsvd";
    std::ofstream fout(path, std::ios::binary);
    xq_svd_header hdr = {
        { 'X', 'Q', 'S', 'V', '1', '\0' },
        1, 1, 0
    };
    fout.write(reinterpret_cast<char *>(&hdr), sizeof(hdr));
    uint32_t rk = 8, rv = 8;
    fout.write(reinterpret_cast<char *>(&rk), sizeof(rk));
    fout.write(reinterpret_cast<char *>(&rv), sizeof(rv));
    fout.close();

    bool ok = mem.load_svd(path, *reinterpret_cast<llama_model *>(&stub));
    std::remove(path);
    return ok ? 0 : 1;
}
