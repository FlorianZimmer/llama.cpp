#include "llama-memory-xquant.h"

#include "llama-impl.h"

llama_memory_xquant_cl::llama_memory_xquant_cl(const llama_model & model, const llama_cparams & cparams)
    : llama_memory_xquant(model, cparams) {
    LLAMA_LOG_INFO("%s: initialized XQuant-CL scaffolding\n", __func__);
}
