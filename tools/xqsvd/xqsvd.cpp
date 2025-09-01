#include "llama.h"

#include <cstdio>
#include <cstring>
#include <fstream>

struct xq_svd_header {
    char     magic[6];
    uint32_t version;
    uint32_t n_layer;
    uint32_t d_model;
};

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <out.xqsvd>\n", argv[0]);
        return 1;
    }

    const char * path = argv[1];
    std::ofstream fout(path, std::ios::binary);
    if (!fout) {
        std::fprintf(stderr, "cannot open %s for writing\n", path);
        return 1;
    }

    xq_svd_header hdr{};
    std::memcpy(hdr.magic, "XQSV1", 6);
    hdr.version = 1;
    hdr.n_layer = 0;
    hdr.d_model = 0;

    fout.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
    std::printf("wrote placeholder SVD file %s\n", path);
    return 0;
}
