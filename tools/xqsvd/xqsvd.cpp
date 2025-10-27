#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

void print_usage(const char * prog) {
    std::fprintf(stderr,
        "Usage: %s --model <gguf> --output <xqsvd>\n"
        "\n"
        "Generates XQuant SVD factors for grouped-query attention models.\n"
        "Currently only the CLI skeleton is implemented; the numerical\n"
        "factorization pipeline will be added in a later phase.\n",
        prog);
}

struct options {
    std::string model_path;
    std::string output_path;
};

bool parse_args(int argc, char ** argv, options & out) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
            out.model_path = argv[++i];
        } else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            out.output_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return false;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return false;
        }
    }

    if (out.model_path.empty() || out.output_path.empty()) {
        std::fprintf(stderr, "Missing required --model/--output arguments\n");
        return false;
    }

    return true;
}

} // namespace

int main(int argc, char ** argv) {
    options opts;
    if (!parse_args(argc, argv, opts)) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::fprintf(stderr,
        "xqsvd: model='%s' output='%s' -- implementation pending\n",
        opts.model_path.c_str(),
        opts.output_path.c_str());
    return EXIT_FAILURE;
}
