#include "pipeline.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>
#include <filesystem>

using json = nlohmann::json;

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --dataset <name> --target <file.pts>"
              << " [--data-root <path>] [--target-points <N>]\n";
}

int main(int argc, char** argv) {
    pipeline::Config cfg;
    cfg.data_root = std::filesystem::current_path() / "data";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) {
            cfg.dataset = argv[++i];
        } else if (arg == "--target" && i + 1 < argc) {
            cfg.target_file = argv[++i];
        } else if (arg == "--data-root" && i + 1 < argc) {
            cfg.data_root = argv[++i];
        } else if (arg == "--target-points" && i + 1 < argc) {
            cfg.target_points = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (cfg.dataset.empty() || cfg.target_file.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    // Progress: emit JSON lines to stderr for FastAPI integration
    auto progress = [](const std::string& stage,
                       const std::string& detail,
                       float prog) {
        json msg;
        msg["stage"] = stage;
        msg["detail"] = detail;
        if (prog >= 0) msg["progress"] = prog;
        std::cerr << msg.dump() << "\n" << std::flush;
    };

    int rc = pipeline::run(cfg, progress);

    if (rc != 0) {
        std::cerr << json({{"stage", "error"}, {"detail", "exit code " + std::to_string(rc)}}).dump() << "\n";
    }

    return rc;
}
