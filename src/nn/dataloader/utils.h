#pragma once

#include <numeric>

#include "../../misc.h"

namespace nn::dataloader::utils {

inline void validate_files(std::vector<std::string> files) {
    for (const auto& f : files) {
        std::ifstream file(f);
        if (!file.good())
            error("Dataloader: File " + f + " does not exist or is not accessible!");
    }

    std::vector<std::string> non_binpack_files;
    for (const auto& f : files)
        if (!f.ends_with(".binpack"))
            non_binpack_files.push_back(f);

    if (!non_binpack_files.empty()) {
        error(
            "Dataloader: The following files do not have .binpack extension:\n" +
            std::accumulate(
                std::next(non_binpack_files.begin()),
                non_binpack_files.end(),
                non_binpack_files[0],
                [](const std::string& a, const std::string& b) { return a + "\n" + b; }
            )
        );
    }

    if (files.empty())
        error("Dataloader: No training data files provided!");
}

} // namespace nn::dataloader::utils
