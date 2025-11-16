#pragma once

#include "../../misc.h"
#include <filesystem>

namespace nn::utils {

std::vector<std::string> files_from_path(const std::vector<std::string> &paths);

int epoch_from_checkpoint(const std::string &checkpoint_name);

} // namespace nn::utils
