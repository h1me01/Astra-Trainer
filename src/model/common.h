#pragma once

#include "../nn/include.h"

namespace model {

using Loss = SPtr<nn::Loss>;
using Optimizer = SPtr<nn::Optimizer>;
using LRScheduler = SPtr<nn::LRScheduler>;
using Operation = SPtr<nn::Operation>;
using SelectIndices = SPtr<nn::SelectIndices>;
using Input = SPtr<nn::Input>;

} // namespace model
