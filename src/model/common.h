#pragma once

#include "../nn/include.h"

namespace model {

using Loss = SPtr<nn::loss::Loss>;
using Optimizer = SPtr<nn::optim::Optimizer>;
using LRScheduler = SPtr<nn::lr_sched::LRScheduler>;
using WDLScheduler = SPtr<nn::wdl_sched::WDLScheduler>;
using Operation = SPtr<nn::op::Operation>;
using SelectIndices = SPtr<nn::op::SelectIndices>;
using Input = SPtr<nn::Input>;

} // namespace model
