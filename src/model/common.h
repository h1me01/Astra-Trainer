#pragma once

#include "../nn/include.h"

namespace model {

using Loss = UPtr<nn::loss::Loss>;
using Optimizer = UPtr<nn::optim::Optimizer>;
using LRScheduler = UPtr<nn::lr_sched::LRScheduler>;
using WDLScheduler = UPtr<nn::wdl_sched::WDLScheduler>;
using Input = nn::op::Input*;
using Operation = nn::op::Operation*;
using SelectIndices = SPtr<nn::op::SelectIndices>;
using Dataloader = UPtr<nn::dataloader::Dataloader>;

using Node = SPtr<nn::graph::Node>;
using InputNode = SPtr<nn::graph::InputNode>;

} // namespace model
