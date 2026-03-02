#pragma once

#include "../nn/include.h"

namespace model {

using Loss = Ptr<nn::loss::Loss>;
using Optimizer = Ptr<nn::optim::Optimizer>;
using LRScheduler = Ptr<nn::lr_sched::LRScheduler>;
using WDLScheduler = Ptr<nn::wdl_sched::WDLScheduler>;
using Input = nn::op::Input*;
using Operation = nn::op::Operation*;
using SelectIndices = nn::op::SelectIndices*;
using Dataloader = Ptr<nn::dataloader::Dataloader>;

using Node = nn::graph::Node*;
using InputNode = nn::graph::InputNode*;

} // namespace model
