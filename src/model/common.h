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

using Node = nn::graph::Node*;
using InputNode = nn::graph::InputNode*;

inline int num_buckets(const std::array<int, 64>& bucket_map) {
    int max_bucket = 0;
    for (int b : bucket_map)
        max_bucket = std::max(max_bucket, b);
    return max_bucket + 1;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

} // namespace model
