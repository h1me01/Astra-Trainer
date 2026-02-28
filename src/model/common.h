#pragma once

#include "../nn/include.h"

namespace model {

using Loss = Ptr<nn::loss::Loss>;
using Optimizer = Ptr<nn::optim::Optimizer>;
using LRScheduler = Ptr<nn::lr_sched::LRScheduler>;
using WDLScheduler = Ptr<nn::wdl_sched::WDLScheduler>;
using Input = SPtr<nn::op::Input>;
using Operation = SPtr<nn::op::Operation>;
using SelectIndices = SPtr<nn::op::SelectIndices>;

using Node = SPtr<nn::graph::Node>;
using InputNode = SPtr<nn::graph::InputNode>;

inline int num_buckets(const std::array<int, 64>& bucket_map) {
    int max_bucket = 0;
    for (int b : bucket_map)
        max_bucket = std::max(max_bucket, b);
    return max_bucket + 1;
}

} // namespace model
