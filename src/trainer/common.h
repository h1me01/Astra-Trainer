#pragma once

#include "../nn/include.h"

namespace trainer {

using Loss = SPtr<nn::loss::Loss>;
using Optimizer = SPtr<nn::optim::Optimizer>;
using LRScheduler = SPtr<nn::lr_sched::LRScheduler>;
using WDLScheduler = SPtr<nn::wdl_sched::WDLScheduler>;
using Input = nn::op::Input;
using Operation = nn::op::Operation*;
using SelectIndices = SPtr<nn::op::SelectIndices>;
using Dataloader = SPtr<nn::dataloader::Dataloader>;

using Node = SPtr<nn::graph::Node>;
using InputNode = SPtr<nn::graph::InputNode>;

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline void make_deterministic() {
    setenv("CUBLAS_WORKSPACE_CONFIG", ":4294967296", 1);
    setenv("CUDA_LAUNCH_BLOCKING", "1", 1);
    setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 1);

    external::rng::reset_thread_local_rng();
    rng::reset_tensor_rng();
}

} // namespace trainer
