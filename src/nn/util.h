#pragma once

#include "../kernel/include.h"
#include "graph/common.h"

namespace nn::util {

inline kernel::ActOp get_activation_op(graph::OpType act_type) {
    switch (act_type) {
    case graph::OpType::ReLU:
        return kernel::ReLU{};
    case graph::OpType::ClippedReLU:
        return kernel::ClippedReLU{};
    case graph::OpType::SqrClippedReLU:
        return kernel::SqrClippedReLU{};
    case graph::OpType::Sigmoid:
        return kernel::Sigmoid{};
    default:
        return kernel::Linear{};
    }
}

} // namespace nn::util
