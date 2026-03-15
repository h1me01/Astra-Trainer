#pragma once

#include <string>

#include "../../misc.h"

namespace nn::graph {

enum class OpType {
    None,
    Input,

    SparseAffine,
    Affine,

    Concat,
    Select,
    PairwiseMul,

    ReLU,
    ClippedReLU,
    SqrClippedReLU,
    Sigmoid,

    Add,
    Sub,
    Mul,
    Div,
};

inline bool is_activation(OpType t) {
    return t == OpType::ReLU ||           //
           t == OpType::ClippedReLU ||    //
           t == OpType::SqrClippedReLU || //
           t == OpType::Sigmoid;
}

inline bool is_unary(OpType t) {
    return is_activation(t);
}

inline bool is_elemwise(OpType t) {
    return t == OpType::Add || //
           t == OpType::Sub || //
           t == OpType::Mul || //
           t == OpType::Div;
}

inline std::string op_type_str(OpType op_type) {
    switch (op_type) {
    case OpType::None:
        return "None";
    case OpType::Input:
        return "Input";
    case OpType::SparseAffine:
        return "SparseAffine";
    case OpType::Affine:
        return "Affine";
    case OpType::Concat:
        return "Concat";
    case OpType::Select:
        return "Select";
    case OpType::PairwiseMul:
        return "PairwiseMul";
    case OpType::ReLU:
        return "ReLU";
    case OpType::ClippedReLU:
        return "ClippedReLU";
    case OpType::SqrClippedReLU:
        return "SqrClippedReLU";
    case OpType::Sigmoid:
        return "Sigmoid";
    case OpType::Add:
        return "Add";
    case OpType::Sub:
        return "Sub";
    case OpType::Mul:
        return "Mul";
    case OpType::Div:
        return "Div";
    default:
        CHECK(false);
        return "";
    }
}

} // namespace nn::graph
