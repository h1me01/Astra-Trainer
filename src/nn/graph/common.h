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

    AddUnary,
    SubUnary,
    MulUnary,
    DivUnary,

    AddBinary,
    SubBinary,
    MulBinary,
    DivBinary,

    COUNT
};

namespace OpTypeFlags {
constexpr uint8_t Activation = 1 << 0;
constexpr uint8_t Unary = 1 << 1;
constexpr uint8_t ElemWise = 1 << 2;
} // namespace OpTypeFlags

struct OpInfo {
    const char* name;
    uint8_t flags;
};

// clang-format off
constexpr OpInfo OP_INFO[] = {
    {"None",           0},
    {"Input",          0},
    {"SparseAffine",   0},
    {"Affine",         0},
    {"Concat",         0},
    {"Select",         0},
    {"PairwiseMul",    0},
    {"ReLU",           OpTypeFlags::Activation | OpTypeFlags::Unary},
    {"ClippedReLU",    OpTypeFlags::Activation | OpTypeFlags::Unary},
    {"SqrClippedReLU", OpTypeFlags::Activation | OpTypeFlags::Unary},
    {"Sigmoid",        OpTypeFlags::Activation | OpTypeFlags::Unary},
    {"AddUnary",       OpTypeFlags::Unary},
    {"SubUnary",       OpTypeFlags::Unary},
    {"MulUnary",       OpTypeFlags::Unary},
    {"DivUnary",       OpTypeFlags::Unary},
    {"AddBinary",      OpTypeFlags::ElemWise},
    {"SubBinary",      OpTypeFlags::ElemWise},
    {"MulBinary",      OpTypeFlags::ElemWise},
    {"DivBinary",      OpTypeFlags::ElemWise},
};
// clang-format on

static_assert(std::size(OP_INFO) == (size_t)OpType::COUNT);

inline const OpInfo& op_info(OpType t) {
    return OP_INFO[(size_t)t];
}

inline bool is_activation(OpType t) {
    return op_info(t).flags & OpTypeFlags::Activation;
}

inline bool is_unary(OpType t) {
    return op_info(t).flags & OpTypeFlags::Unary;
}

inline bool is_elemwise(OpType t) {
    return op_info(t).flags & OpTypeFlags::ElemWise;
}

inline const char* op_type_str(OpType t) {
    return op_info(t).name;
}

} // namespace nn::graph
