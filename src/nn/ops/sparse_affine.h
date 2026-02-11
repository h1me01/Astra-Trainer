#pragma once

#include "ops.h"

namespace nn {

class SparseAffine : public Operation {
  public:
    SparseAffine(Ptr<Param> params, Ptr<Input> input, bool fuse_pairwise_mul = false)
        : SparseAffine(params, input, nullptr, fuse_pairwise_mul) {
            name = fuse_pairwise_mul ? "sparse_affine_pairwise_mul" : "sparse_affine";
        }

    SparseAffine(Ptr<Param> params, Ptr<Input> input1, Ptr<Input> input2, bool fuse_pairwise_mul = false)
        : params(params),
          fuse_pairwise_mul(fuse_pairwise_mul) {

        name = fuse_pairwise_mul ? "sparse_affine_pairwise_mul_concatenated" : "sparse_affine_concatenated";

        inputs.push_back(input1);
        if (input2)
            inputs.push_back(input2);

        input_dim = params->get_input_dim();

        if (fuse_pairwise_mul) {
            if (params->get_output_dim() % 2 != 0)
                error("SparseAffine with pairwise multiply requires even output dimension!");
            output_dim = (params->get_output_dim() / 2) * inputs.size();
        } else {
            output_dim = inputs.size() * params->get_output_dim();
        }

        if (input_dim % 768 != 0)
            error("SparseAffine input dimension must be a multiple of 768!");
    }

    void forward() override {
        const int input_count = inputs.size();

        if (fuse_pairwise_mul) {
            const int half_output = params->get_output_dim() / 2;
            for (int i = 0; i < input_count; i++) {
                kernel::sparse_affine_pairwise_mul_fwd(
                    params->get_weights().get_data(),
                    params->get_biases().get_data(),
                    output.get_data(),
                    inputs[i]->get_output(),
                    inputs[i]->get_size(),
                    i * half_output,
                    act_type
                );
            }
        } else {
            for (int i = 0; i < input_count; i++) {
                kernel::sparse_affine_fwd(
                    params->get_weights().get_data(),
                    params->get_biases().get_data(),
                    output.get_data(),
                    inputs[i]->get_output(),
                    inputs[i]->get_size(),
                    i * (output_dim / input_count),
                    act_type
                );
            }
        }
    }

    void backward() override {
        const int input_count = inputs.size();

        if (fuse_pairwise_mul) {
            const int half_output = params->get_output_dim() / 2;
            for (int i = 0; i < input_count; i++) {
                kernel::sparse_affine_pairwise_mul_bwd(
                    params->get_weights().get_grads(),
                    params->get_biases().get_grads(),
                    params->get_weights().get_data(),
                    params->get_biases().get_data(),
                    output,
                    inputs[i]->get_output(),
                    inputs[i]->get_size(),
                    i * half_output,
                    act_type
                );
            }
        } else {
            for (int i = 0; i < input_count; i++) {
                kernel::sparse_affine_bwd(
                    params->get_weights().get_grads(),
                    params->get_biases().get_grads(),
                    output,
                    inputs[i]->get_output(),
                    inputs[i]->get_size(),
                    i * (output_dim / input_count),
                    act_type
                );
            }
        }
    }

    Ptr<Param> get_param() override { return params; }

    std::vector<Ptr<Input>> get_inputs_ft() const { return inputs; }

    bool uses_pairwise_mul_fusion() const { return fuse_pairwise_mul; }

  private:
    Ptr<Param> params;
    std::vector<Ptr<Input>> inputs;
    bool fuse_pairwise_mul = false;
};

} // namespace nn
