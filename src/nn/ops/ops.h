#pragma once

#include "../../data/include.h"
#include "../../kernel/include.h"
#include "../../misc.h"
#include "../../training_data_format/include.h"
#include "../param/param.h"

namespace nn {

class OpTensor {
  public:
    OpTensor() = default;

    OpTensor(int batch_size, int output_dim, Activation act_type)
        : has_act(has_activation(act_type)) {
        linear_out = DenseMatrix(output_dim, batch_size);
        gradients = DenseMatrix(output_dim, batch_size);

        if (has_act)
            activated = DenseMatrix(output_dim, batch_size);
    }

    void clear_grads() {
        if (gradients.size() > 0)
            gradients.clear_dev();
    }

    DenseMatrix& get_output() { return has_act ? activated : linear_out; }
    const DenseMatrix& get_output() const { return has_act ? activated : linear_out; }

    DenseMatrix& get_linear_output() { return linear_out; }
    const DenseMatrix& get_linear_output() const { return linear_out; }

    DenseMatrix& get_gradients() { return gradients; }
    const DenseMatrix& get_gradients() const { return gradients; }

    DenseMatrix& get_activated() { return activated; }
    const DenseMatrix& get_activated() const { return activated; }

  private:
    DenseMatrix linear_out;
    DenseMatrix gradients;
    DenseMatrix activated;
    bool has_act = false;
};

class Operation : public std::enable_shared_from_this<Operation> {
  public:
    virtual ~Operation() = default;

    virtual void init(int batch_size) { tensor_output = OpTensor(batch_size, output_dim, act_type); }

    virtual void step(const std::vector<TrainingDataEntry>& data_entries) {}

    virtual void forward() = 0;
    virtual void backward() = 0;

    virtual void clear_grads() { tensor_output.clear_grads(); }

    Ptr<Operation> relu() {
        act_type = Activation::ReLU;
        return shared_from_this();
    }

    Ptr<Operation> crelu() {
        act_type = Activation::CReLU;
        return shared_from_this();
    }

    Ptr<Operation> screlu() {
        act_type = Activation::SCReLU;
        return shared_from_this();
    }

    Ptr<Operation> sigmoid() {
        act_type = Activation::Sigmoid;
        return shared_from_this();
    }

    int get_input_dim() const { return input_dim; }
    int get_output_dim() const { return output_dim; }

    DenseMatrix& get_output() { return tensor_output.get_output(); }
    const DenseMatrix& get_output() const { return tensor_output.get_output(); }

    DenseMatrix& get_gradients() { return tensor_output.get_gradients(); }

    OpTensor& get_tensor_output() { return tensor_output; }
    const OpTensor& get_tensor_output() const { return tensor_output; }

    virtual std::vector<Ptr<Operation>> get_inputs() const { return {}; }

    virtual Ptr<Param> get_param() { return nullptr; }

  protected:
    int input_dim = 0;
    int output_dim = 0;
    Activation act_type = Activation::Linear;

    OpTensor tensor_output;
};

}; // namespace nn
