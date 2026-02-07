#pragma once

#include "../../data/include.h"
#include "../../kernel/include.h"
#include "../../misc.h"
#include "../../training_data_format/include.h"
#include "../param/param.h"

namespace nn {

class Operation : public std::enable_shared_from_this<Operation> {
  public:
    virtual ~Operation() = default;

    virtual void init(int batch_size) { output = Tensor(output_dim, batch_size); }

    virtual void step(const std::vector<TrainingDataEntry>& data_entries) {}

    virtual void forward() = 0;
    virtual void backward() = 0;

    virtual void clear_grads() { get_grads().clear_dev(); }

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

    Tensor& get_output() { return output; }
    const Tensor& get_output() const { return output; }

    DenseMatrix& get_data() { return output.get_data(); }
    const DenseMatrix& get_data() const { return output.get_data(); }

    DenseMatrix& get_grads() { return output.get_grads(); }

    virtual std::vector<Ptr<Operation>> get_inputs() const { return {}; }

    virtual Ptr<Param> get_param() { return nullptr; }

  protected:
    int input_dim = 0;
    int output_dim = 0;
    Activation act_type = Activation::Linear;

    Tensor output;
};

}; // namespace nn
