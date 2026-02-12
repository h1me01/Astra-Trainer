#pragma once

#include "../../data/include.h"
#include "../../kernel/include.h"
#include "../../misc.h"
#include "../../training_data_format/include.h"
#include "../param/param.h"

namespace nn {

class Input {
  public:
    Input(int size)
        : size(size) {}

    void init(int batch_size) { output = Array<int>(size * batch_size, true); }

    Array<int>& get_output() { return output; }

    const Array<int>& get_output() const { return output; }

    int get_size() const { return size; }

  private:
    int size;
    Array<int> output;
};

class SelectIndices {
  public:
    template <typename Fn>
    SelectIndices(const int num_partitions, Fn&& fn)
        : num_partitions(num_partitions),
          fn(std::forward<Fn>(fn)) {}

    void init(int batch_size) { indices = Array<int>(batch_size, true); }

    void step(const std::vector<TrainingDataEntry>& data_entries) {
        for (int i = 0; i < (int)data_entries.size(); i++) {
            int idx = fn(data_entries[i].pos);
            if (idx < 0)
                error("Index function of Select returned negative index!");
            indices(i) = idx;
        }
        indices.host_to_dev();
    }

    int partitions_size() const { return num_partitions; }

    operator Array<int>&() { return indices; }
    operator const Array<int>&() const { return indices; }

  private:
    int num_partitions;
    Array<int> indices;
    std::function<int(const Position&)> fn;
};

class Operation : public std::enable_shared_from_this<Operation> {
  public:
    virtual ~Operation() = default;

    virtual void init(int batch_size) { output = Tensor(output_dim, batch_size); }

    virtual void forward() = 0;
    virtual void backward() = 0;

    int get_input_dim() const { return input_dim; }
    int get_output_dim() const { return output_dim; }

    virtual Tensor& get_output() { return output; }
    virtual const Tensor& get_output() const { return output; }

    DenseMatrix& get_data() { return output.get_data(); }
    const DenseMatrix& get_data() const { return output.get_data(); }

    DenseMatrix& get_grads() { return output.get_grads(); }

    virtual Ptr<SelectIndices> get_select_indices() const { return nullptr; }

    virtual std::vector<Ptr<Operation>> get_inputs() const { return {}; }

    virtual Ptr<Param> get_param() { return nullptr; }

    std::string get_name() const { return name; }

    void set_activation(Activation act_type) { this->act_type = act_type; }
    Activation get_activation() const { return act_type; }

  protected:
    std::string name = "";

    bool skip = false;
    int input_dim = 0;
    int output_dim = 0;
    Activation act_type = Activation::Linear;

    Tensor output;
};

}; // namespace nn
