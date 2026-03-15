#pragma once

#include "../../data/include.h"
#include "../../kernel/include.h"
#include "../../misc.h"
#include "../../training_data_format/include.h"
#include "../param/param.h"

namespace nn::op {

using namespace param;

class SelectIndices {
  public:
    template <typename Fn>
    SelectIndices(const int num_partitions, Fn&& fn)
        : num_partitions_(num_partitions),
          fn_(std::forward<Fn>(fn)) {
        if (num_partitions <= 0)
            error("SelectIndices: Number of partitions in SelectIndices must be positive!");
    }

    void init(int batch_size) { indices_ = Array<int>(batch_size, true); }

    void step(const std::vector<TrainingDataEntry>& data_entries) {
        for (int i = 0; i < (int)data_entries.size(); i++) {
            int idx = fn_(data_entries[i].pos);
            if (idx < 0 || idx >= num_partitions_)
                error("SelectIndices: Index function of Select returned invalid index!");
            indices_(i) = idx;
        }
        indices_.host_to_dev();
    }

    int partitions_size() const { return num_partitions_; }

    operator Array<int>&() { return indices_; }
    operator const Array<int>&() const { return indices_; }

  private:
    int num_partitions_;
    Array<int> indices_;
    std::function<int(const Position&)> fn_;
};

class Operation {
  public:
    virtual ~Operation() = default;

    virtual void init(int batch_size) { output_ = Tensor(output_dim_, batch_size); }

    virtual void forward() = 0;
    virtual void backward() = 0;

    void zero_grads() { output_.get_grads().clear_dev(); }

    int get_input_dim() const { return input_dim_; }
    int get_output_dim() const { return output_dim_; }

    Tensor& get_output() { return output_; }
    const Tensor& get_output() const { return output_; }

    DenseMatrix& get_data() { return output_.get_data(); }
    const DenseMatrix& get_data() const { return output_.get_data(); }

    DenseMatrix& get_grads() { return output_.get_grads(); }

    virtual std::vector<Operation*> get_inputs() const { return {}; }

    virtual Param* get_param() { return nullptr; }

    std::string get_name() const { return name_; }

    void set_activation(ActivationType act_type) { this->act_type_ = act_type; }
    ActivationType get_activation() const { return act_type_; }

  protected:
    std::string name_ = "";

    int input_dim_ = 0;
    int output_dim_ = 0;
    ActivationType act_type_ = ActivationType::Linear;

    Tensor output_;
};

}; // namespace nn::op
