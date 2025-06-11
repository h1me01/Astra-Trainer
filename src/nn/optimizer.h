#pragma once

#include <string>

#include "../kernel/kernel.h"
#include "data.h"
#include "layers/layer.h"
#include "lrscheduler.h"

class Optimizer {
  protected:
    std::string name = "Optimizer";

    std::vector<Tensor *> tunables{};

    std::vector<Array<float>> momentum{};
    std::vector<Array<float>> velocity{};

    int step = 0;

    float lr;
    float beta1;
    float beta2;
    float eps;

    float decay = 0.0f;

    float min_val = -1;
    float max_val = -1;

    LRScheduler *scheduler = nullptr;

    float getDecay() {
        return 1.0f - lr * decay;
    }

  public:
    Optimizer(float lr, float beta1, float beta2, float eps) //
        : lr(lr), beta1(beta1), beta2(beta2), eps(eps) {}

    void init(std::vector<LayerBase *> layers) {
        for(LayerBase *l : layers)
            for(auto *t : l->getTunables()) {
                if(min_val != -1)
                    t->clamp(min_val, t->max());
                if(max_val != -1)
                    t->clamp(t->min(), max_val);
                tunables.push_back(t);
            }
        initBuffers();
    }

    void updateLR(int epoch) {
        if(scheduler != nullptr)
            lr = scheduler->getLR(epoch, lr);
    }

    void setDecay(float decay) {
        this->decay = decay;
    }

    void clampWeights(float min, float max) {
        this->min_val = min;
        this->max_val = max;
    }

    void setLRScheduler(LRScheduler *scheduler) {
        this->scheduler = scheduler;
    }

    virtual void initBuffers() {
        for(Tensor *t : tunables) {
            int size = t->getValues().size();
            momentum.push_back(Array<float>{size});
            velocity.push_back(Array<float>{size});
        }
    }

    virtual void apply(int batch_size) = 0;

    std::string getInfo() {
        std::stringstream info;
        info << name << "(";
        info << "lr=" << formatNumber(lr);
        info << ", beta1=" << formatNumber(beta1);
        info << ", beta2=" << formatNumber(beta2);
        info << ", eps=" << formatNumber(eps);
        if(decay != 0.0f)
            info << ", decay=" << formatNumber(decay);
        if(scheduler != nullptr)
            info << ", scheduler=" << scheduler->getInfo();
        info << ")";
        return info.str();
    }
};

struct Adam : Optimizer {
    Adam(float lr = 0.001, float beta1 = 0.9, float beta2 = 0.999, float eps = 1e-8)
        : Optimizer(lr, beta1, beta2, eps) {
        name = "Adam";
    }

    void apply(int batch_size) override {
        step++;

        const float _decay = getDecay();
        const float grad_scale = 1.0f / batch_size;

        constexpr int block_size = 1024;

        for(size_t i = 0; i < tunables.size(); i++) {
            DenseMatrix &values = tunables[i]->getValues();
            DenseMatrix &gradients = tunables[i]->getGradients();

            ASSERT(values.devAddress() &&      //
                   gradients.devAddress() &&   //
                   momentum[i].devAddress() && //
                   velocity[i].devAddress());

            dim3 grid(std::ceil((float) values.size() / block_size));

            // clang-format off
            adam_kernel<<<grid, block_size>>>
            (
                values.devAddress(),
                gradients.devAddress(),
                momentum[i].devAddress(),   
                velocity[i].devAddress(),
                lr,
                beta1,
                beta2,
                eps,
                _decay,
                tunables[i]->min(),
                tunables[i]->max(),
                grad_scale,
                values.size()
            );
            // clang-format on
        }
    }
};

class RAdam : public Optimizer {
  private:
    int N_sma_threshold = 5;

  public:
    RAdam(float lr = 0.001, float beta1 = 0.9, float beta2 = 0.999, float eps = 1e-8)
        : Optimizer(lr, beta1, beta2, eps) {
        name = "RAdam";
    }

    void apply(int batch_size) override {
        step++;

        const float _decay = getDecay();
        const float grad_scale = 1.0f / batch_size;

        constexpr int block_size = 1024;

        for(size_t i = 0; i < tunables.size(); i++) {
            DenseMatrix &values = tunables[i]->getValues();
            DenseMatrix &gradients = tunables[i]->getGradients();

            ASSERT(values.devAddress() &&      //
                   gradients.devAddress() &&   //
                   momentum[i].devAddress() && //
                   velocity[i].devAddress());

            dim3 grid(std::ceil((float) values.size() / block_size));

            // clang-format off
            radam_kernel<<<grid, block_size>>>
            (
                values.devAddress(),
                gradients.devAddress(),
                momentum[i].devAddress(),
                velocity[i].devAddress(),
                lr,
                beta1,
                beta2,
                eps,
                _decay,
                tunables[i]->min(),
                tunables[i]->max(),
                grad_scale,
                N_sma_threshold,
                step,
                values.size()
            );
            // clang-format on
        }
    }

    void setN_SMA_Threshold(int N_sma_threshold) {
        this->N_sma_threshold = N_sma_threshold;
    }
};

class Ranger : public Optimizer {
  private:
    float alpha = 0.5;
    int k = 6;
    int N_sma_threshold = 6;

    std::vector<Array<float>> slow_buffer{};

  public:
    Ranger(float lr = 0.001, float beta1 = 0.95, float beta2 = 0.999, float eps = 1e-5)
        : Optimizer(lr, beta1, beta2, eps) {
        name = "Ranger";
    }

    void initBuffers() override {
        for(Tensor *t : tunables) {
            int size = t->getValues().size();
            momentum.push_back(Array<float>{size});
            velocity.push_back(Array<float>{size});
            slow_buffer.push_back(Array<float>{size});
        }
    }

    void apply(int batch_size) override {
        step++;

        const float _decay = getDecay();
        const float grad_scale = 1.0f / batch_size;

        constexpr int block_size = 1024;

        for(size_t i = 0; i < tunables.size(); i++) {
            DenseMatrix &values = tunables[i]->getValues();
            DenseMatrix &gradients = tunables[i]->getGradients();

            ASSERT(values.devAddress() &&      //
                   gradients.devAddress() &&   //
                   momentum[i].devAddress() && //
                   velocity[i].devAddress());

            dim3 grid(std::ceil((float) values.size() / block_size));

            // clang-format off
            ranger_kernel<<<grid, block_size>>>
            (
                values.devAddress(),
                gradients.devAddress(),
                momentum[i].devAddress(),
                velocity[i].devAddress(),
                slow_buffer[i].devAddress(),
                lr,
                beta1,
                beta2,
                eps,
                _decay,
                tunables[i]->min(),
                tunables[i]->max(),
                grad_scale,
                alpha,
                k,
                 N_sma_threshold,
                step,
                values.size()
            );
            // clang-format on
        }
    }

    void setAlpha(float alpha) {
        this->alpha = alpha;
    }

    void setK(int k) {
        this->k = k;
    }

    void setN_SMA_Threshold(int N_sma_threshold) {
        this->N_sma_threshold = N_sma_threshold;
    }
};
