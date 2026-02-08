#pragma once

#include "../model/include.h"

namespace model {

constexpr std::array<int, 64> input_bucket = {
    0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0, //
};

struct Astra : Model {
    Astra() {
        name = "astra_model";

        config.epochs = 100;
        config.batch_size = 16384;
        config.batches_per_epoch = 6104;
        config.save_rate = 20;
        config.thread_count = 2;
        config.eval_div = 400.0;
        config.lambda_start = 0.5;
        config.lambda_end = 0.5;
    }

    int feature_index(PieceType pt, Color pc, Square psq, Square ksq, Color view) override {
        // if king is on opposite side, flip psq horizontally
        if (ksq.file() > fileD)
            psq.flipHorizontally();

        // relative squares
        if (view == Color::Black) {
            psq.flipVertically();
            ksq.flipVertically();
        }

        return int(psq) + int(pt) * 64 + (int(pc) != int(view)) * 64 * 6 + input_bucket[int(ksq)] * 768;
    }

    bool filter_entry(const TrainingDataEntry& e) override {
        if (std::abs(e.score) >= 32000)
            return true;
        if (e.ply <= 8)
            return true;
        if (e.isCapturingMove() || e.isInCheck())
            return true;

        auto do_wld_skip = [&]() {
            std::bernoulli_distribution distrib(1.0 - e.score_result_prob());
            auto& prng = rng::get_thread_local_rng();
            return distrib(prng);
        };
        if (do_wld_skip())
            return true;

        return false;
    }

    Operation build(const Input stm_in, const Input nstm_in) override {
        using namespace op;

        const int ft_size = 1024;
        const int bucket_count = 8;

        // create layers
        auto ft = feature_transformer(num_buckets(input_bucket) * 768, ft_size);
        auto l1 = affine(ft_size, 16 * bucket_count);
        auto l2 = affine(16, 32 * bucket_count);
        auto l3 = affine(32, bucket_count);

        auto bucket_index = select_indices(bucket_count, [&](const Position& pos) { //
            return (pos.pieceCount() - 2) / 4;
        });

        // save format
        ft.weights_format().type(save_format::int16).scale(255);
        ft.biases_format().type(save_format::int16).scale(255);

        l1.weights_format().type(save_format::int8).scale(64).transpose();
        l2.weights_format().transpose();
        l3.weights_format().transpose();

        // build network
        auto ft_stm = ft(stm_in).crelu();
        auto ft_nstm = ft(nstm_in).crelu();

        auto pwm_out = pairwise_mul(ft_stm, ft_nstm);

        auto l1_out = select(l1(pwm_out), bucket_index).crelu();
        auto l2_out = select(l2(l1_out), bucket_index).crelu();
        auto l3_out = select(l3(l2_out), bucket_index);

        return l3_out;
    }

    Loss get_loss() override { return loss::mse(Activation::Sigmoid); }

    Optimizer get_optim() override { return optim::adamw(0.9, 0.999, 1e-8, 0.01).clamp(-0.99, 0.99); }

    LRScheduler get_lr_scheduler() override {
        float lr = 0.001;
        return lr_sched::cosine_annealing(config.epochs, lr, lr * 0.3 * 0.3 * 0.3);
    }

    std::vector<std::string> get_training_files() override {
        return {
            "/home/h1me/Downloads/data.binpack",
        };
    }
};

} // namespace model
