#pragma once

#include "../model/include.h"

namespace model {

constexpr std::array<int, 64> input_bucket = {
    0, 1, 2, 3, 3, 2, 1, 0, //
    4, 4, 5, 5, 5, 5, 4, 4, //
    6, 6, 6, 6, 6, 6, 6, 6, //
    7, 7, 7, 7, 7, 7, 7, 7, //
    8, 8, 8, 8, 8, 8, 8, 8, //
    8, 8, 8, 8, 8, 8, 8, 8, //
    9, 9, 9, 9, 9, 9, 9, 9, //
    9, 9, 9, 9, 9, 9, 9, 9, //
};

inline int bucket_index(const Position& pos) {
    return (pos.pieceCount() - 2) / 4;
}

struct Astra : Model {
    Astra() {
        name = "astra_model";

        config.epochs = 100;
        config.batch_size = 16384;
        config.batches_per_epoch = 6104;
        config.save_rate = 20;
        config.thread_count = 4;
        config.lr = 0.001;
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

    Ptr<nn::Operation> build(const Ptr<nn::Input>& stm_in, const Ptr<nn::Input>& nstm_in) override {
        const int FT_SIZE = 1024;
        const int L1_SIZE = 16;
        const int L2_SIZE = 32;
        const int OUTPUT_BUCKETS = 8;

        // create params
        auto ft = params::create(num_buckets(input_bucket) * 768, FT_SIZE);
        auto l1 = params::create(FT_SIZE, L1_SIZE * OUTPUT_BUCKETS);
        auto l2 = params::create(L1_SIZE, L2_SIZE * OUTPUT_BUCKETS);
        auto l3 = params::create(L2_SIZE, OUTPUT_BUCKETS);

        // save format
        ft->weights_format().type(save_format::int16).scale(255);
        ft->biases_format().type(save_format::int16).scale(255);

        l1->weights_format().type(save_format::int8).scale(64).transpose();
        l2->weights_format().transpose();
        l3->weights_format().transpose();

        // build network
        auto ft_stm = op::feature_transformer(ft, stm_in)->crelu();
        auto ft_nstm = op::feature_transformer(ft, nstm_in)->crelu();

        auto pwm_out = op::pairwise_mul(ft_stm, ft_nstm);

        auto l1_out = op::select(op::affine(l1, pwm_out), bucket_index)->crelu();
        auto l2_out = op::select(op::affine(l2, l1_out), bucket_index)->crelu();
        auto l3_out = op::select(op::affine(l3, l2_out), bucket_index);

        return l3_out;
    }

    Ptr<nn::Loss> get_loss() override { return loss::mse()->sigmoid(); }

    Ptr<nn::Optimizer> get_optim() override {
        auto optim = optim::adam(0.9, 0.999, 1e-8, 0.01);
        optim->clamp(-0.99, 0.99);
        return optim;
    }

    Ptr<nn::LRScheduler> get_lr_scheduler() override {
        return lr_sched::cosine_annealing(config.epochs, config.lr, config.lr * 0.3 * 0.3 * 0.3);
    }

    std::vector<std::string> get_training_files() override {
        return {
            "/home/h1me/Downloads/data.binpack",
        };
    }
};

} // namespace model
