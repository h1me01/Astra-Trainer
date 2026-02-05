#pragma once

#include "model.h"

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

        params.epochs = 100;
        params.batch_size = 16384;
        params.batches_per_epoch = 6104;
        params.save_rate = 20;
        params.thread_count = 4;
        params.lr = 0.001;
        params.eval_div = 400.0;
        params.lambda_start = 0.5;
        params.lambda_end = 0.5;
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

    void build(const Ptr<Input>& stm_in, const Ptr<Input>& nstm_in) override {
        const int FT_SIZE = 1024;
        const int L1_SIZE = 16;
        const int L2_SIZE = 32;
        const int OUTPUT_BUCKETS = 8;

        // create params
        auto ft_params = make<Params>(num_buckets(input_bucket) * 768, FT_SIZE);
        auto l1_params = make<Params>(FT_SIZE, L1_SIZE * OUTPUT_BUCKETS);
        auto l2_params = make<Params>(L1_SIZE, L2_SIZE * OUTPUT_BUCKETS);
        auto l3_params = make<Params>(L2_SIZE, OUTPUT_BUCKETS);

        // save format
        ft_params->weights_format().type(SaveFormat::Type::INT16).scale(255);
        ft_params->biases_format().type(SaveFormat::Type::INT16).scale(255);

        l1_params->weights_format().type(SaveFormat::Type::INT8).scale(64).transpose();
        l2_params->weights_format().transpose();
        l3_params->weights_format().transpose();

        // build network
        auto ft_stm = make<FeatureTransformer>(ft_params, stm_in)->crelu();
        auto ft_nstm = make<FeatureTransformer>(ft_params, nstm_in)->crelu();

        auto pwm_out = make<PairwiseMul>(ft_stm, ft_nstm);

        auto l1_out = make<Affine>(l1_params, pwm_out);
        auto select_l1 = make<Select>(l1_out, bucket_index)->crelu();

        auto l2_out = make<Affine>(l2_params, select_l1);
        auto select_l2 = make<Select>(l2_out, bucket_index)->crelu();

        auto l3_out = make<Affine>(l3_params, select_l2);
        auto select_l3 = make<Select>(l3_out, bucket_index);
    }

    Ptr<Loss> get_loss() override {
        return make<MSE>(Activation::Sigmoid);
    }

    Ptr<Optimizer> get_optim() override {
        auto optim = make<Ranger>(0.9, 0.999, 1e-8, 0.01);
        optim->clamp(-0.99, 0.99);
        return optim;
    }

    Ptr<LRScheduler> get_lr_scheduler() override {
        return make<CosineAnnealing>(params.epochs, params.lr, params.lr * 0.3 * 0.3 * 0.3);
    }

    std::vector<std::string> get_training_files() override {
        return {
            "/home/h1me/Downloads/data.binpack",
        };
    }
};

} // namespace model
