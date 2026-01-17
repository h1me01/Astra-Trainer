#pragma once

#include "model.h"

namespace model {

constexpr int early_fen_skipping = 8;

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

inline bool filter_entry(const TrainingDataEntry& e) {
    static constexpr int VALUE_NONE = 32002;

    auto do_wld_skip = [&]() {
        std::bernoulli_distribution distrib(1.0 - e.score_result_prob());
        auto& prng = rng::get_thread_local_rng();
        return distrib(prng);
    };

    if (e.score == VALUE_NONE)
        return true;
    if (e.ply <= early_fen_skipping)
        return true;
    if (e.isCapturingMove() || e.isInCheck())
        return true;
    if (do_wld_skip())
        return true;

    return false;
};

struct Astra : Model {
    Astra(std::string name)
        : Model(name) {
        params.epochs = 100;
        params.batch_size = 16384;
        params.batches_per_epoch = 6104;
        params.save_rate = 10;
        params.thread_count = 4;
        params.lr = 0.001;
        params.eval_div = 400.0;
        params.lambda_start = 0.7;
        params.lambda_end = 0.7;
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

    Ptr<Layer> build(const Ptr<Input>& stm_in, const Ptr<Input>& nstm_in) override {
        const int FT_SIZE = 1024;
        const int L1_SIZE = 16;
        const int L2_SIZE = 32;

        // create layers
        auto ft = make<FeatureTransformer>(num_buckets(input_bucket) * 768, FT_SIZE);
        auto l1 = make<Affine>(FT_SIZE, L1_SIZE);
        auto l2 = make<Affine>(L1_SIZE, L2_SIZE);
        auto l3 = make<Affine>(L2_SIZE, 1);

        // set quantization scheme
        ft->get_weights().quant_type(QuantType::INT16).quant_scale(255);
        ft->get_biases().quant_type(QuantType::INT16).quant_scale(255);

        l1->get_weights().quant_type(QuantType::INT8).quant_scale(64).transpose();
        l2->get_weights().transpose();
        l3->get_weights().transpose();

        // connect layers
        auto ft_stm = ft->forward(stm_in)->crelu();
        auto ft_nstm = ft->forward(nstm_in)->crelu();

        auto pwm_out = make<PairwiseMul>(ft_stm, ft_nstm);

        auto l1_out = l1->forward(pwm_out)->crelu();
        auto l2_out = l2->forward(l1_out)->crelu();
        auto l3_out = l3->forward(l2_out);

        return l3_out;
    }

    Ptr<Loss> get_loss() override {
        return make<MPE>(Activation::Sigmoid, 2.5);
    }

    Ptr<Optimizer> get_optim() override {
        auto optim = make<Adam>(0.9, 0.999, 1e-8, 0.01);
        optim->clamp(-0.99, 0.99);
        return optim;
    }

    Ptr<LRScheduler> get_lr_scheduler() override {
        return make<CosineAnnealing>(params.epochs, params.lr, params.lr * 0.3 * 0.3 * 0.3);
    }

    Ptr<Dataloader> get_dataloader() override {
        return make<Dataloader>( //
            params.batch_size,
            params.thread_count,
            files_from_paths({"/home/h1me/Documents/Coding/Astra-Data/training_data"}),
            filter_entry
        );
    }
};

} // namespace model
