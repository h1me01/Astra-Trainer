#pragma once

#include "model.h"

namespace model {

constexpr int early_fen_skipping = 20;
constexpr int random_fen_skipping = 5;

constexpr int FT_SIZE = 1536;
constexpr int OUTPUT_BUCKETS = 8;

constexpr std::array<int, 64> input_bucket = {
    0,  1,  2,  3,  3,  2,  1,  0,  //
    4,  5,  6,  7,  7,  6,  5,  4,  //
    8,  8,  9,  9,  9,  9,  8,  8,  //
    10, 10, 10, 10, 10, 10, 10, 10, //
    10, 10, 10, 10, 10, 10, 10, 10, //
    11, 11, 11, 11, 11, 11, 11, 11, //
    11, 11, 11, 11, 11, 11, 11, 11, //
    11, 11, 11, 11, 11, 11, 11, 11, //
};

inline int bucket_index(const Position &pos) {
    return (pos.pieceCount() - 2) / 4;
}

inline bool filter_entry(const TrainingDataEntry &e) {
    static constexpr int VALUE_NONE = 32002;

    auto do_wld_skip = [&]() {
        std::bernoulli_distribution distrib(1.0 - e.score_result_prob());
        auto &prng = rng::get_thread_local_rng();
        return distrib(prng);
    };

    auto do_skip = [&]() {
        static constexpr double prob = double(random_fen_skipping) / (random_fen_skipping + 1);
        std::bernoulli_distribution distrib(prob);
        auto &prng = rng::get_thread_local_rng();
        return distrib(prng);
    };

    if(e.score == VALUE_NONE)
        return true;
    if(e.ply <= early_fen_skipping)
        return true;
    if(e.isCapturingMove() || e.isInCheck())
        return true;
    if(do_skip())
        return true;
    if(do_wld_skip())
        return true;

    return false;
};

struct Astra : Model {
    Astra(std::string name) : Model(name) {
        params.epochs = 300;
        params.batch_size = 16384;
        params.batches_per_epoch = 6104;
        params.save_rate = 100;
        params.thread_count = 4;
        params.lr = 0.001;
        params.eval_div = 400.0;
        // lambda determines how much the score influences the loss
        // e.g. 1 means full score influence, so wdl has 0 influence
        params.lambda_start = 1.0;
        params.lambda_end = 0.7;
    }

    int feature_index(PieceType pt, Color pc, Square psq, Square ksq, Color view) override {
        // if king is on opposite side, flip psq horizontally
        if(ksq.file() > fileD)
            psq.flipHorizontally();

        // relative squares
        if(view == Color::Black) {
            psq.flipVertically();
            ksq.flipVertically();
        }

        return int(psq) +                        //
               int(pt) * 64 +                    //
               (int(pc) != int(view)) * 64 * 6 + //
               input_bucket[int(ksq)] * 768;
    }

    Ptr<Layer> build(const Ptr<Input> &stm_in, const Ptr<Input> &nstm_in) override {
        return standard(stm_in, nstm_in);
    }

    Ptr<Layer> standard(const Ptr<Input> &stm_in, const Ptr<Input> &nstm_in) {
        // create layers
        auto ft = make<FeatureTransformer>(12 * 768, FT_SIZE);
        auto l1 = make<Affine>(2 * FT_SIZE, OUTPUT_BUCKETS);

        // set quantization scheme
        ft->get_weights().quant_type(QuantType::INT16).quant_scale(255);
        ft->get_biases().quant_type(QuantType::INT16).quant_scale(255);

        l1->get_weights().quant_type(QuantType::INT16).quant_scale(64).transpose();
        l1->get_biases().quant_type(QuantType::INT16).quant_scale(64 * 255);

        // connect layers
        auto ft_out = ft->forward(stm_in, nstm_in)->screlu();

        auto l1_out = l1->forward(ft_out);
        auto l1_select = make<Select>(l1_out, bucket_index);

        return l1_select;
    }

    Ptr<Loss> get_loss() override {
        return make<MPE>(ActivationType::Sigmoid, 2.5);
    }

    Ptr<Optimizer> get_optim() override {
        auto optim = make<Adam>(0.9, 0.999, 1e-8, 0.01);
        optim->clamp(-0.99, 0.99);
        return optim;
    }

    Ptr<LRScheduler> get_lr_scheduler() override {
        return make<CosineAnnealing>(params.epochs, params.lr, 0.000027);
    }

    Ptr<Dataloader> get_dataloader() override {
        return make<Dataloader>( //
            params.batch_size,
            params.thread_count,
            files_from_path({"D:/Astra-Data/training_data"}),
            filter_entry);
    }
};

} // namespace model
