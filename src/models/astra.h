#pragma once

#include "../model/include.h"

namespace model {

using namespace graph;

constexpr std::array<int, 64> input_bucket = {
    0, 1, 2, 3, 3, 2, 1, 0, //
    4, 5, 6, 7, 7, 6, 5, 4, //
    8, 8, 8, 8, 8, 8, 8, 8, //
    9, 9, 9, 9, 9, 9, 9, 9, //
    9, 9, 9, 9, 9, 9, 9, 9, //
    9, 9, 9, 9, 9, 9, 9, 9, //
    9, 9, 9, 9, 9, 9, 9, 9, //
    9, 9, 9, 9, 9, 9, 9, 9, //
};

constexpr int MAX_ACTIVE_FEATURES = 32;
constexpr float EVAL_SCALE = 400.0f;

constexpr int feature_index(PieceType pt, Color pc, Square psq, Square ksq, Color view) {
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

struct Astra : Model {
    Astra() {
        name = "astra_model";

        config.epochs = 100;
        config.batch_size = 16384;
        config.batches_per_epoch = 6104;
        config.save_rate = 20;
        config.thread_count = 2;
    }

    void fill_inputs(const std::vector<TrainingDataEntry>& ds) override {
        auto& stm_features = get_inputs()[0]->get_indices();
        auto& nstm_features = get_inputs()[1]->get_indices();

        for (size_t i = 0; i < ds.size(); i++) {
            const Position& pos = ds[i].pos;

            const Color stm = pos.sideToMove();
            const Square ksq_stm = pos.kingSquare(stm);
            const Square ksq_nstm = pos.kingSquare(!stm);

            const int offset = i * MAX_ACTIVE_FEATURES;

            Bitboard pieces = pos.piecesBB();

            int count = 0;
            for (auto sq : pieces) {
                Piece p = pos.pieceAt(sq);

                int idx = offset + count;
                stm_features(idx) = feature_index(p.type(), p.color(), sq, ksq_stm, stm);
                nstm_features(idx) = feature_index(p.type(), p.color(), sq, ksq_nstm, !stm);

                count++;
            }

            if (count < MAX_ACTIVE_FEATURES) {
                for (int i = count; i < MAX_ACTIVE_FEATURES; i++) {
                    int idx = offset + i;
                    stm_features(idx) = -1;
                    nstm_features(idx) = -1;
                }
            }

            float score_target = 1.0f / (1.0f + expf(-float(ds[i].score) / EVAL_SCALE));
            float wdl_target = (ds[i].result + 1) / 2.0f;

            targets(i) = wdl_sched->get() * score_target + (1.0f - wdl_sched->get()) * wdl_target;
        }
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

    float predict(std::string fen) override { return Model::predict(fen) * EVAL_SCALE; }

    Node build() {
        const int ft_size = 1024;
        const int l1_size = 16;
        const int l2_size = 32;
        const int bucket_count = 8;

        // create layers
        auto ft = sparse_affine(num_buckets(input_bucket) * 768, ft_size).factorized();
        auto l1 = affine(ft_size, l1_size * bucket_count);
        auto l2 = affine(l1_size, l2_size * bucket_count);
        auto l3 = affine(l2_size, bucket_count);

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
        auto stm_in = create_input(MAX_ACTIVE_FEATURES);
        auto nstm_in = create_input(MAX_ACTIVE_FEATURES);

        auto ft_stm = ft(stm_in).clipped_relu().pairwise_mul();
        auto ft_nstm = ft(nstm_in).clipped_relu().pairwise_mul();

        auto cat_ft = concat({ft_stm, ft_nstm});

        auto l1_out = l1(cat_ft).select(bucket_index).clipped_relu();
        auto l2_out = l2(l1_out).select(bucket_index).clipped_relu();
        auto l3_out = l3(l2_out).select(bucket_index);

        return l3_out;
    }

    Loss get_loss() override { return loss::mse(ActivationType::Sigmoid); }

    OptimHandle get_optim() override { return optim::adamw(0.9, 0.999, 0.01).clamp(-0.99, 0.99); }

    LRScheduler get_lr_scheduler() override {
        float lr = 0.001;
        return lr_sched::cosine_annealing(lr, lr * 0.3 * 0.3 * 0.3, config.epochs);
    }

    WDLScheduler get_wdl_scheduler() override { return wdl_sched::constant(0.5); }

    std::vector<std::string> get_training_files() override {
        return {
            "/home/h1me/Downloads/data.binpack",
        };
    }
};

} // namespace model
