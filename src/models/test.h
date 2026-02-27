#pragma once

#include "../model/include.h"

namespace model {

using namespace graph;

constexpr int MAX_ACTIVE_FEATURES = 32;

constexpr int feature_index(PieceType pt, Color pc, Square psq, Square ksq, Color view) {
    // relative squares
    if (view == Color::Black) {
        psq.flipVertically();
        ksq.flipVertically();
    }

    return int(psq) + int(pt) * 64 + (int(pc) != int(view)) * 64 * 6;
}

struct Test : Model {
    Test() {
        name = "test_model";

        config.epochs = 3;
        config.batch_size = 64;
        config.batches_per_epoch = 512;
        config.save_rate = 20;
        config.thread_count = 2;
        config.eval_div = 400.0;
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

        return int(psq) + int(pt) * 64 + (int(pc) != int(view)) * 64 * 6;
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

            float score_target = 1.0f / (1.0f + expf(-float(ds[i].score) / config.eval_div));
            float wdl_target = (ds[i].result + 1) / 2.0f;

            targets(i) = wdl_sched->get() * score_target + (1.0f - wdl_sched->get()) * wdl_target;
        }
    }

    Node build() {
        auto ft = sparse_affine(768, 16);
        auto l1 = affine(2 * 16, 1);

        auto stm_in = create_input(32);
        auto nstm_in = create_input(32);

        auto ft_stm = ft(stm_in).sqr_clipped_relu();
        auto ft_nstm = ft(nstm_in).sqr_clipped_relu();

        auto cat_ft = concat({ft_stm, ft_nstm});

        return l1(cat_ft);
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
