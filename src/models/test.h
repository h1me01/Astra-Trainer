#pragma once

#include "../model/include.h"

namespace model {

using namespace graph;

constexpr int MAX_ACTIVE_FEATURES = 32;
constexpr float EVAL_SCALE = 400.0;
constexpr float LR = 0.001;

constexpr int feature_index(Piece pc, Square psq, Square ksq, Color view) {
    if (view == Color::Black) {
        psq.flipVertically();
        ksq.flipVertically();
    }

    return int(psq) + int(pc.type()) * 64 + (int(pc.color()) != int(view)) * 64 * 6;
}

struct Test : Model {
    Test() {
        name = "test_model";

        config.epochs = 3;
        config.batch_size = 16384;
        config.batches_per_epoch = 4;
        config.save_rate = 20;
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

            int count = 0;
            for (auto sq : pos.piecesBB()) {
                Piece pc = pos.pieceAt(sq);

                int idx = offset + count++;
                stm_features(idx) = feature_index(pc, sq, ksq_stm, stm);
                nstm_features(idx) = feature_index(pc, sq, ksq_nstm, !stm);
            }

            float score_target = sigmoid(ds[i].score / EVAL_SCALE);
            float wdl_target = (ds[i].result + 1) / 2.0f;

            targets(i) = wdl_sched->get() * wdl_target + (1.0f - wdl_sched->get()) * score_target;
        }
    }

    Node build() {
        const int ft_size = 32;
        const int l1_size = 8;
        const int l2_size = 16;
        const int bucket_count = 8;

        auto ft = sparse_affine(768, ft_size);
        auto l1 = affine(ft_size, l1_size * bucket_count);
        auto l2 = affine(l1_size, l2_size * bucket_count);
        auto l3 = affine(l2_size, bucket_count);

        auto bucket_index = select_index_fn(bucket_count, [&](const Position& pos) { //
            return (pos.pieceCount() - 2) / 4;
        });

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

    OptimHandle get_optim() override { return optim::adamw(0.9, 0.999, 0.01).clamp_params(-0.99, 0.99); }

    LRScheduler get_lr_scheduler() override {
        return lr_sched::cosine_annealing(LR, LR * 0.3 * 0.3 * 0.3, config.epochs);
    }

    WDLScheduler get_wdl_scheduler() override { return wdl_sched::constant(0.7); }

    Dataloader get_dataloader() override {
        auto should_skip = [](const TrainingDataEntry& e) {
            return std::abs(e.score) > 10000 //
                   || e.isInCheck()          //
                   || e.isCapturingMove()    //
                   || e.move.type != MoveType::Normal;
        };

        return dataloader::create(4, {"/home/h1me/Downloads/data.binpack"}, should_skip);
    }
};

} // namespace model
