#pragma once

#include "../model/include.h"

namespace model {

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

        //load_params("/home/h1me/Downloads/model.bin");

        train("/home/h1me/Documents/Coding/Astra-Data/nn_output");

        evaluate_positions({
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rn1qk2r/ppp1bppp/5n2/3p1bB1/3P4/2N1P3/PP3PPP/R2QKBNR w KQkq - 1 7",
            "5B2/5ppk/p6p/P7/4b2P/3p1qP1/4rP2/2Q2RK1 w - - 10 33",
            "8/8/pp3p2/3p4/PP1PkPK1/8/8/8 b - - 4 52",
            "rnbqkbnr/p3pppp/8/1p6/2pP4/4P3/1P3PPP/RNBQKBNR w KQkq - 0 6",
        });
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

    Operation build(const Input stm_in, const Input nstm_in) {
        using namespace op;

        const int bucket_count = 8;

        // create layers
        auto ft = sparse_affine(num_buckets(input_bucket) * 768, 1024);
        auto l1 = affine(1024, 16 * bucket_count);
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
        auto ft_stm = ft(stm_in).clamped_relu().pairwise_mul();
        auto ft_nstm = ft(nstm_in).clamped_relu().pairwise_mul();

        auto cat_ft = concat(ft_stm, ft_nstm);

        auto l1_out = l1(cat_ft).select(bucket_index).clamped_relu();
        auto l2_out = l2(l1_out).select(bucket_index).clamped_relu();
        auto l3_out = l3(l2_out).select(bucket_index);

        return l3_out;
    }

    Loss get_loss() override { return loss::mse(Activation::Sigmoid); }

    Optimizer get_optim() override { return optim::adamw(0.9, 0.999, 0.01).clamp(-0.99, 0.99); }

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
