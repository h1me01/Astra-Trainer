#pragma once

#include "../nn/include.h"
#include "model.h"

namespace model {

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

struct Astra : Model {
    Astra(std::string name) : Model(name) {
        params.epochs = 300;
        params.batch_size = 16384;
        params.batches_per_epoch = 6104;
        params.save_rate = 100;
        params.thread_count = 2;
        params.lr = 0.001;
        params.eval_div = 400.0;
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
        const int FT_SIZE = 512;
        const int OUTPUT_BUCKETS = 8;

        const int FT_QUANT = 255;
        const int L1_QUANT = 64;

        // create layers
        auto ft = make<FeatureTransformer>(12 * 768, FT_SIZE);
        auto l1 = make<Affine>(2 * FT_SIZE, OUTPUT_BUCKETS);

        // set quantization scheme
        ft->get_params()[0]->quantize(QuantType::INT16, FT_QUANT); // weights
        ft->get_params()[1]->quantize(QuantType::INT16, FT_QUANT); // biases

        l1->get_params()[0]->quantize(QuantType::INT16, L1_QUANT, true);      // weights (transposed)
        l1->get_params()[1]->quantize(QuantType::INT16, L1_QUANT * FT_QUANT); // biases

        // connect layers
        auto stm_l0 = ft->forward(stm_in)->screlu();
        auto nstm_l0 = ft->forward(nstm_in)->screlu();
        auto merged_l0 = make<Concat>(stm_l0, nstm_l0);

        auto l1_out = l1->forward(merged_l0);
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
};

} // namespace model
