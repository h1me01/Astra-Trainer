#pragma once

#include "../nn/include.h"
#include "model.h"

namespace model {

const int FT_SIZE = 512;
const int OUTPUT_BUCKETS = 8;

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
        // return multi_layer(stm_in, nstm_in);
        // return multi_layer2(stm_in, nstm_in);
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

    Ptr<Layer> multi_layer(const Ptr<Input> &stm_in, const Ptr<Input> &nstm_in) {
        const int L1_SIZE = 16;
        const int L2_SIZE = 32;

        // create layers
        auto ft = make<FeatureTransformer>(12 * 768, FT_SIZE);
        auto l1 = make<Affine>(FT_SIZE, L1_SIZE);
        auto l2 = make<Affine>(L1_SIZE, L2_SIZE);
        auto l3 = make<Affine>(L2_SIZE, 1);

        // set quantization scheme
        ft->get_weights().quant_type(QuantType::INT16).quant_scale(255);
        ft->get_biases().quant_type(QuantType::INT16).quant_scale(255);

        l1->get_weights().quant_type(QuantType::INT16).quant_scale(64).transpose();
        l2->get_weights().quant_type(QuantType::FLOAT).transpose();
        l3->get_weights().quant_type(QuantType::FLOAT).transpose();

        // connect layers
        auto stm_ft = ft->forward(stm_in)->crelu();
        auto nstm_ft = ft->forward(nstm_in)->crelu();

        auto pwm_out = make<PairwiseMul>(stm_ft, nstm_ft);

        auto l1_out = l1->forward(pwm_out)->screlu();
        auto l2_out = l2->forward(l1_out)->screlu();

        auto l3_out = l3->forward(l2_out);

        return l3_out;
    }

    Ptr<Layer> multi_layer2(const Ptr<Input> &stm_in, const Ptr<Input> &nstm_in) {
        const int L1_SIZE = 16;
        const int L2_SIZE = 32;

        // create layers
        auto ft = make<FeatureTransformer>(12 * 768, FT_SIZE);
        auto l1 = make<Affine>(FT_SIZE, L1_SIZE);
        auto l2 = make<Affine>(L1_SIZE, L2_SIZE);
        auto l3 = make<Affine>(L2_SIZE, 1);

        // set quantization scheme
        ft->get_weights().quant_type(QuantType::INT16).quant_scale(255);
        ft->get_biases().quant_type(QuantType::INT16).quant_scale(255);

        l1->get_weights().quant_type(QuantType::INT16).quant_scale(64).transpose();
        l2->get_weights().quant_type(QuantType::FLOAT).transpose();
        l3->get_weights().quant_type(QuantType::FLOAT).transpose();

        // connect layers
        auto stm_ft = ft->forward(stm_in)->crelu();
        auto nstm_ft = ft->forward(nstm_in)->crelu();

        auto stm_pwm = make<PairwiseMul>(stm_ft);
        auto nstm_pwm = make<PairwiseMul>(nstm_ft);
        auto merged_l0 = make<Concat>(stm_pwm, nstm_pwm);

        auto l1_out = l1->forward(merged_l0);
        auto l1_select = make<Select>(l1_out, bucket_index);

        auto l2_out = l2->forward(l1_select->screlu());
        auto l2_select = make<Select>(l2_out, bucket_index);

        auto l3_out = l3->forward(l2_select->screlu());
        auto l3_select = make<Select>(l3_out, bucket_index);

        return l3_select;
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
