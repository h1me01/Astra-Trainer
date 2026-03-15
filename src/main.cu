#include "trainer/include.h"

using namespace trainer;
using namespace graph;

int feature_index(Piece pc, Square psq, Color view) {
    if (view == Color::Black)
        psq.flipVertically();

    return int(psq) + int(pc.type()) * 64 + (int(pc.color()) != int(view)) * 64 * 6;
}

int main() {
    const float lr = 0.001f;
    const float eval_scale = 400.0f;
    const int epochs = 100;
    const int ft_size = 1024;
    const int l1_size = 16;
    const int l2_size = 32;
    const int bucket_count = 8;

    Model model;

    model.set_graph([]() {
        auto ft = sparse_affine(768, ft_size);
        auto l1 = affine(ft_size, l1_size * bucket_count);
        auto l2 = affine(l1_size, l2_size * bucket_count);
        auto l3 = affine(l2_size, bucket_count);

        auto bucket_index = select_index_fn(bucket_count, [](const Position& pos) { //
            return (pos.pieceCount() - 2) / 4;
        });

        ft.weights_format().type(save_format::int16).scale(255);
        ft.biases_format().type(save_format::int16).scale(255);
        l1.weights_format().type(save_format::int8).scale(64).transpose();
        l2.weights_format().transpose();
        l3.weights_format().transpose();

        auto stm_in = create_input(32);
        auto nstm_in = create_input(32);

        auto ft_stm = ft(stm_in).clipped_relu().pairwise_mul();
        auto ft_nstm = ft(nstm_in).clipped_relu().pairwise_mul();

        auto l1_out = l1(concat({ft_stm, ft_nstm})).select(bucket_index).clipped_relu();
        auto l2_out = l2(l1_out).select(bucket_index).clipped_relu();

        return l3(l2_out).select(bucket_index);
    });

    model.set_input_filler([](const auto& batch, auto& inputs) {
        auto& stm_in = inputs[0];
        auto& nstm_in = inputs[1];

        for (size_t i = 0; i < batch.size(); i++) {
            const Position& pos = batch[i].pos;
            const Color stm = pos.sideToMove();

            int j = 0;
            for (auto sq : pos.piecesBB()) {
                Piece pc = pos.pieceAt(sq);
                stm_in(j, i) = feature_index(pc, sq, stm);
                nstm_in(j, i) = feature_index(pc, sq, !stm);
                j++;
            }
        }
    });

    for (auto& p : model.params())
        p->set_bounds(-0.99f, 0.99f);

    TrainingConfig cfg{
        .name = "astra",
        .epochs = epochs,
        .batch_size = 16384,
        .batches_per_epoch = 6104,
        .eval_scale = eval_scale,
    };

    auto dataloader = dataloader::create(2, {"/home/h1me/Downloads/data.binpack"}, [](const TrainingDataEntry& e) {
        return std::abs(e.score) > 10000 //
               || e.isInCheck()          //
               || e.isCapturingMove()    //
               || e.move.type != MoveType::Normal;
    });

    Trainer trainer({
        .model = model,
        .config = cfg,
        .loss = loss::mse(ActivationType::Sigmoid),
        .optim = optim::adamw(0.9f, 0.999f, 0.01f),
        .lr_sched = lr_sched::cosine_annealing(lr, lr * std::pow(0.3f, 3), epochs),
        .wdl_sched = wdl_sched::constant(0.7f),
        .dataloader = dataloader,
    });

    trainer.fit();

    std::cout << "startpos eval: " << model.predict(Position::startPosition().fen()) * eval_scale << std::endl;
}
