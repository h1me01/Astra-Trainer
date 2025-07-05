#include "nn/network.h"

int main() {

    // NETWORK

    Network network({
        100,   // epochs
        16384, // batch size
        6104,  // batches per epoch
        100,   // save rate
        1,     // thread count for dataloader
        400,   // output scalar
        0.7,   // wdl start lambda
        0.8    // wdl end lambda
    });

    // LOSS

    MPELoss<Sigmoid> loss(2.5); // 2.5 = power
    network.set_loss(&loss);

    // OPTIMIZER

    Adam optim({
        0.001, // lr
        0.9,   // beta1
        0.999, // beta2
        1e-8,  // epsilon
        0.01   // decay
    });
    optim.clamp(-0.99, 0.99); // all weights & biases range

    // LEARNING RATE SCHEDULER

    CosineAnnealing lr_scheduler( //
        100,                      // max epochs
        0.001,                    // lr
        0.001 * 0.3 * 0.3 * 0.3   // final lr
    );
    optim.set_lr_scheduler(&lr_scheduler);

    network.set_optim(&optim);

    // INPUT BUCKET

    std::array<int, 64> input_bucket = {
        0,  1,  2,  3,  3,  2,  1,  0,  //
        4,  5,  6,  7,  7,  6,  5,  4,  //
        8,  8,  9,  9,  9,  9,  8,  8,  //
        10, 10, 10, 10, 10, 10, 10, 10, //
        10, 10, 10, 10, 10, 10, 10, 10, //
        11, 11, 11, 11, 11, 11, 11, 11, //
        11, 11, 11, 11, 11, 11, 11, 11, //
        11, 11, 11, 11, 11, 11, 11, 11, //
    };
    network.set_input_bucket(input_bucket);

    // LAYERS

    auto ft = FeatureTransformer<512, SCReLU>( //
        get_bucket_size(input_bucket) * 768,   // input size
        WeightInitType::Uniform                //
    );

    ft.get_params()[0]->quantize<QuantType::INT16>(255); // weights
    ft.get_params()[1]->quantize<QuantType::INT16>(255); // biases

    auto l1 = Affine<OutputBuckets::NUM_BUCKETS>( //
        &ft,                                      // previous layer
        WeightInitType::Uniform                   //
    );

    l1.get_params()[0]->quantize<QuantType::INT16>(64, true); // weights (transposed)
    l1.get_params()[1]->quantize<QuantType::INT16>(255 * 64); // biases

    auto ob = OutputBuckets(&l1);

    network.set_layers({&ft, &l1, &ob});

    // TRAINING

    const std::string root_path = "D:/Astra-Data";

    network.train(                    //
        root_path + "/training_data", // data path
        root_path + "/nn_output",     // output path
        ""                            // checkpoint from output_path
    );

    // TESTING

    network.evaluate_positions({
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rn1qk2r/ppp1bppp/5n2/3p1bB1/3P4/2N1P3/PP3PPP/R2QKBNR w KQkq - 1 7",
    });
}
