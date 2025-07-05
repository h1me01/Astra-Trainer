#include "nn/network.h"

using namespace std;

int main() {
    const string root_path = "D:/Astra-Data";

    // get training data
    vector<string> files = fetch_files_from_path(root_path + "/training_data");

    // init network
    Network network({
        300,   // epochs
        16384, // batch size
        6104,  // batches per epoch
        100,   // save rate
        1,     // thread count for dataloader
        400,   // output scalar
        0.7,   // wdl start lambda
        0.8    // wdl end lambda
    });

    // init loss
    MPELoss<Sigmoid> loss(2.5); // 2.5 = power
    network.set_loss(&loss);

    // init optim
    Adam optim({
        0.001, // lr
        0.9,   // beta1
        0.999, // beta2
        1e-8,  // epsilon
        0.01   // decay
    });

    optim.clamp(-1.99, 1.99); // all weights & biases range [-1.99, 1.99]

    // init lr scheduler
    CosineAnnealing lr_scheduler( //
        300,                      // max epochs
        0.001,                    // lr
        0.001 * 0.3 * 0.3 * 0.3   // final lr
    );

    // StepDecay lr_scheduler( //
    //     100,                // step
    //     0.1f                // gamma
    //);

    optim.set_lr_scheduler(&lr_scheduler);

    network.set_optim(&optim);

    // init input bucket (if needed)
    array<int, 64> input_bucket = {
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

    // init hidden layers

    auto ft = FeatureTransformer<512, SCReLU>( //
        get_bucket_size(input_bucket) * 768,   // input size
        WeightInitType::Uniform                // weight initialization type
    );

    // set quantization for weights and biases
    ft.get_params()[0]->quantize<QuantType::INT16>(255); // weights
    ft.get_params()[1]->quantize<QuantType::INT16>(255); // biases

    auto l1 = Affine<OutputBuckets::NUM_BUCKETS>( //
        &ft,                                      // previous layer
        WeightInitType::Uniform                   // weight initialization type
    );

    // set quantization for weights and biases
    l1.get_params()[0]->quantize<QuantType::INT16>(64, true); // weights (transposed)
    l1.get_params()[1]->quantize<QuantType::INT16>(255 * 64); // biases

    auto ob = OutputBuckets(&l1);

    network.set_hidden_layers({&ft, &l1, &ob});

    // start training
    network.train(               //
        files,                   // training files
        root_path + "/nn_output" // output path
        // ,"training_4/checkpoint-100" // load checkpoint (if needed)
    );

    // test network on some positions
    network.evaluate_positions({
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rn1qk2r/ppp1bppp/5n2/3p1bB1/3P4/2N1P3/PP3PPP/R2QKBNR w KQkq - 1 7",
    });
}
