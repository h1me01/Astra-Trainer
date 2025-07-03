#include "nn/network.h"

using namespace std;

int main() {
    const string root_path = "D:/Astra-Data";

    // get training data
    vector<string> files = fetch_files_from_path(root_path + "/training_data");

    // init network
    Network network( //
        50,          // epochs
        16384,       // batch size
        6104,        // batches per epoch
        100,         // save rate
        1,           // thread count for dataloader
        400,         // output scalar
        1.0,         // wdl start lambda
        1.0          // wdl end lambda
    );

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
    StepDecay lr_sched( //
        160,            // step size
        0.1             // gamma
    );

    // GradualDecay lr_sched(0.99); // 0.99 = gamma

    // CosineAnnealing lr_sched(   //
    //     network.getBatchSize(), // max epochs
    //     0.001,                  // lr
    //     0.001f * powf(0.3f, 5)  // min lr
    //);

    optim.set_scheduler(&lr_sched);

    network.set_optim(&optim);

    // init king bucket (if needed)
    array<int, 64> king_bucket = {
        0,  1,  2,  3,  3,  2,  1,  0,  //
        4,  5,  6,  7,  7,  6,  5,  4,  //
        8,  8,  9,  9,  9,  9,  8,  8,  //
        10, 10, 10, 10, 10, 10, 10, 10, //
        10, 10, 10, 10, 10, 10, 10, 10, //
        11, 11, 11, 11, 11, 11, 11, 11, //
        11, 11, 11, 11, 11, 11, 11, 11, //
        11, 11, 11, 11, 11, 11, 11, 11, //
    };

    network.set_input_bucket(king_bucket);

    // init hidden layers
    auto ft = DualFeatureTransformer<1536, SCReLU>( //
        get_bucket_size(king_bucket) * 768,         // input size
        WeightInitType::Uniform                     // weight initialization type
    );

    auto l1 = FullyConnected<Bucketed::NUM_BUCKETS>( //
        &ft,                                         // previous layer
        WeightInitType::Uniform                      // weight initialization type
    );

    auto b = Bucketed(&l1);

    network.set_hidden_layers({&ft, &l1, &b});

    // setup quantization scheme
    network.set_quant_scheme([&](FILE *f) {
        const int q1 = 255;
        const int q2 = 64;

        ft.get_params()[0]->quant<int16_t>(f, q1);       // weights
        ft.get_params()[1]->quant<int16_t>(f, q1);       // biases
        l1.get_params()[0]->quant<int16_t>(f, q2, true); // weights
        l1.get_params()[1]->quant<int16_t>(f, q1 * q2);  // biases
    });

    const string output_path = root_path + "/nn_output";

    // load weights only (if needed)
    // network.loadWeights(output_path + "/training_5/checkpoint-final/weights.bin");
    network.train( //
        files,
        output_path
        // ,"training_4/checkpoint-100" // load checkpoint (if needed)
    );

    // test network on some positions
    network.evaluate_positions({
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rn1qk2r/ppp1bppp/5n2/3p1bB1/3P4/2N1P3/PP3PPP/R2QKBNR w KQkq - 1 7",
    });
}
