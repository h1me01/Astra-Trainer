#include "nn/network.h"

using namespace std;

int main() {
    const string root_path = "D:/Astra-Data";

    cout << "================================= Training Data ================================\n\n";

    // get training data
    vector<string> files = fetchFilesFromPath(root_path + "/training_data");

    // clang-format off
    Network network
    (
        600,   // epochs
        16384, // batch size
        6104,  // batches per epoch
        100,   // save rate
        1,     // thread count for dataloader
        400,   // output scalar
        0.7,   // start lambda
        0.8    // end lambda
    );
    // clang-format on

    // init loss
    MPELoss<Sigmoid> loss(2.5); // 2.5 = power
    network.setLoss(&loss);

    // init optim
    // clang-format off
    Adam optim
    (
        0.001, // lr
        0.9,   // beta1
        0.999, // beta2
        1e-8   // epsilon
    );
    // clang-format on

    // init learning rate scheduler
    // clang-format off
    StepDecay lr_scheduler
    (
        160, // step size
        0.1  // gamma
    );
    // clang-format on

    optim.setDecay(0.01);
    optim.setLRScheduler(&lr_scheduler);
    optim.clamp(-1.99, 1.99); // all weights & biases range [-1.99, 1.99]

    network.setOptimizer(&optim);

    // init king bucket (if needed)
    // clang-format off
    array<int, 64> king_bucket = 
    {
        0, 1, 2, 3, 3, 2, 1, 0,
        4, 5, 6, 7, 7, 6, 5, 4,
        8, 8, 9, 9, 9, 9, 8, 8,
        10,10,10,10,10,10,10,10,
        10,10,10,10,10,10,10,10,
        11,11,11,11,11,11,11,11,
        11,11,11,11,11,11,11,11,
        11,11,11,11,11,11,11,11,
    };
    // clang-format on

    network.setKingBucket(king_bucket);

    // init hidden layers

    // if you don't want to clamp all weights & biases
    // you can do it individually by doing this:
    // - layer.clampWeights(-1.99, 1.99);
    // - layer.clampBiases(-1.99, 1.99);

    auto ft = FeatureTransformer<1536, CReLU>(getBucketSize(king_bucket) * 768);
    auto fc = FullyConnected<1, Linear>(&ft);

    network.setHiddenLayers({&ft, &fc});

    // setup quantization scheme

    network.setQuantizationScheme([&](FILE *f) {
        const int q1 = 255;
        const int q2 = 64;

        ft.getTunables()[0]->quantize<int16_t>(f, q1, true); // weights
        ft.getTunables()[1]->quantize<int16_t>(f, q1);       // biases
        fc.getTunables()[0]->quantize<int16_t>(f, q2);       // weights
        fc.getTunables()[1]->quantize<int16_t>(f, q1 * q2);  // biases
    });

    const string output_path = root_path + "/nn_output";

    // load weights only (if needed)
    // network.loadWeights(output_path + "/training_6/checkpoint-100/weights.bin");
    // clang-format off
    network.train(
        files,
        output_path
        //,"training_5/checkpoint-final" // load checkpoint (if needed)
    );
    // clang-format on

    cout << "\n================================ Testing Network ===============================\n\n";

    vector<string> test_fens = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rn1qk2r/ppp1bppp/5n2/3p1bB1/3P4/2N1P3/PP3PPP/R2QKBNR w KQkq - 1 7",
    };

    for(auto fen : test_fens) {
        cout << "FEN: " << fen << endl;
        cout << "Eval: " << network.predict(fen) << endl;
    }

    cout << "\n=================================== Finished ===================================\n";
}
