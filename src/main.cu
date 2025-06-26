#include "nn/network.h"

using namespace std;

int main() {
    const string root_path = "D:/Astra-Data";

    cout << "================================= Training Data ================================\n\n";

    // get training data
    vector<string> files = fetchFilesFromPath(root_path + "/training_data");

    // init network
    Network network( //
        800,         // epochs
        16384,       // batch size
        6104,        // batches per epoch
        100,         // save rate
        1,           // thread count for dataloader
        400,         // output scalar
        1.0,         // start lambda
        1.0          // end lambda
    );

    // init loss
    MPELoss<Sigmoid> loss(2.5); // 2.5 = power
    network.setLoss(&loss);

    // init optim
    Adam optim( //
        0.001,  // lr
        0.9,    // beta1
        0.999,  // beta2
        1e-8    // epsilon
    );

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

    optim.setDecay(0.01);
    optim.setLRScheduler(&lr_sched);
    optim.clamp(-1.99, 1.99); // all weights & biases range [-1.99, 1.99]

    network.setOptimizer(&optim);

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

    network.setKingBucket(king_bucket);

    // init hidden layers
    auto ft = FeatureTransformer<256, SCReLU>(getBucketSize(king_bucket) * 768);
    auto fc = FullyConnected<1, Linear>(&ft);

    network.setHiddenLayers({&ft, &fc});

    // setup quantization scheme
    network.setQuantizationScheme([&](FILE *f) {
        const int q1 = 255;
        const int q2 = 64;

        ft.getParams()[0]->quantize<int16_t>(f, q1, true); // weights
        ft.getParams()[1]->quantize<int16_t>(f, q1);       // biases
        fc.getParams()[0]->quantize<int16_t>(f, q2);       // weights
        fc.getParams()[1]->quantize<int16_t>(f, q1 * q2);  // biases
    });

    const string output_path = root_path + "/nn_output";

    // load weights only (if needed)
    // network.loadWeights(output_path + "/training_3/checkpoint-final/weights.bin");
    network.train(  //
        files,      //
        output_path //
                    // "training_4/checkpoint-100" // load checkpoint (if needed)
    );

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
