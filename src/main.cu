#include "models/astra.h"

int main() {
    const std::string root_path = "D:/Astra-Data";

    Astra model("astra_standard_1");

    model.train( //
        {
            // data paths
            root_path + "/training_data", //
        },                                //
        root_path + "/nn_output",         // output path
        ""                                // checkpoint from output path
    );

    model.evaluate_positions({
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rn1qk2r/ppp1bppp/5n2/3p1bB1/3P4/2N1P3/PP3PPP/R2QKBNR w KQkq - 1 7",
    });

    return 0;
}
