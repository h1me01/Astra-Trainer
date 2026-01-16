#include "models/astra.h"

using namespace model;

int main() {
    Astra model("training_1");

    model.train("/home/h1me/Documents/Coding/Astra-Data/nn_output");

    model.evaluate_positions({
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rn1qk2r/ppp1bppp/5n2/3p1bB1/3P4/2N1P3/PP3PPP/R2QKBNR w KQkq - 1 7",
        "5B2/5ppk/p6p/P7/4b2P/3p1qP1/4rP2/2Q2RK1 w - - 10 33",
        "8/8/pp3p2/3p4/PP1PkPK1/8/8/8 b - - 4 52",
        "rnbqkbnr/p3pppp/8/1p6/2pP4/4P3/1P3PPP/RNBQKBNR w KQkq - 0 6",
    });

    return 0;
}
