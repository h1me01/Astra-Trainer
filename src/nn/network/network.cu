#include "network.h"

namespace nn {

void Network::load_weights(const std::string &file) {
    FILE *f = fopen(file.c_str(), "rb");
    if(!f)
        error("File " + file + " does not exist!");

    try {
        for(auto &l : get_layers())
            for(auto &t : l->get_params())
                t->load(f);

        fclose(f);
    } catch(const std::exception &e) {
        fclose(f);
        error("Failed loading weights from " + file + ": " + e.what());
    }
}

void Network::save_weights(const std::string &file) {
    FILE *f = fopen(file.c_str(), "wb");
    if(!f)
        error("Failed writing weights to " + file);

    try {
        for(auto &l : get_layers())
            for(auto &t : l->get_params())
                t->save(f);

        fclose(f);
    } catch(const std::exception &e) {
        fclose(f);
        error("Failed saving weights to " + file + ": " + e.what());
    }
}

void Network::save_quantized_weights(const std::string &file) {
    FILE *f = fopen(file.c_str(), "wb");
    if(!f)
        error("Failed writing quantized weights to " + file);

    try {
        for(auto &l : get_layers())
            for(auto &t : l->get_params())
                t->save_quantized(f);

        fclose(f);
    } catch(const std::exception &e) {
        fclose(f);
        error("Failed saving quantized weights to " + file + ": " + e.what());
    }
}

void Network::fill_inputs(std::vector<DataEntry> &ds, float lambda, float eval_div) {
    auto &[stm_in, nstm_in] = inputs;

    auto &stm_features = stm_in->get_output();
    auto &nstm_features = nstm_in->get_output();

    const int max_entries = stm_in->get_size();

    for(size_t i = 0; i < ds.size(); i++) {
        const Position &pos = ds[i].pos;

        Square ksq_w = pos.kingSquare(Color::White);
        Square ksq_b = pos.kingSquare(Color::Black);

        bool wtm = pos.sideToMove() == Color::White;
        Bitboard pieces = pos.piecesBB();

        const int offset = i * max_entries;

        int count = 0;
        for(auto sq : pieces) {
            Piece p = pos.pieceAt(sq);
            int w_idx = feature_index_fn(p.type(), p.color(), sq, ksq_w, Color::White);
            int b_idx = feature_index_fn(p.type(), p.color(), sq, ksq_b, Color::Black);

            int idx = offset + count;
            stm_features(idx) = wtm ? w_idx : b_idx;
            nstm_features(idx) = wtm ? b_idx : w_idx;

            count++;
        }

        if(count < max_entries) {
            for(int i = count; i < max_entries; i++) {
                int idx = offset + i;
                stm_features(idx) = -1;
                nstm_features(idx) = -1;
            }
        }

        float score_target = 1.0f / (1.0f + expf(-float(ds[i].score) / eval_div));
        float wdl_target = (ds[i].result + 1) / 2.0f;

        targets(i) = lambda * score_target + (1.0f - lambda) * wdl_target;
    }

    stm_features.host_to_dev();
    nstm_features.host_to_dev();
    targets.host_to_dev();
}

} // namespace nn
