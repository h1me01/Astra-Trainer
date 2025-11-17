#include "network.h"

namespace nn {

void Network::load_weights(const std::string &file) {
    std::ifstream f(file, std::ios::binary);

    if(!f)
        error("File " + file + " does not exist!");

    try {
        for(auto &l : get_layers()) {
            for(auto &t : l->get_params()) {
                auto &weights = t->get_values();

                f.read(reinterpret_cast<char *>(weights.host_address()), weights.size() * sizeof(float));
                if(f.gcount() != static_cast<std::streamsize>(weights.size() * sizeof(float))) {
                    error("Insufficient data read from file. Expected " + //
                          std::to_string(weights.size()) + " floats!");
                }

                weights.host_to_dev();
            }
        }
    } catch(const std::exception &e) {
        error("Failed loading weights from " + file + ": " + e.what());
    }
}

void Network::save_weights(const std::string &path) {
    try {
        const std::string file = path + "/weights.bin";
        FILE *f = fopen(file.c_str(), "wb");
        if(!f)
            error("Failed writing weights to " + file);

        for(auto &l : get_layers()) {
            for(auto &t : l->get_params()) {
                auto &weights = t->get_values();
                weights.dev_to_host();

                int written = fwrite(weights.host_address(), sizeof(float), weights.size(), f);
                if(written != weights.size())
                    error("Failed writing weights to " + file);
            }
        }

        fclose(f);
    } catch(const std::exception &e) {
        error(std::string("Failed saving weights to ") + e.what());
    }
}

void Network::save_quantized_weights(const std::string &path) {
    try {
        FILE *f = fopen((path + "/qweights.net").c_str(), "wb");
        if(!f)
            error("Failed writing quantized weights!");

        for(auto &l : get_layers()) {
            for(auto &t : l->get_params()) {
                t->get_values().dev_to_host();

                const auto &scheme = t->get_quant_scheme();
                const auto &values = t->get_values();

                switch(scheme.type) {
                case QuantType::INT8:
                    write_quantized<int8_t>(f, values, scheme);
                    break;
                case QuantType::INT16:
                    write_quantized<int16_t>(f, values, scheme);
                    break;
                case QuantType::FLOAT:
                    write_quantized<float>(f, values, scheme);
                    break;
                default:
                    error("Unknown quantization type");
                }
            }
        }

        fclose(f);
    } catch(const std::exception &e) {
        error(std::string("Failed saving quantized weights: ") + e.what());
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
