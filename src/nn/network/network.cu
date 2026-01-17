#include "network.h"

namespace nn {

void Network::load_weights(const std::string& file) {
    FILE* f = fopen(file.c_str(), "rb");
    if (!f)
        error("File " + file + " does not exist!");

    try {
        for (auto& l : get_layers())
            for (auto& t : l->get_params())
                t->load(f);

        fclose(f);
    } catch (const std::exception& e) {
        fclose(f);
        error("Failed loading weights from " + file + ": " + e.what());
    }
}

void Network::save_weights(const std::string& file) {
    FILE* f = fopen(file.c_str(), "wb");
    if (!f)
        error("Failed writing weights to " + file);

    try {
        for (auto& l : get_layers())
            for (auto& t : l->get_params())
                t->save(f);

        fclose(f);
    } catch (const std::exception& e) {
        fclose(f);
        error("Failed saving weights to " + file + ": " + e.what());
    }
}

void Network::save_quantized_weights(const std::string& file) {
    FILE* f = fopen(file.c_str(), "wb");
    if (!f)
        error("Failed writing quantized weights to " + file);

    try {
        for (auto& l : get_layers())
            for (auto& t : l->get_params())
                t->save_quantized(f);

        fclose(f);
    } catch (const std::exception& e) {
        fclose(f);
        error("Failed saving quantized weights to " + file + ": " + e.what());
    }
}

} // namespace nn
