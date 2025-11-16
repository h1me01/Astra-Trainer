#include "../nn/include.h"

namespace model {

void Model::load_weights(const std::string &file) {
    init_trainer();
    trainer->load_weights(file);
}

void Model::save_weights(const std::string &file) {
    init_trainer();
    trainer->save_weights(file);
}

void Model::evaluate_positions(const std::vector<std::string> &positions) {
    init_trainer();
    trainer->evaluate_positions(positions);
}

void Model::train(std::vector<std::string> data_path, std::string output_path, std::string checkpoint_name) {
    init_trainer();
    trainer->train(data_path, output_path, checkpoint_name);
}

void Model::init_trainer() {
    if(trainer == nullptr)
        trainer = std::make_unique<Trainer>(this);
}

} // namespace model
