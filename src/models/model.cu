#include "model.h"

namespace model {

void Model::init() {
    if (is_initialized)
        return;

    targets = Array<float>(params.batch_size);

    network = std::make_unique<Network>();
    stm_input = std::make_shared<Input>(32);
    nstm_input = std::make_shared<Input>(32);

    dataloader = std::make_unique<Dataloader>(
        params.batch_size, params.thread_count, get_training_files(), [this](const TrainingDataEntry& e) {
            return filter_entry(e);
        }
    );

    loss = get_loss();
    optim = get_optim();
    lr_sched = get_lr_scheduler();

    if (!loss || !optim || !lr_sched)
        error("All components (loss, optimizer, scheduler) must be initialized!");

    build(stm_input, nstm_input);

    network->init(params.batch_size);
    stm_input->init(params.batch_size);
    nstm_input->init(params.batch_size);
    optim->init(network->get_params());

    is_initialized = true;
}

void Model::print_info(int epoch, const std::string& output_path) const {
    std::cout << "\n=============================== Training Data ==============================\n\n";
    const auto& training_files = dataloader->get_filenames();
    if (training_files.empty())
        error("No training data found in the specified paths!");

    for (const auto& f : training_files)
        std::cout << f << std::endl;

    std::cout << "\n=============================== Trainer Info ===============================\n\n";
    std::cout << "Model name:        " << name << std::endl;
    std::cout << "Epochs:            " << params.epochs << std::endl;
    std::cout << "Batch Size:        " << params.batch_size << std::endl;
    std::cout << "Batches/Epoch:     " << params.batches_per_epoch << std::endl;
    std::cout << "Save Rate:         " << params.save_rate << std::endl;
    std::cout << "Thread Count:      " << params.thread_count << std::endl;
    std::cout << "Learning Rate:     " << params.lr << std::endl;
    std::cout << "Eval Div:          " << params.eval_div << std::endl;
    std::cout << "Lambda Start:      " << params.lambda_start << std::endl;
    std::cout << "Lambda End:        " << params.lambda_end << std::endl;
    std::cout << "Output Path:       " << output_path << std::endl;

    if (!loaded_checkpoint.empty())
        std::cout << "Loaded Checkpoint: " << loaded_checkpoint << std::endl;
    else if (!loaded_model.empty())
        std::cout << "Loaded Model:      " << loaded_model << std::endl;

    if (epoch > 0)
        std::cout << "\nResuming from epoch " << epoch << " with learning rate " << lr_sched->get_lr() << std::endl;
}

void Model::fill_inputs(std::vector<TrainingDataEntry>& ds, float lambda) {
    auto& stm_features = stm_input->get_output();
    auto& nstm_features = nstm_input->get_output();

    const int max_entries = stm_input->get_size();

    for (size_t i = 0; i < ds.size(); i++) {
        const Position& pos = ds[i].pos;
        const Color stm = pos.sideToMove();
        const Square ksq_stm = pos.kingSquare(stm);
        const Square ksq_nstm = pos.kingSquare(!stm);
        const int offset = i * max_entries;

        Bitboard pieces = pos.piecesBB();
        int count = 0;

        for (auto sq : pieces) {
            Piece p = pos.pieceAt(sq);
            int idx = offset + count;
            stm_features(idx) = feature_index(p.type(), p.color(), sq, ksq_stm, stm);
            nstm_features(idx) = feature_index(p.type(), p.color(), sq, ksq_nstm, !stm);
            count++;
        }

        // fill remaining slots with -1
        for (int j = count; j < max_entries; j++) {
            int idx = offset + j;
            stm_features(idx) = -1;
            nstm_features(idx) = -1;
        }

        float score_target = 1.0f / (1.0f + expf(-float(ds[i].score) / params.eval_div));
        float wdl_target = (ds[i].result + 1) / 2.0f;
        targets(i) = lambda * score_target + (1.0f - lambda) * wdl_target;
    }

    stm_features.host_to_dev();
    nstm_features.host_to_dev();
    targets.host_to_dev();
}

float Model::predict(const std::string& fen) {
    Position pos;
    pos.set(fen);

    std::vector<TrainingDataEntry> ds{{pos}};

    fill_inputs(ds, 1.0f);
    network->forward(ds);

    auto& output = network->get_output().get_output();
    output.dev_to_host();

    return output(0) * params.eval_div;
}

void Model::train(const std::string& output_path, const std::string& checkpoint_name) {
    init();

    std::string training_folder;
    Logger log;
    int epoch = 0;

    if (checkpoint_name.empty()) {
        training_folder = output_path + "/" + name;
        create_directory(training_folder);

        log.open(training_folder + "/log.txt", false);
        log.write({"epoch", "loss"});
    } else {
        const std::string checkpoint_path = output_path + "/" + checkpoint_name;

        if (!exists(checkpoint_path))
            error("Checkpoint path does not exist: " + checkpoint_path);

        load_params(checkpoint_path + "/model.bin");
        optim->load(checkpoint_path);

        epoch = epoch_from_checkpoint(checkpoint_name);
        lr_sched->lr_from_epoch(epoch);

        training_folder = checkpoint_path.substr(0, checkpoint_path.find_last_of('/'));
        log.open(training_folder + "/log.txt", true);

        loaded_checkpoint = checkpoint_name;
    }

    print_info(epoch, training_folder);

    std::cout << "\n================================= Training =================================\n\n";

    const int positions_per_epoch = params.batch_size * params.batches_per_epoch;

    Timer timer;
    for (epoch = epoch + 1; epoch <= params.epochs; epoch++) {
        timer.start();
        loss->reset();

        float lambda = params.lambda_start + (params.lambda_end - params.lambda_start) * (epoch / float(params.epochs));

        for (int batch = 1; batch <= params.batches_per_epoch; batch++) {
            auto data_entries = dataloader->next();
            fill_inputs(data_entries, lambda);

            timer.stop();
            auto elapsed = timer.elapsed_time();

            if (batch == params.batches_per_epoch || timer.is_time_reached(1000)) {
                printf(
                    "\repoch/batch = %3d/%4d | loss = %1.8f | pos/sec = %7d | time = %3ds",
                    epoch,
                    batch,
                    loss->get_loss() / (params.batch_size * batch),
                    (int)round(1000.0f * params.batch_size * batch / elapsed),
                    (int)elapsed / 1000
                );
                std::cout << std::flush;
            }

            network->forward(data_entries);
            loss->compute(targets, network->get_output());
            network->backward();
            optim->step(lr_sched->get_lr(), data_entries.size());
        }

        float epoch_loss = loss->get_loss() / positions_per_epoch;
        timer.stop();
        auto elapsed = timer.elapsed_time();

        printf("\r\033[K"); // clear the current line
        printf(
            "epoch/batch = %3d/%4d | loss = %1.8f | pos/sec = %7d | time = %3ds\n",
            epoch,
            params.batches_per_epoch,
            epoch_loss,
            (int)round(1000.0f * positions_per_epoch / elapsed),
            (int)elapsed / 1000
        );

        log.write({std::to_string(epoch), std::to_string(epoch_loss)});

        if (epoch % std::max(params.save_rate, 1) == 0 || epoch == params.epochs)
            save_checkpoint(training_folder + "/checkpoint_" + std::to_string(epoch));

        lr_sched->step(epoch);
    }
}

} // namespace model
