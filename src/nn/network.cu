#include "../misc.h"
#include "network.h"

using namespace std::filesystem;

// sparse batch definition

SparseBatch LayerBase::sparse_batch{1, 1};

// helper

std::vector<std::string> files_from_path(const std::vector<std::string> &paths) {
    std::cout << "================================= Training Data ================================\n\n";
    std::cout << "Loading files from folder(s):\n";
    for(const auto &p : paths)
        std::cout << p << std::endl;
    std::cout << std::endl;

    std::vector<std::string> files;
    for(const auto &path : paths) {
        try {
            for(const auto &entry : recursive_directory_iterator(path)) {
                if(entry.is_regular_file()) {
                    files.push_back(entry.path().string());
                    std::cout << entry.path() << std::endl;
                }
            }
        } catch(const filesystem_error &e) {
            std::cerr << "Filesystem error in path " << path << ": " << e.what() << std::endl;
        }
    }

    if(files.empty()) {
        std::string all_paths;
        for(const auto &p : paths)
            all_paths += p + " ";
        error("No training data found in the specified paths: " + all_paths);
    }

    return files;
}

int epoch_from_checkpoint(const std::string &checkpoint_name) {
    size_t dash_pos = checkpoint_name.find_last_of('-');
    if(dash_pos == std::string::npos) {
        std::cout << "Could not parse epoch from checkpoint name, starting from epoch 0.\n";
        return 0;
    }

    std::string epoch_str = checkpoint_name.substr(dash_pos + 1);
    if(epoch_str == "final") {
        std::cout << "Loading from final checkpoint, starting new training cycle.\n";
        return 0;
    }

    try {
        int parsed_epoch = std::stoi(epoch_str);
        return parsed_epoch;
    } catch(...) {
        std::cout << "Could not parse epoch from checkpoint name, starting from epoch 0.\n";
        return 0;
    }
}

int get_next_training_idx(const std::string &output_path) {
    int max_index = 0;

    for(const auto &entry : directory_iterator(output_path)) {
        if(!entry.is_directory())
            continue;

        std::string folder_name = entry.path().filename().string();
        if(folder_name.find("training_") == 0) {
            try {
                int index = std::stoi(folder_name.substr(9));
                max_index = std::max(max_index, index);
            } catch(const std::exception &) {
                continue; // skip folders with invalid numeric suffixes
            }
        }
    }

    return max_index + 1;
}

// network class

void Network::save_checkpoint(const std::string &path) {
    // create directory if it doesn't exist
    try {
        create_directories(path);
    } catch(const filesystem_error &e) {
        error("Failed creating directory " + path + ": " + e.what());
    }

    // save weights
    try {
        const std::string file = path + "/weights.bin";
        FILE *f = fopen(file.c_str(), "wb");
        if(!f)
            error("Failed writing weights to " + file);

        for(LayerBase *l : hp.layers) {
            for(Tensor *t : l->get_params()) {
                DenseMatrix<float> &weights = t->get_data();
                weights.dev_to_host();

                int written = fwrite(weights.host_address(), sizeof(float), weights.size(), f);
                if(written != weights.size())
                    error("Failed writing weights to " + file);
            }
        }

        fclose(f);
    } catch(const std::exception &e) {
        error(std::string("Failed saving weights: ") + e.what());
    }

    // save quantized weights
    try {
        FILE *f = fopen((path + "/qweights.net").c_str(), "wb");
        if(!f)
            error("Failed writing quantized weights");

        for(LayerBase *l : hp.layers)
            for(Tensor *t : l->get_params())
                t->save_quantize(f);

        fclose(f);
    } catch(const std::exception &e) {
        error(std::string("Failed saving quantized weights: ") + e.what());
    }

    // save optimizer state
    if(hp.optim != nullptr)
        hp.optim->save(path);

    std::cout << "Saved checkpoint" << std::endl;
}

int Network::index(PieceType pt, Color pc, Square psq, Square ksq, Color view) {
    int _psq = int(psq);
    int _ksq = int(ksq);
    int _pc = int(pc);
    int _pt = int(pt);
    int _view = int(view);

    const int ksIndex = hp.input_bucket[(56 * _view) ^ _ksq];
    // relative square
    _psq = _psq ^ (56 * _view);
    // horizontal flip if king is on other half
    _psq = _psq ^ (7 * !!(_ksq & 0x4));

    return _psq + _pt * 64 + (_pc != _view) * 64 * 6 + ksIndex * 768;
}

void Network::fill(std::vector<DataEntry> &ds, float lambda) {
    ASSERT(ds.size() == hp.batch_size || ds.size() == 1);

    SparseBatch &sb = hp.layers[0]->get_sparse_batch();

    const int max_entries = sb.get_max_entries();

    auto &psqt_indices = sb.get_psqt_indices();
    auto &features_sizes = sb.get_feature_sizes();
    auto &stm_features = sb.get_features()[0];
    auto &nstm_features = sb.get_features()[1];

    for(size_t i = 0; i < ds.size(); i++) {
        const auto pos = ds[i].pos;

        auto ksq_w = pos.kingSquare(Color::White);
        auto ksq_b = pos.kingSquare(Color::Black);

        bool wtm = pos.sideToMove() == Color::White;
        auto pieces = pos.piecesBB();

        int offset = i * max_entries;
        int count = 0;
        for(auto sq : pieces) {
            auto p = pos.pieceAt(sq);
            auto w_idx = index(p.type(), p.color(), sq, ksq_w, Color::White);
            auto b_idx = index(p.type(), p.color(), sq, ksq_b, Color::Black);

            int idx = offset + count;
            stm_features(idx) = wtm ? w_idx : b_idx;
            nstm_features(idx) = wtm ? b_idx : w_idx;

            count++;
        }

        psqt_indices(i) = (count - 2) / 4;
        ASSERT(psqt_indices(i) >= 0 && psqt_indices(i) < 8);

        features_sizes(i) = count;

        float score_target = 1.0f / (1.0f + expf(-float(ds[i].score) / hp.output_scalar));
        float wdl_target = (ds[i].result + 1) / 2.0f;

        targets(i) = lambda * score_target + (1.0f - lambda) * wdl_target;
    }

    // upload to device
    sb.host_to_dev();
    targets.host_to_dev();
}

void Network::train(std::vector<std::string> data_path, std::string output_path, std::string checkpoint_name) {
    const std::vector<std::string> files = files_from_path(data_path);

    init();

    std::string training_folder;
    Logger log;
    int epoch;

    if(checkpoint_name.empty()) {
        std::cout << "No checkpoint path provided, training from scratch.\n";

        int next_training_index = get_next_training_idx(output_path);
        std::stringstream new_folder_path;
        new_folder_path << output_path << "/training_" << next_training_index;
        training_folder = new_folder_path.str();

        create_directory(training_folder);
        std::cout << "Created folder: " << training_folder << std::endl;

        log.open(training_folder + "/log.txt", false);
        log.write({"epoch", "loss"});

        epoch = 0;
    } else {
        std::cout << "Loading checkpoint from " << checkpoint_name << " ..." << std::endl;
        const std::string checkpoint_path = output_path + "/" + checkpoint_name;

        if(!exists(checkpoint_path)) {
            error("Checkpoint path does not exist: " + checkpoint_path);
        }

        load_weights(checkpoint_path + "/weights.bin");
        hp.optim->load(checkpoint_path);
        std::cout << std::endl;

        epoch = epoch_from_checkpoint(checkpoint_name);
        hp.optim->lr_from_epoch(epoch);

        if(epoch > 0)
            std::cout << "Resuming from epoch " << epoch << " with learning rate " << hp.optim->get_lr() << std::endl;

        training_folder = checkpoint_path.substr(0, checkpoint_path.find_last_of('/'));

        std::cout << "Using existing folder: " << training_folder << std::endl;
        log.open(training_folder + "/log.txt", true);

        hp.loaded_checkpoint = checkpoint_name;
    }

    hp.print_info();

    // init dataloader
    FeaturedBatchStream dataloader(files, hp.thread_count, hp.batch_size, true);

    std::cout << "\n=============================== Training Network ===============================\n\n";

    // save/update network info
    std::ofstream info_file(training_folder + "/info.txt");
    if(info_file.is_open()) {
        info_file << hp.info.str();
        info_file << dataloader.get_info() << "\n";
        info_file.close();
    } else {
        error("Failed opening info file for writing: " + training_folder + "/info.txt");
    }

    const int positions_per_epoch = hp.batch_size * hp.batches_per_epoch;

    Timer timer;
    for(epoch = epoch + 1; epoch <= hp.epochs; epoch++) {
        timer.start();
        hp.loss->reset();

        float lambda = hp.start_lambda + (hp.end_lambda - hp.start_lambda) * (epoch / float(hp.epochs));

        for(int batch = 1; batch <= hp.batches_per_epoch; batch++) {
            auto ds = dataloader.next();
            fill(ds, lambda);

            timer.stop();
            auto elapsed = timer.elapsed_time();

            if(batch == hp.batches_per_epoch || timer.is_time_reached(1000)) {
                printf("\repoch/batch = %3d/%4d, loss = %1.8f, pos/sec = %7d, time = %3ds",
                       epoch,
                       batch,
                       hp.loss->get_loss() / (hp.batch_size * batch),
                       (int) round(1000.0f * hp.batch_size * batch / elapsed),
                       (int) elapsed / 1000);
                std::cout << std::flush;
            }

            forward();
            hp.loss->compute(targets, get_output());
            backward();
            hp.optim->step(ds.size());
        }

        float epoch_loss = hp.loss->get_loss() / positions_per_epoch;

        timer.stop();
        auto elapsed = timer.elapsed_time();

        printf("\r\033[K"); // clear the current line
        printf("epoch/batch = %3d/%4d, loss = %1.8f, pos/sec = %7d, time = %3ds\n",
               epoch,
               hp.batches_per_epoch,
               epoch_loss,
               (int) round(1000.0f * positions_per_epoch / elapsed),
               (int) elapsed / 1000);

        log.write({std::to_string(epoch), std::to_string(epoch_loss)});

        if(epoch % hp.save_rate == 0 || epoch == hp.epochs) {
            std::string suffix = epoch == hp.epochs ? "final" : std::to_string(epoch);
            save_checkpoint(training_folder + "/checkpoint-" + suffix);
        }

        hp.optim->update_lr(epoch);
    }
}
