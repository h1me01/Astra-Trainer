#include "../misc.h"
#include "network.h"

int Network::index(PieceType pt, Color pc, Square psq, Square ksq, Color view) {
    int _psq = int(psq);
    int _ksq = int(ksq);
    int _pc = int(pc);
    int _pt = int(pt);
    int _view = int(view);

    const int ksIndex = king_bucket[(56 * _view) ^ _ksq];
    // relative square
    _psq = _psq ^ (56 * _view);
    // horizontal flip if king is on other half
    _psq = _psq ^ (7 * !!(_ksq & 0x4));

    return _psq + _pt * 64 + (_pc != _view) * 64 * 6 + ksIndex * 768;
}

void Network::fill(std::vector<DataEntry> &ds) {
    SparseBatch &sparse_inputs = layers[0]->getSparseBatch();

    const int max_entries = sparse_inputs.maxEntries();

    auto &stm_features = sparse_inputs.getFeatures()[0];
    auto &nstm_features = sparse_inputs.getFeatures()[1];
    auto &features_sizes = sparse_inputs.getFeatureSizes();

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

        features_sizes(i) = count;

        float score_target = 1.0f / (1.0f + expf(-float(ds[i].score) / OutputScalar));
        float wdl_target = (ds[i].result + 1) / 2.0f;

        float actual_lambda = StartLambda + (EndLambda - StartLambda) * (epoch / float(Epochs));
        targets(i) = actual_lambda * score_target + (1.0f - actual_lambda) * wdl_target;
    }

    // upload to device
    targets.hostToDev();
    features_sizes.hostToDev();
    stm_features.hostToDev();
    nstm_features.hostToDev();
}

void Network::train(std::vector<std::string> &files, std::string output_path, std::string checkpoint_name) {
    init();
    printInfo();

    if(checkpoint_name.empty()) {
        std::cout << "No checkpoint path provided, training from scratch.\n";
    } else {
        std::cout << "Loading checkpoint from " << checkpoint_name << " ..." << std::endl;
        const std::string checkpoint_path = output_path + "/" + checkpoint_name;
        if(!std::filesystem::exists(checkpoint_path)) {
            std::cerr << "Checkpoint path does not exist: " << checkpoint_path << "\n";
            return;
        }

        loadWeights(checkpoint_path + "/weights.bin");
        optim->load(checkpoint_path);
    }

    // init dataloader
    FeaturedBatchStream dataloader(files, 4, BatchSize, false);

    std::cout << "\n=============================== Training Network ===============================\n\n";

    // add new folder to output path
    int next_training_index = getNextTrainingIndex(output_path);

    std::stringstream new_folder_path;
    new_folder_path << output_path << "/training_" << next_training_index;

    std::filesystem::create_directory(new_folder_path.str());
    std::cout << "Created folder: " << new_folder_path.str() << "\n\n";

    // save network info
    std::ofstream info_file(new_folder_path.str() + "/info.txt");
    if(info_file.is_open()) {
        info_file << info.str();
        info_file << dataloader.getInfo() << "\n";
        info_file.close();
    } else {
        std::cerr << "Failed to save info file!\n";
        return;
    }

    Logger log{new_folder_path.str() + "/loss.csv"};
    log.write({"epoch", "training_loss"});

    Timer timer;
    for(epoch = 1; epoch <= Epochs; epoch++) {
        timer.start();
        loss->reset();

        for(int batch = 1; batch <= BatchesPerEpoch; batch++) {
            auto ds = dataloader.next();
            fill(ds);

            timer.stop();
            auto elapsed = timer.getElapsedTime();

            if(batch == BatchesPerEpoch || timer.isTimeReached(1000)) {
                printf("\repoch/batch = %3d/%4d, ", epoch, batch);
                printf("pos/s = %7d, ", (int) round(1000.0f * BatchSize * batch / elapsed));
                printf("time = %3ds", (int) elapsed / 1000);
                std::cout << std::flush;
            }

            forward();
            loss->apply(targets, getOutput());
            backprop();
            optim->apply(ds.size());
        }

        float epoch_loss = loss->getLoss() / (BatchSize * BatchesPerEpoch);

        timer.stop();
        auto elapsed = timer.getElapsedTime();

        printf("\repoch/batch = %3d/%4d, ", epoch, BatchesPerEpoch);
        printf("loss = %1.8f, ", epoch_loss);
        printf("pos/s = %7d, ", (int) round(1000.0f * BatchSize * BatchesPerEpoch / elapsed));
        printf("time = %3ds", (int) elapsed / 1000);
        std::cout << std::endl;

        if(epoch % SaveRate == 0 || epoch == Epochs) {
            log.write({std::to_string(epoch), std::to_string(epoch_loss)});

            std::string suffix = epoch == Epochs ? "final" : std::to_string(epoch);
            saveCheckpoint(new_folder_path.str() + "/checkpoint-" + suffix);
        }

        optim->updateLR(epoch);
    }
}
