#include "utils.h"

using namespace std::filesystem;

namespace nn::utils {

std::vector<std::string> files_from_path(const std::vector<std::string> &paths) {
    std::cout << "\n============================== Training Data =============================\n\n";
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

    std::cout << "\nFound " << files.size() << " training file(s)\n\n";
    return files;
}

int epoch_from_checkpoint(const std::string &checkpoint_name) {
    size_t dash_pos = checkpoint_name.find_last_of('-');
    if(dash_pos == std::string::npos) {
        std::cout << "Could not parse epoch from checkpoint name, starting from epoch 0\n";
        return 0;
    }

    std::string epoch_str = checkpoint_name.substr(dash_pos + 1);
    if(epoch_str == "final") {
        std::cout << "Loading from final checkpoint, starting new training cycle\n";
        return 0;
    }

    try {
        int parsed_epoch = std::stoi(epoch_str);
        return parsed_epoch;
    } catch(...) {
        std::cout << "Could not parse epoch from checkpoint name, starting from epoch 0\n";
        return 0;
    }
}

} // namespace nn::utils
