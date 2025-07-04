#pragma once

#include <vector>

#include "array.h"

class SparseBatch {
  private:
    int batch_size;
    int max_entries;

    Array<int> psqt_indices;

    Array<int> feature_sizes;
    std::vector<Array<int>> features;

  public:
    SparseBatch(int batch_size, int max_entries)
        : batch_size(batch_size),   //
          max_entries(max_entries), //
          psqt_indices(batch_size), //
          feature_sizes(batch_size) //
    {
        features.reserve(2);
        features.emplace_back(batch_size * max_entries); // stm_features
        features.emplace_back(batch_size * max_entries); // nstm_features
    }

    SparseBatch(const SparseBatch &other)
        : batch_size(other.batch_size),       //
          max_entries(other.max_entries),     //
          psqt_indices(other.psqt_indices),   //
          feature_sizes(other.feature_sizes), //
          features(other.features) {}

    SparseBatch &operator=(const SparseBatch &other) {
        if(this != &other) {
            batch_size = other.batch_size;
            max_entries = other.max_entries;
            psqt_indices = other.psqt_indices;
            feature_sizes = other.feature_sizes;
            features = other.features;
        }
        return *this;
    }

    int get_batch_size() const {
        return batch_size;
    }

    int get_max_entries() const {
        return max_entries;
    }

    Array<int> &get_psqt_indices() {
        return psqt_indices;
    }

    Array<int> &get_feature_sizes() {
        return feature_sizes;
    }

    std::vector<Array<int>> &get_features() {
        return features;
    }

    void host_to_dev() {
        psqt_indices.host_to_dev();
        feature_sizes.host_to_dev();
        for(auto &feature : features)
            feature.host_to_dev();
    }
};
