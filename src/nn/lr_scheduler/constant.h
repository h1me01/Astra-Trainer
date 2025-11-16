#pragma once

#include "lr_scheduler.h"

namespace nn {

class Constant : public LRScheduler {
  public:
    Constant(float lr) : LRScheduler(lr) {}

    void step(int epoch) override {}
};

} // namespace nn
