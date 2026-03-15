#pragma once

#include "../../misc.h"

namespace nn::wdl_sched {

class WDLScheduler {
  public:
    WDLScheduler(float val)
        : val_(val) {}

    virtual void step(int epoch) {}

    float get() const { return val_; }

    virtual std::string info() const = 0;

  protected:
    float val_;
};

} // namespace nn::wdl_sched
