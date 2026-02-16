#pragma once

#include "wdl_scheduler.h"

namespace nn::wdl_sched {

struct Constant : public WDLScheduler {
    Constant(float val)
        : WDLScheduler(val) {}

    std::string get_info() const override { return "Constant(val=" + format_number(val) + ")"; }
};

} // namespace nn::wdl_sched
