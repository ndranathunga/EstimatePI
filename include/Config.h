#pragma once

#include <string>
#include <vector>

#include "mc/MCFactory.h"
#include "random/Random.hpp"

struct GlobalConfig {
    BackendType backend;
    int         threadCount;
    int         precision;
    RNGType     rngType;
    DistType    distType;
};

struct ExperimentConfig {
    std::string experimentName;
    // Optional overrides – use -1 or 0 to signal “not set”
    int    threadCount;         // override global if > 0
    int    precision;           // override global if > 0
    double totalSamplesFactor;  // multiplier (default 1.0)
    double chunkSizeFactor;     // multiplier (default 1.0)
};

struct GroupConfig {
    std::string                   groupName;
    GlobalConfig                  global;
    std::vector<ExperimentConfig> experiments;
};

std::vector<GroupConfig> loadConfigurations(const std::string& filename);
