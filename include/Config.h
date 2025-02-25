#pragma once

#include <string>
#include <vector>

#include "mc/MCFactory.h"
#include "random/Random.hpp"

enum class Method { MonteCarlo, GregoryLeibniz, GregoryLeibnizDynamic };

struct GlobalConfig {
    BackendType        backend;
    int                threadCount;
    int                precision;
    RNGType            rngType;
    DistType           distType;
    unsigned long long sampleCount;  // use if > 0
    unsigned long long chunkSize;    // use if > 0
};

struct ExperimentConfig {
    std::string experimentName;
    // Optional overrides – use -1 or 0 to signal “not set”
    int                threadCount;         // override global if > 0
    int                precision;           // override global if > 0
    double             totalSamplesFactor;  // multiplier (default 1.0)
    double             chunkSizeFactor;     // multiplier (default 1.0)
    unsigned long long sampleCount;         // use if > 0
};

struct GroupConfig {
    std::string                   groupName;
    Method                        method;
    GlobalConfig                  global;
    std::vector<ExperimentConfig> experiments;
};

std::vector<GroupConfig> loadConfigurations(const std::string& filename);
