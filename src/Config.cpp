#include "Config.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

using json = nlohmann::json;

BackendType parseBackend(const std::string& backendStr) {
    if (backendStr == "CUDA")
        return BackendType::CUDA;
    else if (backendStr == "OpenMP")
        return BackendType::OpenMP;
    else if (backendStr == "Pthread" || backendStr == "Pthreads")
        return BackendType::Pthread;
    else
        throw std::runtime_error("Unknown backend: " + backendStr);
}

RNGType parseRNG(const std::string& rngStr) {
    if (rngStr == "MT19937")
        return RNGType::MT19937;
    else if (rngStr == "MINSTD_RAND")
        return RNGType::MINSTD_RAND;
    else
        throw std::runtime_error("Unknown RNG: " + rngStr);
}

DistType parseDist(const std::string& distStr) {
    if (distStr == "UniformReal")
        return DistType::UniformReal;
    else if (distStr == "Normal")
        return DistType::Normal;
    else
        throw std::runtime_error("Unknown distribution: " + distStr);
}

std::vector<GroupConfig> loadConfigurations(const std::string& filename) {
    std::vector<GroupConfig> groups;
    std::ifstream            file(filename);
    if (!file) {
        throw std::runtime_error("Failed to open config file: " + filename);
    }

    json j =
        json::parse(file, nullptr, /* allow_exceptions = */ true, /* ignore_comments = */ true);

    // Check for a top-level "groups" array.
    if (!j.contains("groups") || !j["groups"].is_array()) {
        throw std::runtime_error("Config file must contain a 'groups' array.");
    }

    for (const auto& groupJson : j["groups"]) {
        GroupConfig group;

        // Read group name.
        if (groupJson.contains("name") && groupJson["name"].is_string()) {
            group.groupName = groupJson["name"].get<std::string>();
        } else {
            throw std::runtime_error("Each group must have a name.");
        }

        // Read global settings.
        if (groupJson.contains("global") && groupJson["global"].is_object()) {
            auto globalJson = groupJson["global"];

            if (!globalJson.contains("backend") || !globalJson["backend"].is_string()) {
                throw std::runtime_error("Global config must specify a backend.");
            }
            group.global.backend = parseBackend(globalJson["backend"].get<std::string>());

            if (globalJson.contains("threadCount") &&
                globalJson["threadCount"].is_number_integer()) {
                group.global.threadCount = globalJson["threadCount"].get<int>();
            } else {
                throw std::runtime_error("Global config must specify an integer threadCount.");
            }

            if (globalJson.contains("precision") && globalJson["precision"].is_number_integer()) {
                group.global.precision = globalJson["precision"].get<int>();
            } else {
                throw std::runtime_error("Global config must specify an integer precision.");
            }

            if (globalJson.contains("rng") && globalJson["rng"].is_string()) {
                group.global.rngType = parseRNG(globalJson["rng"].get<std::string>());
            } else {
                throw std::runtime_error("Global config must specify an RNG.");
            }

            if (globalJson.contains("dist") && globalJson["dist"].is_string()) {
                group.global.distType = parseDist(globalJson["dist"].get<std::string>());
            } else {
                throw std::runtime_error("Global config must specify a distribution.");
            }
        } else {
            throw std::runtime_error("Each group must have a global configuration.");
        }

        // Read experiments.
        if (groupJson.contains("experiments") && groupJson["experiments"].is_array()) {
            for (const auto& expJson : groupJson["experiments"]) {
                ExperimentConfig exp;
                if (expJson.contains("experimentName") && expJson["experimentName"].is_string()) {
                    exp.experimentName = expJson["experimentName"].get<std::string>();
                } else {
                    throw std::runtime_error("Each experiment must have an experimentName.");
                }
                // Optional overrides: if not provided, we set default values.
                exp.threadCount =
                    (expJson.contains("threadCount") && expJson["threadCount"].is_number_integer())
                        ? expJson["threadCount"].get<int>()
                        : 0;
                exp.precision =
                    (expJson.contains("precision") && expJson["precision"].is_number_integer())
                        ? expJson["precision"].get<int>()
                        : 0;
                exp.totalSamplesFactor = (expJson.contains("totalSamplesFactor") &&
                                          expJson["totalSamplesFactor"].is_number())
                                             ? expJson["totalSamplesFactor"].get<double>()
                                             : 1.0;
                exp.chunkSizeFactor =
                    (expJson.contains("chunkSizeFactor") && expJson["chunkSizeFactor"].is_number())
                        ? expJson["chunkSizeFactor"].get<double>()
                        : 1.0;

                group.experiments.push_back(exp);
            }
        }
        groups.push_back(group);
    }
    return groups;
}
