#include "Config.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

using json = nlohmann::json;

Method parseMethod(const std::string& methodStr) {
    if (methodStr == "MonteCarlo" || methodStr == "MC")
        return Method::MonteCarlo;
    else if (methodStr == "GregoryLeibniz" || methodStr == "GL")
        return Method::GregoryLeibniz;
    else if (methodStr == "GregoryLeibnizDynamic" || methodStr == "GLD")
        return Method::GregoryLeibnizDynamic;
    else
        throw std::runtime_error("Unknown method: " + methodStr);
}

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

static inline std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    size_t end   = s.find_last_not_of(" \t\n\r");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

unsigned long long parseSampleCount(const std::string& str) {
    if (str.empty()) {
        throw std::invalid_argument("Input string is empty");
    }

    std::string trimmed = trim(str);

    // Check for exponentiation expression, e.g., "2^10"
    size_t caretPos = trimmed.find('^');
    if (caretPos != std::string::npos) {
        std::string baseStr = trim(trimmed.substr(0, caretPos));
        std::string expStr  = trim(trimmed.substr(caretPos + 1));

        if (baseStr.empty() || expStr.empty()) {
            throw std::invalid_argument("Invalid exponentiation format");
        }

        // Parse the base and exponent
        double             baseValue     = 0.0;
        double             exponentValue = 0.0;
        std::istringstream issBase(baseStr);
        std::istringstream issExp(expStr);
        if (!(issBase >> baseValue)) {
            throw std::invalid_argument("Invalid base value in exponentiation: " + baseStr);
        }
        if (!(issExp >> exponentValue)) {
            throw std::invalid_argument("Invalid exponent value in exponentiation: " + expStr);
        }

        double result = std::pow(baseValue, exponentValue);
        if (result < 0 ||
            result > static_cast<double>(std::numeric_limits<unsigned long long>::max())) {
            throw std::overflow_error(
                "Exponentiation result is out of range for unsigned long long");
        }
        return static_cast<unsigned long long>(result);
    }

    // handle optional suffixes (e.g., "40B", "5.5M", "100K")
    char               lastChar   = trimmed.back();
    unsigned long long multiplier = 1;
    std::string        numberPart = trimmed;

    if (!std::isdigit(lastChar)) {
        switch (std::toupper(lastChar)) {
            case 'K':
                multiplier = 1000ULL;
                break;
            case 'M':
                multiplier = 1000000ULL;
                break;
            case 'B':
                multiplier = 1000000000ULL;
                break;
            default:
                throw std::invalid_argument("Unsupported suffix in input string: " +
                                            std::string(1, lastChar));
        }
        numberPart = trim(trimmed.substr(0, trimmed.size() - 1));
        if (numberPart.empty()) {
            throw std::invalid_argument("Missing numeric part before suffix");
        }
    }

    double             baseValue = 0.0;
    std::istringstream iss(numberPart);
    if (!(iss >> baseValue)) {
        throw std::invalid_argument("Invalid numeric value in input string: " + numberPart);
    }

    double result = baseValue * multiplier;
    if (result < 0 ||
        result > static_cast<double>(std::numeric_limits<unsigned long long>::max())) {
        throw std::overflow_error("Parsed value is out of range for unsigned long long");
    }

    return static_cast<unsigned long long>(result);
}

std::vector<GroupConfig> loadConfigurations(const std::string& filename) {
    std::vector<GroupConfig> groups;
    std::ifstream            file(filename);
    if (!file) {
        throw std::runtime_error("Failed to open config file: " + filename);
    }

    json j =
        json::parse(file, nullptr, /* allow_exceptions = */ true, /* ignore_comments = */ true);

    // Check for a top-level "groups" array
    if (!j.contains("groups") || !j["groups"].is_array()) {
        throw std::runtime_error("Config file must contain a 'groups' array.");
    }

    for (const auto& groupJson : j["groups"]) {
        GroupConfig group;

        if (groupJson.contains("name") && groupJson["name"].is_string()) {
            group.groupName = groupJson["name"].get<std::string>();
        } else {
            throw std::runtime_error("Each group must have a name.");
        }

        if (groupJson.contains("method") && groupJson["method"].is_string()) {
            group.method = parseMethod(groupJson["method"].get<std::string>());
        } else {
            throw std::runtime_error("Each group must have a method.");
        }

        // Read global settings
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

            if (globalJson.contains("sampleCount") && globalJson["sampleCount"].is_string()) {
                group.global.sampleCount =
                    parseSampleCount(globalJson["sampleCount"].get<std::string>());
            } else {
                group.global.sampleCount = 0;
            }

            if (globalJson.contains("chunkSize") && globalJson["chunkSize"].is_number_integer()) {
                group.global.chunkSize = globalJson["chunkSize"].get<int>();
            } else {
                group.global.chunkSize = 0;
            }
        } else {
            throw std::runtime_error("Each group must have a global configuration.");
        }

        // Read experiments
        if (groupJson.contains("experiments") && groupJson["experiments"].is_array()) {
            for (const auto& expJson : groupJson["experiments"]) {
                ExperimentConfig exp;
                if (expJson.contains("experimentName") && expJson["experimentName"].is_string()) {
                    exp.experimentName = expJson["experimentName"].get<std::string>();
                } else {
                    throw std::runtime_error("Each experiment must have an experimentName.");
                }
                // Optional overrides: if not provided, set default values
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

                exp.sampleCount =
                    (expJson.contains("sampleCount") && expJson["sampleCount"].is_string())
                        ? parseSampleCount(expJson["sampleCount"].get<std::string>())
                        : 0;

                group.experiments.push_back(exp);
            }
        }
        groups.push_back(group);
    }
    return groups;
}
