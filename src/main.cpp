#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "Config.h"
#include "mc/MCFactory.h"
#include "mc/MCPiCalculator.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

using namespace std;
using namespace std::chrono;

long long calculateSampleSize(int decimalPrecision) {
    return 4 * static_cast<long long>(pow(10, decimalPrecision * 2));
}

string getResultFilename() {
    auto   now       = system_clock::now();
    time_t now_time  = system_clock::to_time_t(now);
    tm*    localTime = localtime(&now_time);
    if (!localTime) {
        cerr << "Failed to get local time." << endl;
        throw runtime_error("Failed to get local time.");
    }
    ostringstream oss;
    oss << "results/results_" << put_time(localTime, "%Y-%m-%d_%H-%M-%S") << ".csv";

    return oss.str();
}

int main(int argc, char* argv[]) {
    auto logger = spdlog::stdout_color_mt("console");

    string configFile = (argc > 1) ? argv[1] : "config/config.json";
    logger->info("Using config file: {}", configFile);

    vector<GroupConfig> groups;
    try {
        groups = loadConfigurations(configFile);
    } catch (const exception& e) {
        logger->error("Error loading configurations: {}", e.what());
        return 1;
    }

    string filename;
    try {
        filename = getResultFilename();
    } catch (const exception& e) {
        logger->error("Error getting result filename: {}", e.what());
        return 1;
    }

    ofstream resultsFile(filename);
    if (!resultsFile) {
        logger->error("Error opening file: {}", filename);
        // cerr << "Error opening file: " << filename << endl;
        return 1;
    }

    resultsFile << "Group,Experiment,Backend,ThreadCount,Precision,TotalSamples,"
                   "ChunkSize,PiEstimate,TimeSeconds\n";

    for (const auto& group : groups) {
        logger->info("Processing group: {}", group.groupName);
        for (const auto& exp : group.experiments) {
            BackendType backend = group.global.backend;
            int threadCount   = (exp.threadCount > 0) ? exp.threadCount : group.global.threadCount;
            int precision     = (exp.precision > 0) ? exp.precision : group.global.precision;
            RNGType  rngType  = group.global.rngType;
            DistType distType = group.global.distType;

            long long totalSamples = calculateSampleSize(precision);
            totalSamples           = static_cast<long long>(totalSamples * exp.totalSamplesFactor);
            long long chunkSize    = (totalSamples / threadCount);
            chunkSize              = static_cast<long long>(chunkSize * exp.chunkSizeFactor);

            logger->info("Running experiment: {} (Group: {})", exp.experimentName, group.groupName);
            logger->info(
                "Backend: {}, Threads: {}, Precision: {}, TotalSamples: {}, ChunkSize: "
                "{}",
                (backend == BackendType::OpenMP)    ? "OpenMP"
                : (backend == BackendType::Pthread) ? "Pthreads"
                                                    : "CUDA",
                threadCount,
                precision,
                totalSamples,
                chunkSize);

            IMCPiCalculator* calculator = createCalculator(backend);

            auto   startTime = high_resolution_clock::now();
            double piEstimate =
                calculator->estimatePi(totalSamples, threadCount, chunkSize, rngType, distType);
            auto endTime = high_resolution_clock::now();

            auto   durationMs  = duration_cast<milliseconds>(endTime - startTime);
            double timeSeconds = durationMs.count() / 1000.0;
            delete calculator;

            logger->info("Experiment '{}' result: π = {:.8f}, Time = {:.3f} s",
                         exp.experimentName,
                         piEstimate,
                         timeSeconds);

            resultsFile << group.groupName << "," << exp.experimentName << ","
                        << ((backend == BackendType::OpenMP)    ? "OpenMP"
                            : (backend == BackendType::Pthread) ? "Pthreads"
                                                                : "CUDA")
                        << "," << threadCount << "," << precision << "," << totalSamples << ","
                        << chunkSize << "," << piEstimate << "," << timeSeconds << "\n";
        }
    }

    resultsFile.close();
    logger->info("All experiments complete. Results saved to {}", filename);

    return 0;
}
