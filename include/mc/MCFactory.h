#pragma once

#include <stdexcept>
#include <string>

#include "mc/CudaMCPiCalculator.h"
#include "mc/MCPiCalculator.h"
#include "mc/OpenMPMCPiCalculator.h"
#include "mc/PthreadMCPiCalculator.h"

enum class BackendType { OpenMP, Pthread, CUDA };

IMCPiCalculator* createCalculator(BackendType backend);
