#pragma once

#include "Random.hpp"

class RandomBuilder {
   public:
    static IRandom* build(RNGType rng, DistType dist, unsigned int seed, double param1,
                          double param2) {
        switch (rng) {
            case RNGType::MT19937:
                switch (dist) {
                    case DistType::UniformReal:
                        return new RandomWrapper<RNGType::MT19937, DistType::UniformReal>(
                            seed, param1, param2);
                    case DistType::Normal:
                        return new RandomWrapper<RNGType::MT19937, DistType::Normal>(
                            seed, param1, param2);
                }
                break;
            case RNGType::MINSTD_RAND:
                switch (dist) {
                    case DistType::UniformReal:
                        return new RandomWrapper<RNGType::MINSTD_RAND, DistType::UniformReal>(
                            seed, param1, param2);
                    case DistType::Normal:
                        return new RandomWrapper<RNGType::MINSTD_RAND, DistType::Normal>(
                            seed, param1, param2);
                }
                break;
        }
        throw std::runtime_error("Unknown RNG/Distribution combination");
    }
};
