#pragma once

#include <iostream>
#include <random>
#include <type_traits>

enum class RNGType {
    MT19937,
    MINSTD_RAND,
};

enum class DistType {
    UniformReal,
    Normal,
};

template <RNGType>
struct RNGSelector;

template <>
struct RNGSelector<RNGType::MT19937> {
    using type = std::mt19937;
};

template <>
struct RNGSelector<RNGType::MINSTD_RAND> {
    using type = std::minstd_rand;
};

template <DistType, typename T>
struct DistSelector;

template <typename T>
struct DistSelector<DistType::UniformReal, T> {
    using type = std::uniform_real_distribution<T>;
};

template <typename T>
struct DistSelector<DistType::Normal, T> {
    using type = std::normal_distribution<T>;
};

template <RNGType rng, DistType dist, typename T = double>
class Random {
   private:
    using GeneratorT    = typename RNGSelector<rng>::type;
    using DistributionT = typename DistSelector<dist, T>::type;

    GeneratorT    generator;
    DistributionT distribution;

   public:
    inline Random(unsigned int seed, T param1, T param2)
        : generator(seed), distribution(param1, param2) {
    }

    inline Random() : generator(std::random_device{}()) {
        if constexpr (dist == DistType::UniformReal) {
            distribution = DistributionT(0.0, 1.0);
        } else if constexpr (dist == DistType::Normal) {
            distribution = DistributionT(0.0, 1.0);
        }
    }

    inline T next() {
        return distribution(generator);
    }
};

class IRandom {
   public:
    virtual ~IRandom()    = default;
    virtual double next() = 0;
};

template <RNGType rng, DistType dist, typename T = double>
class RandomWrapper : public IRandom {
   private:
    Random<rng, dist, T> random;

   public:
    RandomWrapper(unsigned int seed, T param1, T param2) : random(seed, param1, param2) {
    }

    double next() override {
        return random.next();
    }
};