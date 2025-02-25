#pragma once

class IGLPiCalculator {
   public:
    virtual ~IGLPiCalculator() {
    }

    virtual long double estimatePi(unsigned long long maxTerms)         = 0;
    virtual long double estimatePiPrecision(unsigned int decimalPlaces) = 0;
};