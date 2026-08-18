#pragma once
#include "TimestampedDataSource.hpp"
#include <vector>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
class Synchronizer {
public:
    const TimestampedDataSource& reference;
    std::vector<const TimestampedDataSource*> others;

    Synchronizer(const TimestampedDataSource& ref, std::initializer_list<const TimestampedDataSource*> otherSources)
        : reference(ref), others(otherSources) {}

    std::vector<std::vector<std::vector<double>>> alignAll() const {
        std::vector<std::vector<std::vector<double>>> results;
        for (auto* src : others) results.push_back(alignOne(*src));
        return results;
    }

private:
    std::vector<std::vector<double>> alignOne(const TimestampedDataSource& source) const {
        size_t startIdx = source.closestIndex(reference.timestamps[0]);
        std::vector<double> startValue = source.values[startIdx];

        std::vector<std::vector<double>> aligned;
        for (int64_t ts : reference.timestamps) {
            size_t idx = source.closestIndex(ts);
            const auto& v = source.values[idx];
            std::vector<double> diff(v.size());
            for (size_t k = 0; k < v.size(); ++k) diff[k] = v[k] - startValue[k];
            aligned.push_back(diff);
        }
        return aligned;
    }
};
