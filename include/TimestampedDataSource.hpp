#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
class TimestampedDataSource {
public:
    std::vector<int64_t> timestamps;
    std::vector<std::vector<double>> values;

    void addSample(int64_t timestamp, const std::vector<double>& value) {
        timestamps.push_back(timestamp);
        values.push_back(value);
    }

    // np.argmin(np.abs(timestamps - ts)) karsiligi
    size_t closestIndex(int64_t ts) const {
        size_t best = 0;
        int64_t bestDiff = std::llabs(timestamps[0] - ts);
        for (size_t i = 1; i < timestamps.size(); ++i) {
            int64_t diff = std::llabs(timestamps[i] - ts);
            if (diff < bestDiff) { bestDiff = diff; best = i; }
        }
        return best;
    }
};
