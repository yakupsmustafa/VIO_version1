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
    // timestamps monoton artan siralı oldugundan (CSV/dosya adi sirasiyla geldigi gibi) binary search
    // ile O(log N) - lower_bound, ts'den >= olan ilk elemani verir; gercek en yakin o ya da bir oncekidir.
    size_t closestIndex(int64_t ts) const {
        auto it = std::lower_bound(timestamps.begin(), timestamps.end(), ts);
        if (it == timestamps.begin()) return 0;
        if (it == timestamps.end()) return timestamps.size() - 1;
        size_t after = it - timestamps.begin();
        size_t before = after - 1;
        return (std::llabs(timestamps[after] - ts) < std::llabs(timestamps[before] - ts)) ? after : before;
    }
};
