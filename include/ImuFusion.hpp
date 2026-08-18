#pragma once
#include <opencv2/opencv.hpp>
#include "TimestampedDataSource.hpp"
#include <numeric>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
constexpr double GRAVITY = 9.81;

class ImuFusion {
public:
    const TimestampedDataSource& imu;
    double alpha, stationaryThreshold;

    ImuFusion(const TimestampedDataSource& imu_, double alpha_ = 0.3, double thresh_ = 0.15)
        : imu(imu_), alpha(alpha_), stationaryThreshold(thresh_) {}

    void fuse(cv::Mat& R, cv::Mat& t, int64_t t0, int64_t t1) {
        auto lo = std::lower_bound(imu.timestamps.begin(), imu.timestamps.end(), t0);
        auto hi = std::upper_bound(imu.timestamps.begin(), imu.timestamps.end(), t1);
        size_t i0 = lo - imu.timestamps.begin();
        size_t i1 = hi - imu.timestamps.begin();
        if (i0 >= i1) return;

        double dt = (t1 - t0) / 1e9;
        cv::Vec3d avgGyro(0,0,0);
        std::vector<double> accelNorms;
        for (size_t i = i0; i < i1; ++i) {
            const auto& v = imu.values[i];
            avgGyro += cv::Vec3d(v[3], v[4], v[5]);
            accelNorms.push_back(cv::norm(cv::Vec3d(v[0], v[1], v[2])));
        }
        avgGyro *= (1.0 / (i1 - i0));

        cv::Mat rotvecImu = (cv::Mat_<double>(3,1) << avgGyro[0]*dt, avgGyro[1]*dt, avgGyro[2]*dt);
        cv::Mat Rimu; cv::Rodrigues(rotvecImu, Rimu);

        cv::Mat rvVo, rvImu;
        cv::Rodrigues(R, rvVo); cv::Rodrigues(Rimu, rvImu);
        cv::Mat rvFused = alpha * rvImu + (1 - alpha) * rvVo;
        cv::Rodrigues(rvFused, R);

        double meanAbsDev = 0;
        for (double n : accelNorms) meanAbsDev += std::abs(n - GRAVITY);
        meanAbsDev /= accelNorms.size();

        if (meanAbsDev < stationaryThreshold) t = cv::Mat::zeros(3,1,CV_64F);
    }
};
