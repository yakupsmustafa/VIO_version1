#pragma once
#include <Eigen/Dense>
#include <Eigen/Geometry>

inline Eigen::MatrixXd umeyamaAlign(const Eigen::MatrixXd& src, const Eigen::MatrixXd& dst) {
    // src, dst: (3, N) - sabit 3 satirli tipe cevirerek Eigen::umeyama'ya veriyoruz
    Eigen::Matrix3Xd src3 = src;
    Eigen::Matrix3Xd dst3 = dst;
    return Eigen::umeyama(src3, dst3, true);
}
