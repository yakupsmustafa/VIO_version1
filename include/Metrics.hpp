#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
inline double computeATE(const Eigen::MatrixXd& voAligned, const Eigen::MatrixXd& gt) {
    // her ikisi de (N,3)
    double sumSq = 0;
    for (int i = 0; i < voAligned.rows(); ++i)
        sumSq += (voAligned.row(i) - gt.row(i)).squaredNorm();
    return std::sqrt(sumSq / voAligned.rows());
}
