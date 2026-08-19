#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/opencv.hpp>
#include "ImuFactor.hpp"
#include "ImuPreintegration.hpp"
#include <array>
#include <vector>

// ---- Reprojection error (gorsel hata) - bagimsiz, disaridan dosyaya ihtiyac duymuyor ----
struct ReprojectionError {
    ReprojectionError(double obsX, double obsY, double fx, double fy, double cx, double cy)
        : obsX(obsX), obsY(obsY), fx(fx), fy(fy), cx(cx), cy(cy) {}

    template <typename T>
    bool operator()(const T* const pose /*[6]: angleAxis(3)+t(3)*/,
                     const T* const point /*[3]*/, T* residuals) const {
        T p[3];
        ceres::AngleAxisRotatePoint(pose, point, p);
        p[0] += pose[3]; p[1] += pose[4]; p[2] += pose[5];

        T xp = p[0] / p[2];
        T yp = p[1] / p[2];
        T predX = T(fx) * xp + T(cx);
        T predY = T(fy) * yp + T(cy);

        residuals[0] = predX - T(obsX);
        residuals[1] = predY - T(obsY);
        return true;
    }

    static ceres::CostFunction* create(double obsX, double obsY, double fx, double fy, double cx, double cy) {
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
            new ReprojectionError(obsX, obsY, fx, fy, cx, cy));
    }

    double obsX, obsY, fx, fy, cx, cy;
};

// Gorsel (reprojection) + IMU (preintegration) hatasini BIRLIKTE optimize eden BA.
class VIOBundleAdjustment {
public:
    cv::Mat K;
    int windowSize;
    cv::Vec3d gravityWorld;
    ImuPreintegrator& imuPreint;

    VIOBundleAdjustment(const cv::Mat& K_, ImuPreintegrator& imuPreint_,
                         const cv::Vec3d& gravity_ = cv::Vec3d(0,0,9.81), int windowSize_ = 5)
        : K(K_), windowSize(windowSize_), gravityWorld(gravity_), imuPreint(imuPreint_) {}

    struct FrameState {
        std::array<double,6> pose;  // [angleAxis(3), t(3)]
        std::array<double,3> vel;
        std::array<double,3> ba;    // accel bias
        std::array<double,3> bg;    // gyro bias
    };

    bool optimize(std::vector<FrameState>& frameStates,
                  const std::vector<int64_t>& frameTimestamps,
                  const std::vector<std::vector<cv::KeyPoint>>& kpList,
                  const std::vector<std::vector<int>>& tracks,
                  std::vector<std::array<double,3>>& landmarks) {

        int n = windowSize;
        if ((int)frameStates.size() != n || (int)frameTimestamps.size() != n) return false;

        ceres::Problem problem;
        double fx = K.at<double>(0,0), fy = K.at<double>(1,1), cx = K.at<double>(0,2), cy = K.at<double>(1,2);

        for (size_t ti = 0; ti < tracks.size(); ++ti) {
            for (int fi = 0; fi < n; ++fi) {
                cv::Point2f obs = kpList[fi][tracks[ti][fi]].pt;
                ceres::CostFunction* cost = ReprojectionError::create(obs.x, obs.y, fx, fy, cx, cy);
                problem.AddResidualBlock(cost, nullptr, frameStates[fi].pose.data(), landmarks[ti].data());
            }
        }

        for (int i = 0; i + 1 < n; ++i) {
            cv::Vec3d bg(frameStates[i].bg[0], frameStates[i].bg[1], frameStates[i].bg[2]);
            cv::Vec3d ba(frameStates[i].ba[0], frameStates[i].ba[1], frameStates[i].ba[2]);

            PreintegratedImuData preint = imuPreint.preintegrate(
                frameTimestamps[i], frameTimestamps[i+1], bg, ba);

            if (preint.deltaT <= 0) continue;

            ceres::CostFunction* imuCost = ImuFactor::create(preint, gravityWorld);
            problem.AddResidualBlock(imuCost, nullptr,
                frameStates[i].pose.data(),   frameStates[i].vel.data(),
                frameStates[i].ba.data(),     frameStates[i].bg.data(),
                frameStates[i+1].pose.data(), frameStates[i+1].vel.data(),
                frameStates[i+1].ba.data(),   frameStates[i+1].bg.data());
        }

        // ---- Guvenlik: parametre problem'e hic eklenmemisse SetConstant cagirma (crash onler) ----
        if (problem.HasParameterBlock(frameStates[0].pose.data()))
            problem.SetParameterBlockConstant(frameStates[0].pose.data());
        if (problem.HasParameterBlock(frameStates[0].vel.data()))
            problem.SetParameterBlockConstant(frameStates[0].vel.data());

        ceres::Solver::Options options;
        options.max_num_iterations = 50;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        return summary.IsSolutionUsable();
    }
};
