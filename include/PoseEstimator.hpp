#pragma once
#include <opencv2/opencv.hpp>
#include "Config.hpp"
#include <optional>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
struct RelativePose { cv::Mat R, t; };

class PoseEstimator {
public:
    cv::Mat K;
    double ransacThreshold;

    explicit PoseEstimator(const Config& cfg) : K(cfg.K), ransacThreshold(cfg.ransacThreshold) {}

    std::optional<RelativePose> estimate(const std::vector<cv::KeyPoint>& kpPrev,
                                          const std::vector<cv::KeyPoint>& kpCur,
                                          const std::vector<cv::DMatch>& matches) {
        if (matches.size() < 8) return std::nullopt;

        std::vector<cv::Point2f> ptsPrev, ptsCur;
        for (auto& m : matches) {
            ptsPrev.push_back(kpPrev[m.queryIdx].pt);
            ptsCur.push_back(kpCur[m.trainIdx].pt);
        }

        cv::Mat mask;
        cv::Mat E = cv::findEssentialMat(ptsPrev, ptsCur, K, cv::RANSAC, 0.999, ransacThreshold, 1000,mask);
        if (E.empty() || E.rows != 3 || E.cols != 3) return std::nullopt;

        cv::Mat R, t;
        cv::recoverPose(E, ptsPrev, ptsCur, K, R, t, mask);

        // recoverPose'un R,t'si "noktalari 1.kareden 2.kareye tasima" - TERSINI aliyoruz
        cv::Mat Rmotion = R.t();
        cv::Mat tmotion = -R.t() * t;
        return RelativePose{Rmotion, tmotion};
    }
};

class TrajectoryTracker {
public:
    cv::Mat curR = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat curT = cv::Mat::zeros(3, 1, CV_64F);
    std::vector<cv::Vec3d> positions{cv::Vec3d(0,0,0)};

    void update(const std::optional<RelativePose>& rel) {
        if (rel.has_value()) {
            curT = curT + curR * rel->t;
            curR = rel->R * curR;
        }
        positions.push_back(cv::Vec3d(curT.at<double>(0), curT.at<double>(1), curT.at<double>(2)));
    }
};
