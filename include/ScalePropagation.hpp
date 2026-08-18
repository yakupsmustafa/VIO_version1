#pragma once
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <optional>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
// --- ONCE bu iki tanim, class'tan ONCE gelmeli ---
struct RelativePose_Raw { cv::Mat R, t; };

inline std::optional<RelativePose_Raw> computeRawRelativePose(
        const cv::Mat& K, const std::vector<cv::KeyPoint>& kpA, const std::vector<cv::KeyPoint>& kpB,
        const std::vector<cv::DMatch>& matches) {
    std::vector<cv::Point2f> ptsA, ptsB;
    for (auto& m : matches) { ptsA.push_back(kpA[m.queryIdx].pt); ptsB.push_back(kpB[m.trainIdx].pt); }
    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(ptsA, ptsB, K, cv::RANSAC, 0.999, 1.0, 1000,mask);
    if (E.empty() || E.rows != 3) return std::nullopt;
    cv::Mat R, t;
    cv::recoverPose(E, ptsA, ptsB, K, R, t, mask);
    return RelativePose_Raw{R, t};
}

// --- SONRA class tanimi ---
class ScalePropagation {
public:
    cv::Mat K;
    double prevScale = 1.0;

    explicit ScalePropagation(const cv::Mat& K_) : K(K_) {}

    double estimateScale(const std::vector<cv::KeyPoint>& kp0, const std::vector<cv::KeyPoint>& kp1,
                          const std::vector<cv::KeyPoint>& kp2,
                          const std::vector<cv::DMatch>& m01, const std::vector<cv::DMatch>& m12,
                          int maxPoints = 20) {
        auto rel = computeRawRelativePose(K, kp0, kp1, m01);
        auto rel2 = computeRawRelativePose(K, kp1, kp2, m12);
        if (!rel || !rel2) return prevScale;

        std::unordered_map<int,int> trainToQuery;
        for (auto& m : m01) trainToQuery[m.trainIdx] = m.queryIdx;

        std::vector<std::array<int,3>> common;
        for (auto& m : m12) {
            auto it = trainToQuery.find(m.queryIdx);
            if (it != trainToQuery.end()) common.push_back({it->second, m.queryIdx, m.trainIdx});
        }
        if ((int)common.size() < 3) return prevScale;
        if ((int)common.size() > maxPoints) common.resize(maxPoints);

        std::vector<cv::Point2f> p0, p1, p2;
        for (auto& c : common) {
            p0.push_back(kp0[c[0]].pt); p1.push_back(kp1[c[1]].pt); p2.push_back(kp2[c[2]].pt);
        }

        cv::Mat P0, P1, P2, RT0, RT1, RT2;
        cv::hconcat(cv::Mat::eye(3,3,CV_64F), cv::Mat::zeros(3,1,CV_64F), RT0);
        cv::hconcat(rel->R, rel->t, RT1);
        cv::hconcat(rel2->R, rel2->t, RT2);
        P0 = K * RT0; P1 = K * RT1; P2 = K * RT2;

        cv::Mat pts4d_1, pts4d_2;
        cv::triangulatePoints(P0, P1, p0, p1, pts4d_1);
        cv::triangulatePoints(P0, P2, p1, p2, pts4d_2);

        auto toCloud = [](cv::Mat& pts4d) {
            std::vector<cv::Point3d> cloud;
            for (int i = 0; i < pts4d.cols; ++i) {
                double w = pts4d.at<float>(3, i);
                cloud.emplace_back(pts4d.at<float>(0,i)/w, pts4d.at<float>(1,i)/w, pts4d.at<float>(2,i)/w);
            }
            return cloud;
        };
        auto cloud1 = toCloud(pts4d_1);
        auto cloud2 = toCloud(pts4d_2);

        std::vector<double> ratios;
        int n = (int)cloud1.size();
        for (int i = 0; i < n; ++i)
            for (int j = i+1; j < n; ++j) {
                double d1 = cv::norm(cloud1[i] - cloud1[j]);
                double d2 = cv::norm(cloud2[i] - cloud2[j]);
                if (d2 > 1e-6) ratios.push_back(d1/d2);
            }
        if (ratios.empty()) return prevScale;
        std::sort(ratios.begin(), ratios.end());
        prevScale = ratios[ratios.size()/2];
        return prevScale;
    }
};
