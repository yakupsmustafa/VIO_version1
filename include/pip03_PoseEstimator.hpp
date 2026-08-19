#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <optional>
#include <vector>

// Iki kare arasindaki ham (zincirlenmemis) goreli poz
struct RelativePose_Raw {
    cv::Mat R; // 3x3 rotasyon
    cv::Mat t; // 3x1 oteleme (SADECE YON - gercek olcek bilinmiyor, essential matrix'ten geliyor)
};

using RelativePose = RelativePose_Raw;

// cv::findEssentialMat + cv::recoverPose'un TUM parametreleri
struct EssentialMatParams {
    // ---- cv::findEssentialMat parametreleri ----
    int method = cv::RANSAC;      // cv::RANSAC ya da cv::LMEDS
    double prob = 0.999;          // istenen basari olasiligi (guven araligi) - yuksek = daha cok iterasyon
    double threshold = 1.0;       // inlier/outlier ayrimi icin piksel mesafesi esigi (epipolar kisitlamaya uzaklik)
    int maxIters = 1000;          // RANSAC'in maksimum deneme sayisi

    // ---- cv::recoverPose parametreleri ----
    // (E, points1, points2, cameraMatrix, R, t, distanceThresh, mask, triangulatedPoints) overload'u kullanilir
    double recoverPoseDistanceThresh = 50.0; // triangule edilen noktalar icin gecerli sayilacak max derinlik (kamera birimi)
    bool computeTriangulatedPoints = false;  // true ise triangule edilen 3B noktalar da geri dondurulur
};

class PoseEstimator {
public:
    cv::Mat K;
    EssentialMatParams params;

    PoseEstimator(const cv::Mat& K_, const EssentialMatParams& p = EssentialMatParams())
        : K(K_), params(p) {}

    // kp1: onceki kare keypoint'leri, kp2: sonraki kare keypoint'leri
    // matches: kp1 -> kp2 eslesmeleri (FeatureMatcher::match ciktisi)
    // Donen: basariliysa RelativePose_Raw (R,t), degilse std::nullopt
    std::optional<RelativePose_Raw> estimate(
            const std::vector<cv::KeyPoint>& kp1,
            const std::vector<cv::KeyPoint>& kp2,
            const std::vector<cv::DMatch>& matches) {

        if (matches.size() < 8) return std::nullopt; // 8-point algoritmasi icin minimum

        std::vector<cv::Point2f> pts1, pts2;
        pts1.reserve(matches.size());
        pts2.reserve(matches.size());
        for (auto& m : matches) {
            pts1.push_back(kp1[m.queryIdx].pt);
            pts2.push_back(kp2[m.trainIdx].pt);
        }

        cv::Mat inlierMask;
        cv::Mat E = cv::findEssentialMat(
            pts1, pts2, K,
            params.method,
            params.prob,
            params.threshold,
            params.maxIters,
            inlierMask
        );

        if (E.empty()) return std::nullopt;

        cv::Mat R, t;
        cv::Mat triangulatedPoints;
        int inliers = cv::recoverPose(
            E, pts1, pts2, K, R, t,
            params.recoverPoseDistanceThresh,
            inlierMask,
            params.computeTriangulatedPoints ? triangulatedPoints : cv::noArray()
        );

        if (inliers < 8) return std::nullopt; // cok az inlier -> guvenilmez sonuc

        RelativePose_Raw result;
        result.R = R;
        result.t = t;
        return result;
    }
};
