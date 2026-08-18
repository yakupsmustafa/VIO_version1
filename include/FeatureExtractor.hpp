#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "Config.hpp"
#include <stdexcept>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
class FeatureExtractor {
public:
    cv::Ptr<cv::Feature2D> detector;

    explicit FeatureExtractor(const Config& cfg) {
        if (cfg.algorithm == "SIFT")       detector = cv::SIFT::create();
        else if (cfg.algorithm == "ORB")   detector = cv::ORB::create();
        else if (cfg.algorithm == "AKAZE") detector = cv::xfeatures2d::AKAZE::create(cv::xfeatures2d::AKAZE::DESCRIPTOR_MLDB, 0, 3, (float)cfg.akazeThreshold);
    }

    void detect(const cv::Mat& grayImage, std::vector<cv::KeyPoint>& kps, cv::Mat& descs) {
        detector->detectAndCompute(grayImage, cv::noArray(), kps, descs);
    }
};
