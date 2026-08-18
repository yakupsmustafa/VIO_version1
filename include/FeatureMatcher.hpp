#pragma once
#include <opencv2/opencv.hpp>
#include "Config.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
class FeatureMatcher {
public:
    cv::Ptr<cv::DescriptorMatcher> matcher;

    explicit FeatureMatcher(const Config& cfg) {
        int normType = (cfg.algorithm == "SIFT") ? cv::NORM_L2 : cv::NORM_HAMMING;
        if (cfg.matcherType == "BF") {
            matcher = cv::BFMatcher::create(normType, /*crossCheck=*/true);
        } else {
            // FLANN: binary icin LSH, float icin KDTree
            if (cfg.algorithm == "SIFT")
                matcher = cv::FlannBasedMatcher::create();
            else
                matcher = cv::makePtr<cv::FlannBasedMatcher>(
                    cv::makePtr<cv::flann::LshIndexParams>(6, 12, 1));
        }
    }

    std::vector<cv::DMatch> match(const cv::Mat& descPrev, const cv::Mat& descCur) {
        std::vector<cv::DMatch> matches;
        matcher->match(descPrev, descCur, matches);
        return matches;
    }
};
