#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

// ORB icin tum ayarlanabilir parametreleri tutan yapi
struct ORBParams {
    int nfeatures = 1000;
    float scaleFactor = 1.2f;
    int nlevels = 8;
    int edgeThreshold = 31;
    int firstLevel = 0;
    int WTA_K = 2;
    cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
    int patchSize = 31;
    int fastThreshold = 20;
};

class ORBExtractor {
public:
    cv::Ptr<cv::ORB> detector;

    explicit ORBExtractor(const ORBParams& p = ORBParams()) {
        detector = cv::ORB::create(
            p.nfeatures,
            p.scaleFactor,
            p.nlevels,
            p.edgeThreshold,
            p.firstLevel,
            p.WTA_K,
            p.scoreType,
            p.patchSize,
            p.fastThreshold
        );
    }
    void detect(const cv::Mat& grayImage, std::vector<cv::KeyPoint>& kps, cv::Mat& descs) {
        detector->detectAndCompute(grayImage, cv::noArray(), kps, descs);
    }
};
