//
//  main.cpp
//  VisualOdometry
//
//  Created by asd on 18.08.2026.
//
#include <opencv2/opencv.hpp>
#include <iostream>
#include "Metrics.h"
#include "pip01_FeatureExtractor.h"
#include "DataLoader.h"
#include "TimestampedDataSource.hpp"
#include "Synchronizer.hpp"
#include "pip02_Matcher.hpp"
#include "pip03_PoseEstimator.hpp"
#include "Test.hpp"

// ---- Veri yollarini tutan yapi ----
struct DatasetPaths {
    std::string imageFolder;
    std::string imuCsvPath;
    std::string gtCsvPath;
};

DatasetPaths buildDatasetPaths(const std::string& datasetRoot) {
    DatasetPaths paths;
    paths.imageFolder = datasetRoot + "/cam0_frames";
    paths.imuCsvPath  = datasetRoot + "/imu.csv";
    paths.gtCsvPath    = datasetRoot + "/ground_truth.csv";
    return paths;
}

// ---- Goruntuleri yukler ----
ImageLoader loadImages(const std::string& imageFolder, int nFrames) {
    std::cout << "Goruntuler yukleniyor...\n";
    ImageLoader images(imageFolder, nFrames);
    std::cout << "Toplam goruntu: " << images.imageFiles.size() << "\n";
    return images;
}

// ---- IMU verisini yukler ----
CSVImuSource loadImu(const std::string& imuCsvPath) {
    std::cout << "IMU verisi yukleniyor...\n";
    return CSVImuSource(imuCsvPath);
}

// ---- Ground truth verisini yukler ----
CSVGroundTruthSource loadGroundTruth(const std::string& gtCsvPath) {
    std::cout << "Ground truth yukleniyor...\n";
    return CSVGroundTruthSource(gtCsvPath);
}

// ---- IMU ve GT'yi goruntu zaman damgalarina hizalar ----
std::vector<std::vector<std::vector<double>>> synchronizeData(
        const ImageLoader& images, const CSVImuSource& imu, const CSVGroundTruthSource& gt) {
    Synchronizer sync(images, {&imu, &gt});
    auto aligned = sync.alignAll();
    std::cout << "Hizalanan IMU orneği: " << aligned[0].size() << "\n";
    std::cout << "Hizalanan GT orneği: "  << aligned[1].size() << "\n";
    return aligned;
}

// ---- ORB extractor'i olusturur ----
ORBExtractor buildExtractor() {
    ORBParams params;
    params.nfeatures = 2000;
    params.scaleFactor = 1.2f;
    params.nlevels = 8;
    params.edgeThreshold = 31;
    params.firstLevel = 0;
    params.WTA_K = 2;
    params.scoreType = cv::ORB::HARRIS_SCORE;
    params.patchSize = 31;
    params.fastThreshold = 15;
    return ORBExtractor(params);
}

// ---- Matcher'i olusturur ----
FeatureMatcher buildMatcher() {
    MatcherParams mparams;
    mparams.normType = cv::NORM_HAMMING;
    mparams.crossCheck = false;
    mparams.useRatioTest = true;
    mparams.ratioThresh = 0.75f;
    mparams.knnK = 2;
    mparams.useRadiusMatch = false;
    mparams.maxDistance = 30.0f;
    return FeatureMatcher(mparams);
}

// ---- Kamera kalibrasyon matrisini olusturur ----
// NOT: Bu degerler EuRoC MH01 cam0'in bilinen yaklasik kalibrasyonudur.
// Kendi ground_truth.csv/imu.csv ile birlikte gelen GERCEK kalibrasyon dosyan
// varsa (ornegin cam0/sensor.yaml gibi) fx,fy,cx,cy degerlerini oradan al.
cv::Mat buildCameraMatrix() {
    double fx = 458.654, fy = 457.296;
    double cx = 367.215, cy = 248.375;
    return (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0,  0,  1);
}

// ---- PoseEstimator'i olusturur ----
PoseEstimator buildPoseEstimator(const cv::Mat& K) {
    EssentialMatParams params;
    params.method = cv::RANSAC;
    params.prob = 0.999;
    params.threshold = 1.0;
    params.maxIters = 1000;
    params.recoverPoseDistanceThresh = 50.0;
    params.computeTriangulatedPoints = false;
    return PoseEstimator(K, params);
}

// ---- TrajectoryViewer'i olusturur ----
TrajectoryViewer buildViewer() {
    ViewerParams vparams;
    vparams.showFrame = true;
    vparams.showTrajectory = true;
    vparams.canvasSize = 800;
    vparams.trajScale = 3.0;
    vparams.showMetrics = true;
    vparams.computeLiveATE = true;
    vparams.ateUpdateInterval = 10;
    return TrajectoryViewer(vparams);
}

// ---- Tum kareler icin feature extraction + ardisik matching + poz zincirleme + canli gorsellestirme ----
void processFrames(const ImageLoader& images, ORBExtractor& extractor,
                    FeatureMatcher& matcher, PoseEstimator& poseEst,
                    const std::vector<std::vector<double>>& alignedGt,
                    TrajectoryViewer& viewer) {
    std::vector<cv::KeyPoint> prevKps;
    cv::Mat prevDescs;
    bool hasPrev = false;

    // Mutlak poz zincirleme icin baslangic degerleri (kare 0 = referans/orijin)
    cv::Mat chainedR = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat chainedT = cv::Mat::zeros(3, 1, CV_64F);

    for (size_t i = 0; i < images.imageFiles.size(); ++i) {
        cv::Mat image = cv::imread(images.imageFiles[i], cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "HATA: goruntu okunamadi: " << images.imageFiles[i] << "\n";
            continue;
        }

        std::vector<cv::KeyPoint> kps;
        cv::Mat descs;
        extractor.detect(image, kps, descs);
        std::cout << "Kare " << i << " -> bulunan kose sayisi: " << kps.size() << "\n";

        if (hasPrev) {
            std::vector<cv::DMatch> matches = matcher.match(prevDescs, descs);
            std::cout << "  Kare " << (i - 1) << " -> " << i << " eslesme sayisi: " << matches.size() << "\n";

            auto pose = poseEst.estimate(prevKps, kps, matches);
            if (pose) {
                // ---- Goreli pozu mutlak pozla zincirle ----
                chainedT = pose->R * chainedT + pose->t;
                chainedR = pose->R * chainedR;
            } else {
                std::cout << "  UYARI: poz tahmini basarisiz (yetersiz eslesme/inlier), onceki poz korunuyor\n";
            }
        }

        // ---- VO ve GT konumlarini viewer'a gonder (TEK SATIR CAGRI) ----
        cv::Point3d voPos(chainedT.at<double>(0), chainedT.at<double>(1), chainedT.at<double>(2));
        std::optional<cv::Point3d> gtPos;
        if (i < alignedGt.size() && alignedGt[i].size() == 3) {
            gtPos = cv::Point3d(alignedGt[i][0], alignedGt[i][1], alignedGt[i][2]);
        }
        viewer.update(image, voPos, gtPos);

        prevKps = kps;
        prevDescs = descs;
        hasPrev = true;
    }
}

int main() {
    const std::string datasetRoot = "../EUROC_MH01_Easy";
    const int nFrames = 800;

    DatasetPaths paths = buildDatasetPaths(datasetRoot);

    ImageLoader images = loadImages(paths.imageFolder, nFrames);
    CSVImuSource imu = loadImu(paths.imuCsvPath);
    CSVGroundTruthSource gt = loadGroundTruth(paths.gtCsvPath);

    auto aligned = synchronizeData(images, imu, gt);
    const auto& alignedGt = aligned[1]; // 0: imu, 1: gt (synchronizeData'daki sira ile ayni)

    ORBExtractor extractor = buildExtractor();
    FeatureMatcher matcher = buildMatcher();
    cv::Mat K = buildCameraMatrix();
    PoseEstimator poseEst = buildPoseEstimator(K);
    TrajectoryViewer viewer = buildViewer();

    processFrames(images, extractor, matcher, poseEst, alignedGt, viewer);

    std::cout << "Bitti.\n";
    return 0;
}
