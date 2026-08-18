#include "TimestampedDataSource.hpp"
#include "DataLoaders.hpp"
#include "Synchronizer.hpp"
#include "Config.hpp"
#include "FeatureExtractor.hpp"
#include "FeatureMatcher.hpp"
#include "PoseEstimator.hpp"
#include "ScalePropagation.hpp"
#include "BundleAdjustment.hpp"
#include "ImuFusion.hpp"
#include "MathUtils.hpp"
#include "Metrics.hpp"
#include "SimplePlotter.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <deque>

int main() {
    Config cfg = Config::load("config.json");

    ImageLoader images(cfg.imageFolder, cfg.nFrames);
    CSVGroundTruthSource gt(cfg.gtCsvPath);
    CSVImuSource imu(cfg.imuCsvPath);

    std::cout << "Toplam " << images.imageFiles.size() << " goruntu, "
              << gt.timestamps.size() << " GT, " << imu.timestamps.size() << " IMU satiri.\n";

    Synchronizer sync(images, {&gt});
    auto results = sync.alignAll();
    auto& gtAligned = results[0];

    FeatureExtractor extractor(cfg);
    FeatureMatcher matcher(cfg);
    PoseEstimator poseEstimator(cfg);
    TrajectoryTracker tracker;
    SimplePlotter plotter;

    std::optional<ScalePropagation> scaleProp;
    std::optional<BundleAdjustment> bundleAdj;
    if (cfg.translationMethod == "scale_propagation") scaleProp.emplace(cfg.K);
    if (cfg.translationMethod == "bundle_adjustment") bundleAdj.emplace(cfg.K, 5);

    std::optional<ImuFusion> imuFusion;
    if (cfg.useImuFusion) imuFusion.emplace(imu);

    cv::Mat grayPrev = cv::imread(images.imageFiles[0], cv::IMREAD_GRAYSCALE);
    std::vector<cv::KeyPoint> kpPrev, kpPrevPrev;
    cv::Mat desPrev, desCur;
    extractor.detect(grayPrev, kpPrev, desPrev);

    std::vector<cv::DMatch> matchesHist;
    std::deque<std::vector<cv::KeyPoint>> windowKps{kpPrev};
    std::deque<std::vector<cv::DMatch>> windowMatches;
    auto tStart = std::chrono::steady_clock::now();
    int processedFrames = 0;
    for (int i = 1; i < (int)images.imageFiles.size(); ++i) {
        processedFrames ++;
        cv::Mat grayCur = cv::imread(images.imageFiles[i], cv::IMREAD_GRAYSCALE);
        std::vector<cv::KeyPoint> kpCur;
        extractor.detect(grayCur, kpCur, desCur);

        std::optional<RelativePose> rel;
        std::vector<cv::DMatch> matches;

        if (!desPrev.empty() && !desCur.empty() && kpPrev.size() >= 8 && kpCur.size() >= 8) {
            matches = matcher.match(desPrev, desCur);
            rel = poseEstimator.estimate(kpPrev, kpCur, matches);

            if (rel && scaleProp && !kpPrevPrev.empty() && !matchesHist.empty()) {
                double scale = scaleProp->estimateScale(kpPrevPrev, kpPrev, kpCur, matchesHist, matches);
                rel->t *= scale;
            }
        }

        if (bundleAdj && !matches.empty()) {
            windowKps.push_back(kpCur);
            windowMatches.push_back(matches);
            if ((int)windowKps.size() == 5) {
                std::vector<std::vector<cv::KeyPoint>> kpVec(windowKps.begin(), windowKps.end());
                std::vector<std::vector<cv::DMatch>> mVec(windowMatches.begin(), windowMatches.end());
                auto baResult = bundleAdj->optimize(kpVec, mVec);
                if (baResult) {
                    auto& last = baResult->back();
                    auto& prevLast = (*baResult)[baResult->size()-2];
                    cv::Mat Rrel = last.R * prevLast.R.t();
                    cv::Mat trel = last.t - Rrel * prevLast.t;
                    rel = RelativePose{Rrel.t(), -Rrel.t() * trel};
                }
                windowKps.pop_front();
                windowMatches.pop_front();
            }
        }

        kpPrevPrev = kpPrev;
        matchesHist = matches;
        grayPrev = grayCur; kpPrev = kpCur; desPrev = desCur.clone();

        if (imuFusion && rel) {
            imuFusion->fuse(rel->R, rel->t, images.timestamps[i-1], images.timestamps[i]);
        }
        tracker.update(rel);

        if (i % cfg.updateEvery == 0 || i == (int)images.imageFiles.size() - 1) {
            int n = (int)tracker.positions.size();
            Eigen::MatrixXd vo(n, 3), gtM(n, 3);
            for (int k = 0; k < n; ++k) {
                vo(k,0)=tracker.positions[k][0]; vo(k,1)=tracker.positions[k][1]; vo(k,2)=tracker.positions[k][2];
                gtM(k,0)=gtAligned[k][0]; gtM(k,1)=gtAligned[k][1]; gtM(k,2)=gtAligned[k][2];
            }

            Eigen::MatrixXd T = umeyamaAlign(vo.transpose(), gtM.transpose());
            Eigen::MatrixXd rotated = T.block(0,0,3,3) * vo.transpose();       // (3,N)
            Eigen::MatrixXd voAligned = rotated.colwise() + T.block(0,3,3,1).col(0);  // (3,N)
            voAligned.transposeInPlace();                                       // (N,3)'e cevir

            double ate = computeATE(voAligned, gtM);

            std::vector<cv::Point2d> gtPts, voPts;
            for (int k = 0; k < n; ++k) {
                gtPts.emplace_back(gtM(k,0), gtM(k,1));
                voPts.emplace_back(voAligned(k,0), voAligned(k,1));
            }
            plotter.updateTrajectory(gtPts, voPts, "Kare " + std::to_string(i) + " ATE=" + std::to_string(ate));

            cv::Mat kpImg;
            cv::drawKeypoints(grayCur, kpCur, kpImg, cv::Scalar(0,255,255));
            cv::imshow("Kamera", kpImg);
            if (cv::waitKey(1) == 'q') break;
        }
    }
    auto tEnd = std::chrono::steady_clock::now();
    double totalSeconds = std::chrono::duration<double>(tEnd - tStart).count();
    double fps = processedFrames / totalSeconds;
    std::cout << "\n--- Sonuclar ---\n";
    std::cout << "Toplam sure: " << totalSeconds << " saniye\n";
    std::cout << "FPS: " << fps << "\n";
    std::cout << "Bitti.\n";
    cv::waitKey(0);
    return 0;
}
