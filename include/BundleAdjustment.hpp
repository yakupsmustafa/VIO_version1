#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
// Reprojection error: Ceres'in kendi autodiff'i turevi otomatik hesapliyor,
// Python'daki gibi manuel/sayisal Jacobian yazmaya GEREK YOK.
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

class BundleAdjustment {
public:
    cv::Mat K;
    int windowSize;

    BundleAdjustment(const cv::Mat& K_, int windowSize_ = 5) : K(K_), windowSize(windowSize_) {}

    // kpList: her karenin keypoint'leri, matchesList: ardisik eslesmeler (n-1 tane)
    // Donen: her karenin (R,t) optimize edilmis pozu, bos ise basarisiz
    std::optional<std::vector<RelativePose>> optimize(
            const std::vector<std::vector<cv::KeyPoint>>& kpList,
            const std::vector<std::vector<cv::DMatch>>& matchesList) {

        int n = windowSize;
        auto tracks = chainTracks(matchesList, kpList[0].size());
        if ((int)tracks.size() > 30) tracks.resize(30);
        if ((int)tracks.size() < 6) return std::nullopt;

        // Baslangic pozlari: zincirleme (ham, ters cevrilmemis)
        std::vector<RelativePose_Raw> rawPoses;
        for (int i = 0; i < (int)matchesList.size(); ++i) {
            auto rel = computeRawRelativePose(K, kpList[i], kpList[i+1], matchesList[i]);
            if (!rel) return std::nullopt;
            rawPoses.push_back(*rel);
        }

        std::vector<cv::Mat> chainedR(1, cv::Mat::eye(3,3,CV_64F));
        std::vector<cv::Mat> chainedT(1, cv::Mat::zeros(3,1,CV_64F));
        for (auto& rp : rawPoses) {
            chainedR.push_back(rp.R * chainedR.back());
            chainedT.push_back(rp.R * chainedT.back() + rp.t);
        }

        // Triangulasyon (ilk iki kareden)
        cv::Mat P0, P1, RT1;
        cv::hconcat(cv::Mat::eye(3,3,CV_64F), cv::Mat::zeros(3,1,CV_64F), P0);
        cv::hconcat(rawPoses[0].R, rawPoses[0].t, RT1);
        P0 = K * P0; P1 = K * RT1;

        std::vector<cv::Point2f> p0, p1;
        for (auto& tr : tracks) { p0.push_back(kpList[0][tr[0]].pt); p1.push_back(kpList[1][tr[1]].pt); }
        cv::Mat pts4d;
        cv::triangulatePoints(P0, P1, p0, p1, pts4d);

        // Ceres parametre bloklari
        std::vector<std::array<double,6>> poseParams(n); // [0]=birim (sabit), [1..n-1]=optimize edilecek
        for (int i = 0; i < n; ++i) {
            double aa[3];
            cv::Mat rvec;
            cv::Rodrigues(chainedR[i], rvec);
            for (int k=0;k<3;++k) aa[k] = rvec.at<double>(k);
            poseParams[i] = {aa[0], aa[1], aa[2],
                              chainedT[i].at<double>(0), chainedT[i].at<double>(1), chainedT[i].at<double>(2)};
        }

        std::vector<std::array<double,3>> landmarks(tracks.size());
        for (size_t ti = 0; ti < tracks.size(); ++ti) {
            double w = pts4d.at<float>(3, (int)ti);
            landmarks[ti] = {pts4d.at<float>(0,(int)ti)/w, pts4d.at<float>(1,(int)ti)/w, pts4d.at<float>(2,(int)ti)/w};
        }

        ceres::Problem problem;
        double fx = K.at<double>(0,0), fy = K.at<double>(1,1), cx = K.at<double>(0,2), cy = K.at<double>(1,2);

        for (size_t ti = 0; ti < tracks.size(); ++ti) {
            for (int fi = 0; fi < n; ++fi) {
                cv::Point2f obs = kpList[fi][tracks[ti][fi]].pt;
                ceres::CostFunction* cost = ReprojectionError::create(obs.x, obs.y, fx, fy, cx, cy);
                problem.AddResidualBlock(cost, nullptr, poseParams[fi].data(), landmarks[ti].data());
            }
        }
        // Ilk pozu sabitle (referans kare, optimize edilmesin)
        problem.SetParameterBlockConstant(poseParams[0].data());

        ceres::Solver::Options options;
        options.max_num_iterations = 50;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        std::vector<RelativePose> result;
        for (int i = 0; i < n; ++i) {
            cv::Mat rvec = (cv::Mat_<double>(3,1) << poseParams[i][0], poseParams[i][1], poseParams[i][2]);
            cv::Mat R; cv::Rodrigues(rvec, R);
            cv::Mat t = (cv::Mat_<double>(3,1) << poseParams[i][3], poseParams[i][4], poseParams[i][5]);
            result.push_back({R, t});
        }
        return result;
    }

private:
    std::vector<std::vector<int>> chainTracks(const std::vector<std::vector<cv::DMatch>>& matchesList, size_t nKp0) {
        std::vector<std::unordered_map<int,int>> maps;
        for (auto& m_ : matchesList) {
            std::unordered_map<int,int> mp;
            for (auto& m : m_) mp[m.queryIdx] = m.trainIdx;
            maps.push_back(mp);
        }
        std::vector<std::vector<int>> tracks;
        for (size_t idx0 = 0; idx0 < nKp0; ++idx0) {
            std::vector<int> track{(int)idx0};
            int cur = (int)idx0;
            bool ok = true;
            for (auto& mp : maps) {
                auto it = mp.find(cur);
                if (it == mp.end()) { ok = false; break; }
                cur = it->second;
                track.push_back(cur);
            }
            if (ok) tracks.push_back(track);
        }
        return tracks;
    }
};
