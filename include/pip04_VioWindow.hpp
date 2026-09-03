#pragma once
#include <opencv2/opencv.hpp>
#include <array>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include "TimestampedDataSource.hpp"
#include "Imupreintegration.hpp"
#include "Viobundleadjustment.hpp"

// ---- IMU olcumlerini govde (IMU) cercevesinden kamera cercevesine dondurur ----
// R_BS: kameradan govdeye rotasyon (main.cpp:buildCameraToBodyRotation). R_CB = R_BS^T (govdeden kameraya).
// ImuFactor/ImuPreintegrator, FrameState.pose ile AYNI (kamera) cercevesinde calismak ZORUNDA; aksi halde
// IMU'nun govde-cercevesindeki olcumleri optimize edilen kamera rotasyonuyla tutarsiz olur (R_BS kimlikten
// uzak, ~90 derecelik gercek bir rotasyon oldugu icin bu atlanamaz).
inline TimestampedDataSource rotateImuToCameraFrame(const TimestampedDataSource& rawImu, const cv::Mat& R_BS) {
    cv::Matx33d R_CB = cv::Matx33d(R_BS).t(); // govdeden kameraya
    TimestampedDataSource out;
    out.timestamps = rawImu.timestamps;
    out.values.reserve(rawImu.values.size());
    for (const auto& v : rawImu.values) {
        cv::Vec3d accelBody(v[0], v[1], v[2]);
        cv::Vec3d gyroBody(v[3], v[4], v[5]);
        cv::Vec3d accelCam = R_CB * accelBody;
        cv::Vec3d gyroCam  = R_CB * gyroBody;
        out.values.push_back({accelCam[0], accelCam[1], accelCam[2], gyroCam[0], gyroCam[1], gyroCam[2]});
    }
    return out;
}

// ---- Sekans basindaki (varsayilan durgun) ilk N ivme orneginin ortalamasindan yercekimi yonunu kestirir ----
// Durgun haldeyken (R=I, kare-0 kendi cercevesi): 0 = g + R*accel -> g = -mean(accel).
// Buyuklugu 9.81 m/s^2'ye normalize edilir (ivmeolcer olcek hatalarina karsi).
inline cv::Vec3d estimateGravityCameraFrame(const TimestampedDataSource& rotatedImu, int nStaticSamples = 200) {
    int n = std::min((int)rotatedImu.values.size(), nStaticSamples);
    cv::Vec3d sum(0, 0, 0);
    for (int i = 0; i < n; ++i) {
        sum += cv::Vec3d(rotatedImu.values[i][0], rotatedImu.values[i][1], rotatedImu.values[i][2]);
    }
    cv::Vec3d mean = sum / std::max(n, 1);
    double norm = cv::norm(mean);
    if (norm < 1e-6) return cv::Vec3d(0, 0, 9.81); // guvenlik: durgun veri yoksa varsayilana don
    return -(mean / norm) * 9.81;
}

// ---- FrameState.pose'dan (angleAxis+t, world->camera) gercek kamera merkezini cikarir: C = -R^T * t ----
// (Adim 1'de main.cpp icin dogrulanan formulun aynisi.)
inline cv::Point3d cameraCenterFromPose(const std::array<double,6>& pose) {
    cv::Mat rvec = (cv::Mat_<double>(3,1) << pose[0], pose[1], pose[2]);
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    cv::Mat t = (cv::Mat_<double>(3,1) << pose[3], pose[4], pose[5]);
    cv::Mat C = -R.t() * t;
    return cv::Point3d(C.at<double>(0), C.at<double>(1), C.at<double>(2));
}

// ---- K * [R|t] projeksiyon matrisini pose'dan (angleAxis+t, world->camera) kurar ----
inline cv::Mat projectionMatrixFromPose(const cv::Mat& K, const std::array<double,6>& pose) {
    cv::Mat rvec = (cv::Mat_<double>(3,1) << pose[0], pose[1], pose[2]);
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    cv::Mat Rt(3, 4, CV_64F);
    R.copyTo(Rt(cv::Rect(0, 0, 3, 3)));
    Rt.at<double>(0,3) = pose[3]; Rt.at<double>(1,3) = pose[4]; Rt.at<double>(2,3) = pose[5];
    return K * Rt;
}

// ---- Pencere icindeki ardisik kare eslesmelerini zincirleyip TUM pencerede hayatta kalan track'leri kurar ----
// pairwiseMatches[s]: local kare s -> s+1 eslesmeleri (queryIdx: kare s, trainIdx: kare s+1)
// Donen: tracks[trackIdx][frameIdx] = kpList[frameIdx] icindeki keypoint indeksi (TUM frameIdx'ler icin dolu,
// VIOBundleAdjustment::optimize'in gerektirdigi "full coverage" kisitini insa yoluyla saglar).
inline std::vector<std::vector<int>> buildFullCoverageTracks(
        const std::vector<std::vector<cv::DMatch>>& pairwiseMatches,
        int firstFrameKpCount) {

    int windowSize = (int)pairwiseMatches.size() + 1;
    std::vector<std::unordered_map<int,int>> matchMap(pairwiseMatches.size());
    for (size_t s = 0; s < pairwiseMatches.size(); ++s) {
        for (const auto& m : pairwiseMatches[s]) matchMap[s][m.queryIdx] = m.trainIdx;
    }

    std::vector<std::vector<int>> tracks;
    for (int k = 0; k < firstFrameKpCount; ++k) {
        std::vector<int> chain;
        chain.reserve(windowSize);
        chain.push_back(k);
        int idx = k;
        bool alive = true;
        for (int s = 0; s < (int)pairwiseMatches.size(); ++s) {
            auto it = matchMap[s].find(idx);
            if (it == matchMap[s].end()) { alive = false; break; }
            idx = it->second;
            chain.push_back(idx);
        }
        if (alive) tracks.push_back(std::move(chain));
    }
    return tracks;
}

// ---- Pencerenin ilk/son karesindeki IMU-seed'lenmis pozlarla track'leri triangule edip filtreler ----
// tracks ve landmarks ES ZAMANLI olarak sadece gecerli (sonlu + iki kamera onunde) noktalarla daraltilir.
inline void triangulateAndFilterWindowTracks(
        const cv::Mat& K,
        const std::array<double,6>& poseFirst,
        const std::array<double,6>& poseLast,
        const std::vector<std::vector<cv::KeyPoint>>& kpList,
        std::vector<std::vector<int>>& tracks,
        std::vector<std::array<double,3>>& landmarks) {

    landmarks.clear();
    if (tracks.empty()) return;

    cv::Mat P0 = projectionMatrixFromPose(K, poseFirst);
    cv::Mat Plast = projectionMatrixFromPose(K, poseLast);
    int lastFrame = (int)kpList.size() - 1;

    std::vector<cv::Point2f> pts0, ptsLast;
    pts0.reserve(tracks.size());
    ptsLast.reserve(tracks.size());
    for (auto& tr : tracks) {
        pts0.push_back(kpList[0][tr.front()].pt);
        ptsLast.push_back(kpList[lastFrame][tr.back()].pt);
    }

    cv::Mat pts4D;
    cv::triangulatePoints(P0, Plast, pts0, ptsLast, pts4D);

    cv::Mat rvec0 = (cv::Mat_<double>(3,1) << poseFirst[0], poseFirst[1], poseFirst[2]);
    cv::Mat Rfirst; cv::Rodrigues(rvec0, Rfirst);
    cv::Mat tfirst = (cv::Mat_<double>(3,1) << poseFirst[3], poseFirst[4], poseFirst[5]);
    cv::Mat rvecL = (cv::Mat_<double>(3,1) << poseLast[0], poseLast[1], poseLast[2]);
    cv::Mat Rlast; cv::Rodrigues(rvecL, Rlast);
    cv::Mat tlast = (cv::Mat_<double>(3,1) << poseLast[3], poseLast[4], poseLast[5]);

    std::vector<std::vector<int>> keptTracks;
    keptTracks.reserve(tracks.size());

    for (int i = 0; i < pts4D.cols; ++i) {
        double w = pts4D.at<float>(3, i);
        if (std::abs(w) < 1e-9) continue;
        double x = pts4D.at<float>(0, i) / w;
        double y = pts4D.at<float>(1, i) / w;
        double z = pts4D.at<float>(2, i) / w;
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;

        cv::Mat Xw = (cv::Mat_<double>(3,1) << x, y, z);
        cv::Mat Xc0 = Rfirst * Xw + tfirst;
        cv::Mat XcL = Rlast * Xw + tlast;
        if (Xc0.at<double>(2) <= 0 || XcL.at<double>(2) <= 0) continue; // cheirality: her iki kamera onunde olmali

        keptTracks.push_back(tracks[i]);
        landmarks.push_back({x, y, z});
    }
    tracks = std::move(keptTracks);
}

// ---- Pencerenin ilk karesinden (o anki) son karesine, tam-kapsama izlerinin medyan piksel yer
// degistirmesi (paralaks) ----
// Paralaks-tabanli degisken pencere kapatma kararinda kullanilir (main.cpp): kare SAYISI yerine
// kameranin GERCEKTEN ne kadar hareket ettigini olcer. Medyan (ortalama degil) tercih edildi:
// birkac yanlis-eslesen/dinamik-nesne izi tek basina sonucu carpitmasin diye.
inline double computeMedianParallaxPx(
        const std::vector<std::vector<cv::KeyPoint>>& kpList,
        const std::vector<std::vector<int>>& tracks) {
    if (tracks.empty() || kpList.empty()) return 0.0;
    int lastFrame = (int)kpList.size() - 1;

    std::vector<double> disp;
    disp.reserve(tracks.size());
    for (const auto& tr : tracks) {
        cv::Point2f p0 = kpList[0][tr.front()].pt;
        cv::Point2f pl = kpList[lastFrame][tr.back()].pt;
        disp.push_back(cv::norm(pl - p0));
    }
    size_t mid = disp.size() / 2;
    std::nth_element(disp.begin(), disp.begin() + mid, disp.end());
    return disp[mid];
}

// ---- Bir pencerenin baslangic durumlarini (anchor'dan itibaren) IMU dead-reckoning ile doldurur ----
// world->camera parametrizasyonuna gore dogru turetilmis tekrarlama bagintisi (bkz. plan / ImuFactor'un
// residual'larindan ters cozum):
//   R[j] = deltaR^T * R[j-1]                                  (R[j-1]*deltaR DEGIL - ters!)
//   v[j] = v[j-1] + g*dt + R[j-1]^T * deltaV
//   C[j] = C[j-1] + v[j-1]*dt + 0.5*g*dt^2 + R[j-1]^T * deltaP  (C = fiziksel kamera merkezi, t DEGIL)
//   t[j] = -R[j] * C[j]                                        (FrameState.pose icin geri cevir)
inline std::vector<VIOBundleAdjustment::FrameState> seedWindowStatesFromImu(
        ImuPreintegrator& imuPreintCam,
        const VIOBundleAdjustment::FrameState& anchor,
        const std::vector<int64_t>& frameTimestamps,
        const cv::Vec3d& gravityCam) {

    int n = (int)frameTimestamps.size();
    std::vector<VIOBundleAdjustment::FrameState> states(n);
    states[0] = anchor;

    cv::Mat rvecAnchor = (cv::Mat_<double>(3,1) << anchor.pose[0], anchor.pose[1], anchor.pose[2]);
    cv::Mat Rmat0;
    cv::Rodrigues(rvecAnchor, Rmat0);
    cv::Matx33d R = cv::Matx33d(Rmat0);
    cv::Vec3d v(anchor.vel[0], anchor.vel[1], anchor.vel[2]);
    cv::Vec3d tAnchor(anchor.pose[3], anchor.pose[4], anchor.pose[5]);
    cv::Vec3d C = -R.t() * tAnchor;

    for (int j = 1; j < n; ++j) {
        cv::Vec3d bg(states[j-1].bg[0], states[j-1].bg[1], states[j-1].bg[2]);
        cv::Vec3d ba(states[j-1].ba[0], states[j-1].ba[1], states[j-1].ba[2]);
        PreintegratedImuData preint = imuPreintCam.preintegrate(frameTimestamps[j-1], frameTimestamps[j], bg, ba);

        double dt = preint.deltaT;
        cv::Matx33d Rnew;
        cv::Vec3d vNew, Cnew;
        if (dt > 0) {
            Rnew = preint.deltaR.t() * R;
            vNew = v + gravityCam * dt + R.t() * preint.deltaV;
            Cnew = C + v * dt + 0.5 * gravityCam * dt * dt + R.t() * preint.deltaP;
        } else {
            // yetersiz IMU orneği (dt<=0): poz/hiz degismedi varsay (guvenli fallback)
            Rnew = R;
            vNew = v;
            Cnew = C;
        }

        cv::Vec3d tNew = -(Rnew * Cnew);

        cv::Mat rvecNew;
        cv::Rodrigues(cv::Mat(Rnew), rvecNew);

        states[j].pose = {rvecNew.at<double>(0), rvecNew.at<double>(1), rvecNew.at<double>(2),
                           tNew[0], tNew[1], tNew[2]};
        states[j].vel = {vNew[0], vNew[1], vNew[2]};
        states[j].ba = states[j-1].ba;
        states[j].bg = states[j-1].bg;

        R = Rnew; v = vNew; C = Cnew;
    }

    return states;
}
