#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_map>
#include <cstdint>

// ---- KLT (Lucas-Kanade) optik akis parametreleri ----
struct KltParams {
    int    maxCorners            = 300;   // ORB'un 3000'ine kiyasla cok daha kucuk havuz yeterli:
                                           // KLT bir siçramayi hayatta kalmak icin devasa yedeklemeye
                                           // ihtiyac duymuyor (~%90-95 tekli-siçrama hayatta kalma orani).
    double qualityLevel          = 0.01;
    double minDistance           = 15.0;  // goodFeaturesToTrack yeni-yeni nokta araligi
    double replenishMaskRadius   = 30.0;  // yeni-MEVCUT nokta araligi (mevcut noktalarin etrafi maskelenir)
    int    replenishTriggerCount = 150;   // havuz bu sayinin altina inince takviye yapilir
    cv::Size winSize  = cv::Size(21, 21);
    int      maxLevel = 3;
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
    double ransacReprojThreshold = 1.0;
    double ransacConfidence      = 0.99;
    int    minPointsForRansac    = 8;
};

// ---- Surekli akan, PENCERE KAVRAMINI BILMEYEN optik akis izleyici ----
// ORBExtractor+FeatureMatcher'in yerini alir: her karede AYNI noktayi id'siyle takip eder (yeniden
// tespit edip eslestirmek yerine), bu yuzden bir izin N kare boyunca hayatta kalmasi "N bagimsiz
// eslesmenin hepsinin tesadufen hizalanmasi" degil, "ayni id'nin LK+RANSAC'tan art arda gecmesi"
// sorusu haline gelir. Pencere sinirini asip asmadigini (main.cpp'nin sorumlulugu) HIC bilmez -
// izlerin pencere sinirlarini "bedavaya" asmasini saglayan tasarim budur.
class KltTracker {
public:
    explicit KltTracker(const KltParams& p = KltParams()) : params(p) {}

    // Her karede bir kez cagrilir - extractor.detect()+matcher.match() bloğunun yerini alir.
    void track(const cv::Mat& grayUndistorted) {
        if (prevGray.empty()) {
            seedFresh(grayUndistorted);
            prevGray = grayUndistorted;
            return;
        }

        if (!activePoints.empty()) {
            std::vector<cv::Point2f> nextPoints;
            std::vector<uchar> status;
            std::vector<float> err;
            cv::calcOpticalFlowPyrLK(prevGray, grayUndistorted, activePoints, nextPoints,
                                      status, err, params.winSize, params.maxLevel, params.criteria);

            // ---- Katman 1: LK status + goruntu siniri kontrolu ----
            std::vector<cv::Point2f> prevKept, currKept;
            std::vector<int64_t> keptIds;
            prevKept.reserve(activePoints.size());
            currKept.reserve(activePoints.size());
            keptIds.reserve(activeIds.size());
            for (size_t i = 0; i < nextPoints.size(); ++i) {
                if (!status[i]) continue;
                const cv::Point2f& p = nextPoints[i];
                if (p.x < 0 || p.y < 0 || p.x >= grayUndistorted.cols || p.y >= grayUndistorted.rows) continue;
                prevKept.push_back(activePoints[i]);
                currKept.push_back(p);
                keptIds.push_back(activeIds[i]);
            }

            // ---- Katman 2: RANSAC temel matris tutarlilik kontrolu (tekli-siçrama) ----
            // LK sessizce kayabilir (tekrarli doku, hizli hareket) - status==1 olsa bile hatali
            // olabilir. Epipolar geometriyle tutarsiz olanlar burada elenir.
            if ((int)currKept.size() >= params.minPointsForRansac) {
                std::vector<uchar> inlierMask;
                cv::findFundamentalMat(prevKept, currKept, cv::FM_RANSAC,
                                        params.ransacReprojThreshold, params.ransacConfidence, inlierMask);
                std::vector<cv::Point2f> ransacKept;
                std::vector<int64_t> ransacIds;
                ransacKept.reserve(currKept.size());
                ransacIds.reserve(keptIds.size());
                for (size_t i = 0; i < currKept.size(); ++i) {
                    if (inlierMask[i]) { ransacKept.push_back(currKept[i]); ransacIds.push_back(keptIds[i]); }
                }
                activePoints = std::move(ransacKept);
                activeIds = std::move(ransacIds);
            } else {
                // 8-point RANSAC icin yetersiz nokta - filtrelenmemis (kucuk) havuzu kabul et
                activePoints = std::move(currKept);
                activeIds = std::move(keptIds);
            }
        }

        if ((int)activePoints.size() < params.replenishTriggerCount) {
            replenish(grayUndistorted);
        }

        prevGray = grayUndistorted;
    }

    // Su anki tum aktif izlerin id -> piksel konumu haritasi (main.cpp bunu her karede biriktirir).
    std::unordered_map<int64_t, cv::Point2f> idPointMap() const {
        std::unordered_map<int64_t, cv::Point2f> m;
        m.reserve(activePoints.size());
        for (size_t i = 0; i < activePoints.size(); ++i) m[activeIds[i]] = activePoints[i];
        return m;
    }

private:
    void seedFresh(const cv::Mat& gray) {
        std::vector<cv::Point2f> newPts;
        cv::goodFeaturesToTrack(gray, newPts, params.maxCorners, params.qualityLevel, params.minDistance);
        activePoints = newPts;
        activeIds.resize(newPts.size());
        for (auto& id : activeIds) id = nextId++;
    }

    void replenish(const cv::Mat& gray) {
        int need = params.maxCorners - (int)activePoints.size();
        if (need <= 0) return;

        // Mevcut noktalarin etrafini maskeleyerek yeni noktalarin onlarla cakismasini engelle
        // (goodFeaturesToTrack'in kendi minDistance'i sadece yeni-yeni nokta araligini kontrol eder).
        cv::Mat mask(gray.size(), CV_8UC1, cv::Scalar(255));
        for (const auto& p : activePoints) {
            cv::circle(mask, p, (int)params.replenishMaskRadius, cv::Scalar(0), cv::FILLED);
        }

        std::vector<cv::Point2f> newPts;
        cv::goodFeaturesToTrack(gray, newPts, need, params.qualityLevel, params.minDistance, mask);
        for (auto& p : newPts) {
            activePoints.push_back(p);
            activeIds.push_back(nextId++);
        }
    }

    cv::Mat prevGray;
    std::vector<cv::Point2f> activePoints;
    std::vector<int64_t> activeIds;
    int64_t nextId = 0;
    KltParams params;
};

// ---- idPointHistory'den pencere-tam-kapsamali izleri kurar ----
// idPointHistory[frameIdx] = o karede hayatta olan id -> piksel konumu haritasi.
// idPointHistory[0]'daki HER id icin, pencerenin TUMUNDE (fi=1..windowSize-1) var mi diye bakar
// (kesisim - eski buildFullCoverageTracks'teki zincir-hopping YERINE). Var olanlar icin senkron
// cv::KeyPoint(pt,1.0f) listesi (kpList) + ozdeslik-eslemeli tracks (tracks[ti]={ti,ti,...,ti})
// uretir - VIOBundleAdjustment::optimize'in beklentisiyle BIREBIR uyumlu (o fonksiyon sadece .pt
// okur, gercek ORB KeyPoint'i olup olmadigina hic bakmaz).
inline void buildWindowTracksFromKltHistory(
        const std::vector<std::unordered_map<int64_t, cv::Point2f>>& idPointHistory,
        std::vector<std::vector<cv::KeyPoint>>& kpList,
        std::vector<std::vector<int>>& tracks) {

    int windowSize = (int)idPointHistory.size();
    kpList.assign(windowSize, {});
    tracks.clear();
    if (windowSize == 0 || idPointHistory[0].empty()) return;

    for (const auto& kv : idPointHistory[0]) {
        int64_t id = kv.first;
        bool full = true;
        for (int fi = 1; fi < windowSize && full; ++fi) {
            full = idPointHistory[fi].count(id) != 0;
        }
        if (!full) continue;

        int ti = (int)kpList[0].size();
        for (int fi = 0; fi < windowSize; ++fi) {
            kpList[fi].emplace_back(idPointHistory[fi].at(id), 1.0f);
        }
        tracks.push_back(std::vector<int>(windowSize, ti));
    }
}
