#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

// BFMatcher::create'in tum parametreleri + matching davranisini ayarlayan ek secenekler
struct MatcherParams {
    // ---- cv::BFMatcher::create parametreleri ----
    // Descriptor mesafe olcum tipi. ORB/AKAZE(MLDB) gibi binary descriptor'lar icin
    // NORM_HAMMING (WTA_K=2 ise) ya da NORM_HAMMING2 (WTA_K=3/4 ise) kullanilir.
    // SIFT gibi float descriptor'lar icin NORM_L1 ya da NORM_L2 kullanilir.
    int normType = cv::NORM_HAMMING;

    // true: bir eslesme sadece iki yonde de (A->B ve B->A) ayni ciktiginda kabul edilir.
    // NOT: crossCheck=true iken knnMatch/ratio test KULLANILAMAZ (OpenCV kisitlamasi),
    // bu yuzden crossCheck ve useRatioTest ayni anda true olamaz.
    bool crossCheck = false;

    // ---- Eslestirme modu ve ek filtreleme ----
    // Lowe's ratio test: en yakin eslesme, ikinciye gore yeterince belirgin degilse elenir.
    bool useRatioTest = true;
    float ratioThresh = 0.75f;   // dusuk deger = daha siki eleme, daha az ama daha guvenilir eslesme
    int knnK = 2;                // ratio test icin knnMatch'te kac aday tutulacak (genelde 2)

    // radiusMatch modu: knn yerine, mesafesi belirli bir esigin altinda olan TUM
    // adaylari alir. useRatioTest=false ve useRadiusMatch=true ise bu mod kullanilir.
    bool useRadiusMatch = false;
    float maxDistance = 30.0f;   // radiusMatch icin mesafe esigi (NORM_HAMMING olcegine gore)
};

class FeatureMatcher {
public:
    cv::Ptr<cv::BFMatcher> matcher;
    MatcherParams params;

    explicit FeatureMatcher(const MatcherParams& p = MatcherParams()) : params(p) {
        if (params.crossCheck && params.useRatioTest) {
            throw std::invalid_argument(
                "FeatureMatcher: crossCheck ve useRatioTest ayni anda true olamaz");
        }
        matcher = cv::BFMatcher::create(params.normType, params.crossCheck);
    }

    // descs1: onceki kare, descs2: sonraki kare
    std::vector<cv::DMatch> match(const cv::Mat& descs1, const cv::Mat& descs2) {
        std::vector<cv::DMatch> goodMatches;

        if (params.useRatioTest) {
            std::vector<std::vector<cv::DMatch>> knnMatches;
            matcher->knnMatch(descs1, descs2, knnMatches, params.knnK);

            for (auto& m : knnMatches) {
                if (m.size() < 2) continue;
                if (m[0].distance < params.ratioThresh * m[1].distance) {
                    goodMatches.push_back(m[0]);
                }
            }
        } else if (params.useRadiusMatch) {
            std::vector<std::vector<cv::DMatch>> radiusMatches;
            matcher->radiusMatch(descs1, descs2, radiusMatches, params.maxDistance);

            for (auto& m : radiusMatches) {
                if (!m.empty()) goodMatches.push_back(m[0]); // en yakin adayi al
            }
        } else {
            matcher->match(descs1, descs2, goodMatches);
        }

        return goodMatches;
    }
};
