#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <vector>
#include <array>
#include <optional>
#include "pip02_Matcher.hpp"
#include "pip04_VioWindow.hpp"
#include "Imufactor.hpp"

// ---- Loop closure: keyframe veritabani + yer tanima (descriptor eslestirme) + geometrik dogrulama
// (PnP, metrik olcekli) + pose graph optimizasyonu ----
// Tasarim: BoW/vocabulary YOK (brute-force descriptor eslestirme, mevcut FeatureMatcher altyapisi
// kullanilir). Dogrulama essential-matrix DEGIL, PnP - cunku BA'dan cikan landmark'lar zaten IMU
// sayesinde metrik olcekte, PnP dogrudan olcekli bir mutlak poz verir (essential-matrix sadece yon
// verir, monokuler olcek belirsizligi tasir). Duzeltme "offline" uygulanir - canli VIO state'ine
// (anchor/currentPrior) geri beslenmez, sadece RAPORLANAN/olculen yorungeyi duzeltir.

struct Keyframe {
    int frameIndex;                       // global kare indeksi (zaman/mesafe gate'i icin)
    std::array<double,6> pose;            // world(kare-0)->camera, ReprojectionError/ImuFactor ile AYNI kural
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<cv::Point3d> landmarks;   // world(kare-0) cercevesinde 3B noktalar (BA'dan, YENIDEN HESAPLANMAZ)
    std::vector<int> landmarkKptIdx;      // landmarks[k] <-> keypoints[landmarkKptIdx[k]]
};

class KeyframeDatabase {
public:
    std::vector<Keyframe> keyframes;
    double minKeyframeDist;  // metre - son keyframe'den bu kadar hareket edilmeden yeni keyframe eklenmez
    int maxFrameGapFallback; // BULUNAN GERCEK SORUN (deneyle): sadece mesafeye guvenmek, VIO'nun kendi
                              // tahmininin uzun sure ("durgun bolge") neredeyse hareketsiz gorundugu
                              // donemlerde DEV bir goruntu-kapsama BOSLUGU birakiyordu - gercek dongu
                              // noktasi boyle bir boslugun icine dusup HIC keyframe olarak
                              // yakalanamiyordu. main.cpp'nin KENDI pencere-kapatma mantigindaki
                              // "maxWindowFrames guvenlik tavani" ile AYNI ilkeyle: mesafe esigi
                              // gecilmese BILE bu kadar kare gectiyse keyframe yine de eklenir.
    explicit KeyframeDatabase(double minDist = 1.0, int maxGap = 50)
        : minKeyframeDist(minDist), maxFrameGapFallback(maxGap) {}

    // ---- Yeni bir keyframe adayi ekler (mesafe-tabanli seyreltme + kare-sayisi guvenlik tavaniyla) ----
    bool addKeyframe(Keyframe kf) {
        if (!keyframes.empty()) {
            cv::Point3d prevC = cameraCenterFromPose(keyframes.back().pose);
            cv::Point3d newC = cameraCenterFromPose(kf.pose);
            double d = cv::norm(newC - prevC);
            int frameGap = kf.frameIndex - keyframes.back().frameIndex;
            if (d < minKeyframeDist && frameGap < maxFrameGapFallback) return false;
        }
        keyframes.push_back(std::move(kf));
        return true;
    }
};

// ---- "Gercekten gidip geldi mi, yoksa kamera hic mi hareket etmedi?" ayrimini VIO pozundan (dolayisiyla
// surukleme sapmasindan) TAMAMEN BAGIMSIZ, SAF GORSEL bir sinyalle yapar. ----
// ONCEKI DENEME (bu fonksiyonun ilk hali) bunun icin "VIO pozlarindan kumulatif kat edilen mesafe"
// kullanmisti - DENEYLE BASARISIZ OLDU: bu oturumdaki bilinen sistematik surukleme sorunu yuzunden
// VIO'nun kendi pozu da "durgun bolge"de gercek hareketi COK KUCUK gosteriyor (bu dataset'te kare
// 4->900 arasi VIO'ya gore sadece ~0.3m, gercekte cok daha fazla) - yani hatali sinyale hatali bir
// baska filtre eklenmis oluyordu. Bunun yerine: gercek bir donusun ORTASINDAKI keyframe, HER IKI uctan
// da (aday VE yeni) GORSEL OLARAK BELIRGIN SEKILDE FARKLI gorunmelidir (kamera gercekten baska bir
// yere gitmisse). "Durgun bolge" yanlis pozitiflerinde ise ORTA kare de uclarla neredeyse ayni gorunur
// (kamera hic hareket etmedigi icin) - bu, kare 4<->395 vakasinin (ATE'yi kotulestirmisti) ayirt
// edici ozelligiydi ve saf descriptor eslestirmesiyle DOGRUDAN test edilebilir.
inline bool intermediateKeyframeDiffers(
        const KeyframeDatabase& db, int candIdx, int newIdx, FeatureMatcher& matcher,
        int maxSimilarToMiddle = 25) {
    if (newIdx - candIdx < 2) return true; // aralarinda ayri bir orta kare YOK (bitisik keyframe'ler) - gecerli say
    int midIdx = (candIdx + newIdx) / 2;
    if (midIdx == candIdx || midIdx == newIdx) return true;
    const Keyframe& mid = db.keyframes[midIdx];
    if (mid.descriptors.empty()) return true;

    const Keyframe& cand = db.keyframes[candIdx];
    const Keyframe& newKf = db.keyframes[newIdx];
    if (cand.descriptors.empty() || newKf.descriptors.empty()) return true;

    int mCand = (int)matcher.match(cand.descriptors, mid.descriptors).size();
    int mNew  = (int)matcher.match(mid.descriptors, newKf.descriptors).size();
    // Gercek bir donus: ORTA kare HER IKI uctan da FARKLI olmali (dusuk eslesme). Herhangi biri
    // yuksek eslesme veriyorsa (orta kare bir uca cok benziyor), muhtemelen kamera o araligin
    // tamaminda ayni yerde durmus demektir - loop degil.
    return mCand < maxSimilarToMiddle && mNew < maxSimilarToMiddle;
}

// ---- Yeni eklenen (son) keyframe'i, zaman/indeks olarak yeterince ESKI tum keyframe'lere karsi
// descriptor eslestirmesiyle tarar. Eslesme sayisi esigini gecenler VE intermediateKeyframeDiffers
// testini gecenler aday olarak donulur (henuz GEOMETRIK dogrulama yapilmadi - bu adim sadece UCUZ
// bir on-filtre). ----
inline std::vector<int> findLoopCandidates(
        const KeyframeDatabase& db, FeatureMatcher& matcher,
        int minFrameGap = 100, int minMatchCount = 30) {

    std::vector<int> candidates;
    if (db.keyframes.empty()) return candidates;
    int newIdx = (int)db.keyframes.size() - 1;
    const Keyframe& newKf = db.keyframes[newIdx];

    for (int i = 0; i < newIdx; ++i) {
        const Keyframe& old = db.keyframes[i];
        if (newKf.frameIndex - old.frameIndex < minFrameGap) continue; // cok yakin zamanli - loop degil, normal takip
        if (old.descriptors.empty() || newKf.descriptors.empty()) continue;

        std::vector<cv::DMatch> m = matcher.match(old.descriptors, newKf.descriptors);
        if ((int)m.size() < minMatchCount) continue;
        if (!intermediateKeyframeDiffers(db, i, newIdx, matcher)) continue; // durgun-bolge yanlis pozitif filtresi
        candidates.push_back(i);
    }
    return candidates;
}

struct LoopVerificationResult {
    bool valid = false;
    int candidateIdx = -1;   // eski (dongu kapanan) keyframe'in db'deki indeksi
    int loopKeyframeIdx = -1; // YENI (dongu tespit edildigindeki "guncel") keyframe'in db'deki indeksi -
                               // solvePoseGraph'in DOGRU dugume prior uygulayabilmesi icin GEREKLI
                               // (dongu bulunduktan SONRA da yeni keyframe'ler eklenebilir - db'nin
                               // SON elemani ile KARISTIRILMAMALI).
    std::array<double,6> pose; // PnP'den cikan, newKf'nin world(kare-0) cercevesindeki MUTLAK pozu
    int inlierCount = 0;
};

// ---- Aday keyframe'in 3B landmark'lari (world/kare-0 cercevesinde) + yeni keyframe'in 2B keypoint'leri
// arasinda PnP (solvePnPRansac) calistirir. Essential-matrix YERINE PnP kullanilmasinin nedeni: PnP
// dogrudan METRIK olcekli mutlak poz verir (bkz. dosya basi yorumu), monokuler essential-matrix'in
// olcek belirsizligi burada YOK. ----
inline LoopVerificationResult verifyLoopPnP(
        const KeyframeDatabase& db, int candidateIdx, const Keyframe& newKf, int newKfDbIdx,
        FeatureMatcher& matcher, const cv::Mat& K,
        int minInliers = 20) {

    LoopVerificationResult result;
    const Keyframe& cand = db.keyframes[candidateIdx];
    if (cand.descriptors.empty() || newKf.descriptors.empty()) return result;

    std::vector<cv::DMatch> matches = matcher.match(cand.descriptors, newKf.descriptors);
    if (matches.empty()) return result;

    // ---- cand keypoint indeksinden landmark'a hizli erisim haritasi ----
    std::vector<int> kptToLandmark(cand.keypoints.size(), -1);
    for (size_t k = 0; k < cand.landmarkKptIdx.size(); ++k) {
        int kptIdx = cand.landmarkKptIdx[k];
        if (kptIdx >= 0 && kptIdx < (int)kptToLandmark.size()) kptToLandmark[kptIdx] = (int)k;
    }

    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;
    for (const auto& m : matches) {
        int landmarkIdx = kptToLandmark[m.queryIdx]; // queryIdx: cand (bkz. matcher.match(descs1=cand, descs2=newKf))
        if (landmarkIdx < 0) continue; // bu cand keypoint'inin bilinen bir 3B karsiligi yok
        const cv::Point3d& p3 = cand.landmarks[landmarkIdx];
        objectPoints.emplace_back((float)p3.x, (float)p3.y, (float)p3.z);
        imagePoints.push_back(newKf.keypoints[m.trainIdx].pt);
    }
    if ((int)objectPoints.size() < minInliers) return result; // esik-alti aday veri - dogrulamaya bile girme

    cv::Mat rvec, tvec, inliers;
    // Goruntuler zaten distorsiyon-duzeltilmis (bkz. main.cpp buildDistortionCoeffs/remap) - distCoeffs=0.
    bool ok = cv::solvePnPRansac(objectPoints, imagePoints, K, cv::Mat(),
                                   rvec, tvec, false, 200, 4.0, 0.999, inliers);
    if (!ok || inliers.rows < minInliers) return result;

    result.valid = true;
    result.candidateIdx = candidateIdx;
    result.loopKeyframeIdx = newKfDbIdx;
    result.pose = { rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2),
                     tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2) };
    result.inlierCount = inliers.rows;
    return result;
}

// ================= Pose graph =================

// ---- angleAxis'ten GERCEK rotasyon matrisini row-major duzende, BELIRSIZLIK OLMADAN kurar ----
// Imufactor.hpp'deki "AngleAxisToRotationMatrix + mat3Transpose hilesi" (column-major/row-major
// karisikligina dayanan, sadece o dosyanin KENDI residual'lari icin dogrulanmis) TEKRAR EDILMEDI -
// bunun yerine Ceres'in kendi AngleAxisRotatePoint'i (belirsizliksiz: R*v hesaplar) ile R'nin 3
// sutununu (standart baz vektorlerini dondurerek) ayri ayri kurup row-major diziye yerlestiriyoruz.
// Daha ayrintili ama YANLIS row/column-major varsayimi riski YOK - yeni (test edilmemis) matematik
// oldugu icin en denetlenebilir yol tercih edildi.
template <typename T>
inline void trueRotationMatrixRowMajor(const T* angleAxis, T Rout[9]) {
    const T e0[3] = { T(1), T(0), T(0) };
    const T e1[3] = { T(0), T(1), T(0) };
    const T e2[3] = { T(0), T(0), T(1) };
    T c0[3], c1[3], c2[3];
    ceres::AngleAxisRotatePoint(angleAxis, e0, c0);
    ceres::AngleAxisRotatePoint(angleAxis, e1, c1);
    ceres::AngleAxisRotatePoint(angleAxis, e2, c2);
    for (int r = 0; r < 3; ++r) {
        Rout[r*3+0] = c0[r];
        Rout[r*3+1] = c1[r];
        Rout[r*3+2] = c2[r];
    }
}

// ---- Iki ardisik keyframe pozu arasindaki OLCULEN goreli donusumu korumaya calisan residual ----
// (pose graph'in "odometri kenari" - VIO'nun zaten bulmus oldugu YEREL sekli buyuk olcude korur,
// sadece loop kisiti onu global olarak yumusakca "cekistirdiginde" esneme paylarini saglar).
// R_ij = R_j * R_i^T, t_ij = t_j - R_ij*t_i (iki world->camera donusumunun bilesimi - bkz. plan notu).
struct RelativePoseError {
    cv::Matx33d R_ij_meas;
    cv::Vec3d t_ij_meas;
    double rotWeight, transWeight;

    RelativePoseError(const cv::Matx33d& R, const cv::Vec3d& t, double rotW, double transW)
        : R_ij_meas(R), t_ij_meas(t), rotWeight(rotW), transWeight(transW) {}

    template <typename T>
    bool operator()(const T* pose_i, const T* pose_j, T* residual) const {
        T Ri[9], Rj[9];
        trueRotationMatrixRowMajor(pose_i, Ri);
        trueRotationMatrixRowMajor(pose_j, Rj);

        T RiT[9]; mat3Transpose(Ri, RiT);
        T Rij[9]; mat3Mul(Rj, RiT, Rij);

        T ti[3] = { pose_i[3], pose_i[4], pose_i[5] };
        T tj[3] = { pose_j[3], pose_j[4], pose_j[5] };
        T Rij_ti[3]; mat3VecMul(Rij, ti, Rij_ti);
        T tij[3] = { tj[0]-Rij_ti[0], tj[1]-Rij_ti[1], tj[2]-Rij_ti[2] };

        T Rmeas[9]; mat3FromDouble(R_ij_meas, Rmeas);
        T RmeasT[9]; mat3Transpose(Rmeas, RmeasT);
        T Rerr[9]; mat3Mul(RmeasT, Rij, Rerr);      // row-major: Log(R_ij_meas^T * R_ij)
        T RerrColMajor[9]; mat3Transpose(Rerr, RerrColMajor); // ceres::RotationMatrixToAngleAxis COLUMN-MAJOR bekler
        T rotErr[3];
        ceres::RotationMatrixToAngleAxis(RerrColMajor, rotErr);

        T tMeas[3] = { T(t_ij_meas[0]), T(t_ij_meas[1]), T(t_ij_meas[2]) };
        T transErr[3] = { tij[0]-tMeas[0], tij[1]-tMeas[1], tij[2]-tMeas[2] };

        for (int k = 0; k < 3; ++k) residual[k]   = T(rotWeight) * rotErr[k];
        for (int k = 0; k < 3; ++k) residual[3+k] = T(transWeight) * transErr[k];
        return true;
    }

    static ceres::CostFunction* create(const cv::Matx33d& R, const cv::Vec3d& t, double rotW, double transW) {
        return new ceres::AutoDiffCostFunction<RelativePoseError, 6, 6, 6>(
            new RelativePoseError(R, t, rotW, transW));
    }
};

// ---- pose_i, pose_j'den (double) OLCULEN goreli donusumu hesaplar - odometri kenarlari kurulurken
// (pose graph optimizasyonundan ONCE, mevcut VIO pozlariyla) BIR KEZ cagrilir. ----
inline void computeRelativePoseDouble(const std::array<double,6>& pose_i, const std::array<double,6>& pose_j,
                                        cv::Matx33d& R_ij, cv::Vec3d& t_ij) {
    cv::Mat rvecI = (cv::Mat_<double>(3,1) << pose_i[0], pose_i[1], pose_i[2]);
    cv::Mat rvecJ = (cv::Mat_<double>(3,1) << pose_j[0], pose_j[1], pose_j[2]);
    cv::Mat Ri, Rj;
    cv::Rodrigues(rvecI, Ri);
    cv::Rodrigues(rvecJ, Rj);
    cv::Mat RiT = Ri.t();
    cv::Mat Rij = Rj * RiT;
    cv::Vec3d ti(pose_i[3], pose_i[4], pose_i[5]);
    cv::Vec3d tj(pose_j[3], pose_j[4], pose_j[5]);
    cv::Mat tiMat = (cv::Mat_<double>(3,1) << ti[0], ti[1], ti[2]);
    cv::Mat Rij_ti = Rij * tiMat;
    R_ij = cv::Matx33d(Rij);
    t_ij = cv::Vec3d(tj[0]-Rij_ti.at<double>(0), tj[1]-Rij_ti.at<double>(1), tj[2]-Rij_ti.at<double>(2));
}

// ---- Tum keyframe pozlarini pose graph olarak kurup cozer: ardisik kenarlar icin RelativePoseError,
// dogrulanmis loop icin PosePriorError (Imufactor.hpp'den - YENI residual GEREKMEDI, PnP zaten mutlak
// bir poz olcumu, PosePriorError'in "bilinen kovaryansli mutlak poz prior'u" kalibiyla TAM ORTUSUYOR).
// kare-0 (ilk keyframe) sabitlenir - VIOBundleAdjustment::optimize()'daki AYNI kural. ----
inline std::vector<std::array<double,6>> solvePoseGraph(
        const KeyframeDatabase& db, const LoopVerificationResult& loop,
        double odomRotWeight, double odomTransWeight, double loopWeight) {

    std::vector<std::array<double,6>> poses;
    for (const auto& kf : db.keyframes) poses.push_back(kf.pose);
    if (poses.size() < 2 || !loop.valid) return poses;

    ceres::Problem problem;
    for (size_t i = 0; i + 1 < poses.size(); ++i) {
        cv::Matx33d R_meas; cv::Vec3d t_meas;
        computeRelativePoseDouble(poses[i], poses[i+1], R_meas, t_meas);
        problem.AddResidualBlock(RelativePoseError::create(R_meas, t_meas, odomRotWeight, odomTransWeight),
                                  nullptr, poses[i].data(), poses[i+1].data());
    }

    Eigen::Matrix<double,6,6,Eigen::RowMajor> loopSqrtInfo = Eigen::Matrix<double,6,6,Eigen::RowMajor>::Identity() * loopWeight;
    Eigen::Matrix<double,6,1> loopX0;
    for (int k = 0; k < 6; ++k) loopX0[k] = loop.pose[k];
    // NOT: loop.loopKeyframeIdx kullanilir (db'nin SON elemani DEGIL - dongu tespit edildikten SONRA
    // da yeni keyframe'ler eklenmis olabilir, bkz. LoopVerificationResult yorumu).
    problem.AddResidualBlock(PosePriorError::create(loopSqrtInfo, loopX0), nullptr, poses[loop.loopKeyframeIdx].data());

    if (problem.HasParameterBlock(poses[0].data())) problem.SetParameterBlockConstant(poses[0].data());

    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (!summary.IsSolutionUsable()) {
        // guvenlik: cozum kullanilamazsa duzeltmesiz (orijinal VIO) pozlara don
        std::vector<std::array<double,6>> fallback;
        for (const auto& kf : db.keyframes) fallback.push_back(kf.pose);
        return fallback;
    }

    return poses;
}
