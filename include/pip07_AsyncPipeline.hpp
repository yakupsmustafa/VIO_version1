#pragma once
#include <mutex>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <vector>
#include <array>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "Viobundleadjustment.hpp"
#include "pip04_VioWindow.hpp"
#include "pip06_LoopClosure.hpp"
#include "pip02_Matcher.hpp"

// ---- Front-end/back-end ayrisimi icin paylasilan yapilar ----
// Amac: BA+kovaryans (pencere basina 100-1000ms, bkz. main.cpp'deki "BA+kov" olcumleri) ana kare
// dongusunu BLOKE ETMESIN. Front-end (kare-basi: oku+distorsiyon-duzelt+ORB+eslestir+IMU-seed+canli
// gorsellestirme) kesintisiz devam eder; BA, AYRI bir thread'de (back-end) arka planda calisir.
// NOT (bilincli tercih, kullaniciyla konusuldu): bu, sistemi artik GERCEKTEN zamanlamaya bagimli
// yapar - bugune kadarki determinizm caplarindan FARKLI olarak, artik OpenCV/Ceres ic thread'leri
// DEGIL, gercek wall-clock yarisi soz konusu. Gercek VIO sistemleri de boyledir, kabul edilen bedel.

// ---- Bir pencerenin BA'ya gonderilecek TUM girdisi - front-end'de paketlenir, back-end'de tuketilir ----
struct WindowJob {
    int windowIndex = 0;
    int framesUsed = 0;
    int lastFrameGlobalIndex = 0; // pencerenin SON karesinin global kare indeksi (keyframe.frameIndex icin)
    double parallaxPx = 0.0;
    bool lowConfidenceForceClose = false;
    double velPriorWeight = 0.1;

    std::vector<std::vector<cv::KeyPoint>> kpList;
    std::vector<std::vector<int>> tracks;
    std::vector<std::array<double,3>> landmarks;
    std::vector<int64_t> frameTimestamps;
    std::vector<VIOBundleAdjustment::FrameState> windowStates; // IMU-seed (BA baslangic degeri + basarisizlik fallback'i)

    cv::Mat lastFrameDescriptors; // keyframe olusturmak icin (pip06_LoopClosure.hpp)

    bool incomingPriorValid = false;
    VIOBundleAdjustment::WindowPrior incomingPrior;
};

// ---- Front-end'in okudugu, back-end'in yazdigi paylasilan capa durumu (mutex korumali) ----
class SharedAnchorState {
public:
    explicit SharedAnchorState(const VIOBundleAdjustment::FrameState& initAnchor) : anchor_(initAnchor) {
        currentPrior_.valid = false;
    }

    void get(VIOBundleAdjustment::FrameState& outAnchor, VIOBundleAdjustment::WindowPrior& outPrior) const {
        std::lock_guard<std::mutex> lock(mtx_);
        outAnchor = anchor_;
        outPrior = currentPrior_;
    }

    void set(const VIOBundleAdjustment::FrameState& newAnchor, const VIOBundleAdjustment::WindowPrior& newPrior) {
        std::lock_guard<std::mutex> lock(mtx_);
        anchor_ = newAnchor;
        currentPrior_ = newPrior;
    }

private:
    mutable std::mutex mtx_;
    VIOBundleAdjustment::FrameState anchor_;
    VIOBundleAdjustment::WindowPrior currentPrior_;
};

// ---- Sinirli boyutlu, thread-safe pencere-isi kuyrugu ----
// Kuyruk DOLUYSA front-end BLOKLANMAZ - tryPush false doner, cagiran (front-end) pencereyi ATAR
// (IMU-seed korunur, sonraki pencerelerle devam edilir). Bu, gercek-zamanli sistemlerin standart
// davranisidir: back-end geride kalsa bile front-end hizi hic dusmez.
class WindowJobQueue {
public:
    explicit WindowJobQueue(size_t maxSize) : maxSize_(maxSize) {}

    bool tryPush(WindowJob job) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (jobs_.size() >= maxSize_) return false;
            jobs_.push_back(std::move(job));
        }
        cv_.notify_one();
        return true;
    }

    // Kuyruk bosaldiginda VE stopRequested=true ise false doner (back-end'in cikis sinyali).
    bool pop(WindowJob& out, std::atomic<bool>& stopRequested) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [&] { return !jobs_.empty() || stopRequested.load(); });
        if (jobs_.empty()) return false;
        out = std::move(jobs_.front());
        jobs_.pop_front();
        return true;
    }

    void notifyStop() { cv_.notify_all(); }

private:
    mutable std::mutex mtx_;
    std::condition_variable cv_;
    std::deque<WindowJob> jobs_;
    size_t maxSize_;
};

// ---- Kucuk yardimci: imuBufferMutex NULL olabilir (EuRoC yolu, main.cpp) - o durumda no-op ----
// AirSim canli yolunda (main_airsim.cpp) DOLU gecirilir: vio.optimize() ICINDE cagrilan
// imuPreint.preintegrate() ham IMU tamponunu (timestamps/values) DOGRUDAN okur (bkz.
// Viobundleadjustment.hpp) - EuRoC'ta bu tampon dosyadan bir kez yuklenip BIR DAHA
// DEGISMEDIGI icin coklu-okuyucu güvenlidir, ama canli yolda front-end AYNI tampona
// surekli addSample ile YAZARKEN back-end bu kilit OLMADAN okursa veri yarisi olusur.
struct OptionalLockGuard {
    std::mutex* m;
    explicit OptionalLockGuard(std::mutex* m_) : m(m_) { if (m) m->lock(); }
    ~OptionalLockGuard() { if (m) m->unlock(); }
};

// ---- BACK-END: WindowJob kuyruktan cekilir, BA+kovaryans (pencere basina 100-1000ms) burada
// calisir - front-end'i HIC BLOKLAMAZ (bkz. dosya basi yorumu). Basarili/basarisiz sonuc
// SharedAnchorState uzerinden front-end'e (mutex korumali) yayinlanir. Keyframe/loop-closure
// mantigi da BURADA (optimize edilmis poz/landmark'lara bagimli oldugu icin buraya ait).
// Veri KAYNAGINI (dosya mi, AirSim mi) bilmiyor - main.cpp (EuRoC) VE main_airsim.cpp (canli)
// tarafindan AYNEN paylasilir (bkz. main_airsim.cpp plani, Tasarim Karari 2). imuBufferMutex:
// EuRoC'ta nullptr (dosyadan bir kez yuklenen IMU tamponu bir daha degismez, kilide GEREK YOK);
// AirSim canli yolunda DOLU gecirilir (front-end AYNI tampona esZAMANLI yaziyor, bkz. main_airsim.cpp). ----
inline void backEndLoop(VIOBundleAdjustment& vio, WindowJobQueue& jobQueue, SharedAnchorState& sharedAnchor,
                         std::atomic<bool>& stopRequested, const cv::Mat& K, FeatureMatcher& matcherBackend,
                         KeyframeDatabase& kfDb, std::vector<LoopVerificationResult>& foundLoops,
                         int& windowFailCount, int& priorSuccessCount,
                         std::mutex* imuBufferMutex = nullptr) {
    WindowJob job;
    while (jobQueue.pop(job, stopRequested)) {
        bool ok = false;
        VIOBundleAdjustment::WindowPrior outPrior;
        std::vector<VIOBundleAdjustment::FrameState> optStates = job.windowStates;

        int64 tBaStart = cv::getTickCount();
        {
            // ---- imuPreint.preintegrate() (optimize() icinde) ham IMU tamponunu OKUR - AirSim canli
            // yolunda bu tampon front-end tarafindan ESZAMANLI buyutuluyor. EuRoC yolunda
            // imuBufferMutex=nullptr, kilit NO-OP. ----
            OptionalLockGuard lg(imuBufferMutex);
            ok = vio.optimize(optStates, job.frameTimestamps, job.kpList, job.tracks, job.landmarks,
                               job.incomingPriorValid ? &job.incomingPrior : nullptr, &outPrior, job.velPriorWeight);
        }
        double baMs = (cv::getTickCount() - tBaStart) / cv::getTickFrequency() * 1000.0;
        std::cout << "Pencere " << job.windowIndex << " (" << job.framesUsed << " kare, paralaks="
                  << job.parallaxPx << "px" << (job.lowConfidenceForceClose ? ", MAKS-KARE limiti, siki-hiz-prior" : "")
                  << "): " << job.tracks.size() << " tam kapsama izi, (BA+kov: " << baMs << " ms) ";

        if (ok) {
            // isfinite tek basina yetmiyor: teshis edilen vaka BA'nin NaN/Inf URETMEDEN, ama fiziksel
            // olarak SACMA (|v|=26 m/s, sonra 115 m/s) bir cozume "basarili" olarak yakinsayabildigini
            // gosterdi. Gevsek ama makul bir ust sinir: bu dataset'te (EuRoC MH01) gozlenen gercek
            // hizlar hep <5 m/s, 15 m/s bol pay birakir.
            const double maxPlausibleVel = 15.0;
            for (const auto& st : optStates) {
                for (double v : st.pose) if (!std::isfinite(v)) ok = false;
                for (double v : st.vel)  if (!std::isfinite(v)) ok = false;
                for (double v : st.ba)   if (!std::isfinite(v)) ok = false;
                for (double v : st.bg)   if (!std::isfinite(v)) ok = false;
                double velNorm = std::sqrt(st.vel[0]*st.vel[0] + st.vel[1]*st.vel[1] + st.vel[2]*st.vel[2]);
                if (velNorm > maxPlausibleVel) ok = false;
            }
        }

        VIOBundleAdjustment::FrameState newAnchor;
        VIOBundleAdjustment::WindowPrior newPrior;

        if (ok) {
            newAnchor = optStates.back();
            newPrior = outPrior; // outPrior.valid=false ise sonraki pencere sert-sabite doner
            if (outPrior.valid) priorSuccessCount++;
            std::cout << "BA basarili=true\n";

            // ---- Loop closure: keyframe ekle + yer tanima + PnP dogrulama (bkz. pip06_LoopClosure.hpp) ----
            Keyframe kf;
            kf.frameIndex = job.lastFrameGlobalIndex;
            kf.pose = optStates.back().pose;
            kf.keypoints = job.kpList.back();
            kf.descriptors = job.lastFrameDescriptors;
            int lastFrameIdx = (int)job.kpList.size() - 1;
            for (size_t ti = 0; ti < job.tracks.size(); ++ti) {
                kf.landmarks.emplace_back(job.landmarks[ti][0], job.landmarks[ti][1], job.landmarks[ti][2]);
                kf.landmarkKptIdx.push_back(job.tracks[ti][lastFrameIdx]);
            }
            if (kfDb.addKeyframe(kf)) {
                int newKfDbIdx = (int)kfDb.keyframes.size() - 1;
                std::vector<int> candidates = findLoopCandidates(kfDb, matcherBackend);
                for (int cIdx : candidates) {
                    LoopVerificationResult lv = verifyLoopPnP(kfDb, cIdx, kfDb.keyframes.back(), newKfDbIdx, matcherBackend, K);
                    if (lv.valid) {
                        std::cout << "  [LOOP] Dogrulanan dongu: keyframe " << cIdx << " (kare "
                                  << kfDb.keyframes[cIdx].frameIndex << ") <-> yeni keyframe (kare "
                                  << kf.frameIndex << "), inlier=" << lv.inlierCount << "\n";
                        foundLoops.push_back(lv);
                    }
                }
            }
        } else {
            newAnchor = job.windowStates.back();
            newPrior.valid = false; // mevcut IMU-seed fallback'iyle BIREBIR ayni mantik
            std::cout << "BA basarili=false (UYARI: BA sonucu kullanilamadi/NaN/mantiksiz hiz veya bos iz seti, IMU seed korunuyor)\n";
            windowFailCount++;
        }

        sharedAnchor.set(newAnchor, newPrior);

        // ---- TESHIS: capa hiz/konum buyuklugu - kacis nerede basliyor gormek icin ----
        {
            double velNorm = std::sqrt(newAnchor.vel[0]*newAnchor.vel[0] + newAnchor.vel[1]*newAnchor.vel[1] + newAnchor.vel[2]*newAnchor.vel[2]);
            cv::Point3d c = cameraCenterFromPose(newAnchor.pose);
            double posNorm = std::sqrt(c.x*c.x + c.y*c.y + c.z*c.z);
            double baNorm = std::sqrt(newAnchor.ba[0]*newAnchor.ba[0] + newAnchor.ba[1]*newAnchor.ba[1] + newAnchor.ba[2]*newAnchor.ba[2]);
            std::cout << "  [TESHIS] capa: |v|=" << velNorm << " m/s, |C|=" << posNorm
                      << " m, |ba|=" << baNorm << " m/s^2, kare=" << job.framesUsed
                      << ", paralaks=" << job.parallaxPx << "px, capa_prior="
                      << (newPrior.valid ? "var" : "yok") << "\n";
        }
    }
}
