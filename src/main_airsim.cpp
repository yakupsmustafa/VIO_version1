//  VisualOdometry - AirSim canli kopru
// NOT: Bu makinede AirLib YOK - bu dosya burada DERLENEMEZ/CALISTIRILAMAZ (bkz. plan). Sadece
// AIRSIM_ROOT tanimlandiginda (arkadasinin makinesinde) derlenen vio_airsim hedefinin kaynagi.

#include <opencv2/opencv.hpp>
#include <iostream>
#include <optional>
#include <thread>
#include <atomic>
#include "pip01_FeatureExtractor.h"
#include "pip02_Matcher.hpp"
#include "Test.hpp"
#include "Imupreintegration.hpp"
#include "Imufactor.hpp"
#include "Viobundleadjustment.hpp"
#include "pip04_VioWindow.hpp"
#include "pip06_LoopClosure.hpp"
#include "pip07_AsyncPipeline.hpp"
#include "pip08_AirSimCalibration.hpp"
#include "pip09_AirSimBridge.hpp"

// ---- main.cpp'deki buildExtractor()/buildMatcher()/buildViewer() ile AYNI - main.cpp'ye
// DOKUNULMAYACAGI icin (plan Karar 1) kucuk bir kod-tekrari olarak tercih edildi ----
ORBExtractor buildExtractor() {
    ORBParams params;
    params.nfeatures = 3000;
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

TrajectoryViewer buildViewer() {
    ViewerParams vparams;
    vparams.showFrame = true;
    vparams.showTrajectory = true;
    vparams.frameWindowName = "AirSim Kamera";
    vparams.trajWindowName = "Yorunge (VO vs AirSim GT)";
    vparams.waitKeyDelay = 1;
    vparams.canvasSize = 800;
    vparams.trajScale = 3.0;
    vparams.voColor = cv::Scalar(0, 255, 0);
    vparams.gtColor = cv::Scalar(0, 0, 255);
    vparams.pointRadius = 2;
    vparams.lineThickness = 2;
    vparams.autoScaleGT = true;
    vparams.minGtPathLength = 1e-6;
    vparams.autoFit = true;
    vparams.fitMargin = 0.15;
    vparams.lateralBoost = 1.0;
    vparams.voRotationDeg = 0.0;
    vparams.gtAxisSign = cv::Vec3d(1, 1, 1);
    vparams.showMetrics = true;
    vparams.computeLiveATE = true;
    vparams.ateUpdateInterval = 10;
    vparams.minPointsForATE = 3;
    return TrajectoryViewer(vparams);
}

// ---- FRONT-END (canli): dosyadan okumak yerine AirSim'i YOKLAR. main.cpp'deki frontEndLoop ile
// AYNI pencere-kapatma/WindowJob-paketleme mantigini kullanir (bkz. plan Tasarim Karari 3) ----
void liveAirSimFrontEndLoop(AirSimBridge& bridge, const std::string& cameraName,
                             ORBExtractor& extractor, FeatureMatcher& matcher,
                             const cv::Mat& K, const cv::Mat& R_BS,
                             int minWindowFrames, int maxWindowFrames, double parallaxThresholdPx,
                             int minTracksForParallaxTrust,
                             const cv::Vec3d& gravity,
                             TrajectoryViewer& viewer,
                             SharedAnchorState& sharedAnchor, WindowJobQueue& jobQueue,
                             std::atomic<bool>& stopRequested,
                             ImuPreintegrator& imuPreint, TimestampedDataSource& liveImuCam,
                             std::mutex& imuBufferMutex,
                             int& windowTotalCount, int& windowDroppedCount) {

    // ---- Buyuyen, canli IMU tamponu - ZATEN kamera cercevesinde tutulur (her ornek geldiginde
    // R_CB ile dondurulur) - main.cpp'nin dosya-basi tek-seferlik rotateImuToCameraFrame()'inin
    // canli/artimli karsiligi. DIKKAT (plan Karar 4'ten DUZELTME): bu tampon TEK sahipli DEGIL -
    // back-end thread'i vio.optimize() icinden AYNI tampona (imuPreint uzerinden) esZAMANLI OKUYOR
    // (bkz. main.cpp backEndLoop / OptionalLockGuard). Bu yuzden buraya yazma da AYNI mutex ile
    // korunuyor - main()'de kurulup backEndLoop'a da gecirilen imuBufferMutex. Yercekimi de main()'de,
    // bu thread baslamadan ONCE (durgun kalkis-oncesi ornekleriyle) hesaplanip buraya gecirilir -
    // vio.gravityWorld (back-end) ile buradaki gravity AYNI deger olmali. ----
    cv::Matx33d R_CB = cv::Matx33d(R_BS).t();

    auto pollAndAppendImu = [&]() {
        AirSimImuSample s = bridge.pollImu();
        cv::Vec3d aCam = R_CB * cv::Vec3d(s.ax, s.ay, s.az);
        cv::Vec3d gCam = R_CB * cv::Vec3d(s.gx, s.gy, s.gz);
        std::lock_guard<std::mutex> lock(imuBufferMutex);
        liveImuCam.addSample(s.timestampNs, {aCam[0], aCam[1], aCam[2], gCam[0], gCam[1], gCam[2]});
    };

    std::vector<std::vector<cv::KeyPoint>> kpList;
    std::vector<cv::Mat> descList;
    std::vector<int64_t> frameTimestamps;
    std::vector<std::vector<cv::DMatch>> pairwiseMatches;

    int windowCount = 0;
    int64_t lastImageTs = -1;

    while (!stopRequested.load()) {
        pollAndAppendImu();

        AirSimImageSample img = bridge.pollImage(cameraName);
        if (!img.valid || img.timestampNs == lastImageTs) {
            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 'Q') { std::cout << "'q' tusuna basildi, cikiliyor...\n"; break; }
            continue; // yeni kare henuz gelmedi - sadece IMU biriktirmeye devam
        }
        lastImageTs = img.timestampNs;

        std::vector<cv::KeyPoint> kps;
        cv::Mat descs;
        extractor.detect(img.gray, kps, descs);
        std::cout << "Kare (t=" << img.timestampNs << ") -> bulunan kose sayisi: " << kps.size() << "\n";

        kpList.push_back(kps);
        descList.push_back(descs);
        frameTimestamps.push_back(img.timestampNs);

        if (kpList.size() > 1) {
            std::vector<cv::DMatch> m = matcher.match(descList[descList.size() - 2], descList.back());
            std::cout << "  eslesme sayisi: " << m.size() << "\n";
            pairwiseMatches.push_back(m);
        }

        VIOBundleAdjustment::FrameState anchor;
        VIOBundleAdjustment::WindowPrior currentPrior;
        sharedAnchor.get(anchor, currentPrior);

        std::vector<VIOBundleAdjustment::FrameState> windowStates =
            seedWindowStatesFromImu(imuPreint, anchor, frameTimestamps, gravity);

        cv::Point3d camCenter = cameraCenterFromPose(windowStates.back().pose);
        cv::Mat centerMat = (cv::Mat_<double>(3,1) << camCenter.x, camCenter.y, camCenter.z);
        cv::Mat voBody = R_BS * centerMat;
        cv::Point3d voPos(voBody.at<double>(0), voBody.at<double>(1), voBody.at<double>(2));

        cv::Point3d gtPos = bridge.pollGroundTruthPosition();
        viewer.update(img.gray, voPos, std::optional<cv::Point3d>(gtPos));

        if ((int)kpList.size() >= minWindowFrames) {
            std::vector<std::vector<int>> tracks = buildFullCoverageTracks(pairwiseMatches, (int)kpList[0].size());
            double parallaxPx = computeMedianParallaxPx(kpList, tracks);
            bool reachedMax = (int)kpList.size() >= maxWindowFrames;
            bool enoughParallax = ((int)tracks.size() >= minTracksForParallaxTrust) && (parallaxPx >= parallaxThresholdPx);

            if (enoughParallax || reachedMax) {
                int framesUsed = (int)kpList.size();
                bool lowConfidenceForceClose = reachedMax && !enoughParallax;
                windowTotalCount++;

                double velPriorWeight = lowConfidenceForceClose ? 2.0 : 0.1;

                std::vector<std::array<double,3>> landmarks;
                triangulateAndFilterWindowTracks(K, windowStates.front().pose, windowStates.back().pose,
                                                  kpList, tracks, landmarks);

                if (!tracks.empty()) {
                    WindowJob job;
                    job.windowIndex = windowCount;
                    job.framesUsed = framesUsed;
                    job.lastFrameGlobalIndex = windowCount * 1000 + framesUsed; // dosya-indeksi YOK - keyframe.frameIndex burada sira/zaman damgasi amacli bir yer tutucu
                    job.parallaxPx = parallaxPx;
                    job.lowConfidenceForceClose = lowConfidenceForceClose;
                    job.velPriorWeight = velPriorWeight;
                    job.kpList = kpList;
                    job.tracks = std::move(tracks);
                    job.landmarks = std::move(landmarks);
                    job.frameTimestamps = frameTimestamps;
                    job.windowStates = windowStates;
                    job.lastFrameDescriptors = descList.back();
                    job.incomingPriorValid = currentPrior.valid;
                    if (currentPrior.valid) job.incomingPrior = currentPrior;

                    if (!jobQueue.tryPush(std::move(job))) {
                        windowDroppedCount++;
                        std::cout << "Pencere " << windowCount << " ATLANDI (back-end kuyrugu dolu, IMU-seed ile devam)\n";
                    } else {
                        std::cout << "Pencere " << windowCount << " back-end'e gonderildi ("
                                  << framesUsed << " kare, paralaks=" << parallaxPx << "px)\n";
                    }
                } else {
                    std::cout << "Pencere " << windowCount << " ATLANDI (hicbir iz tum kareleri kapsamadi)\n";
                }

                kpList = { kpList.back() };
                descList = { descList.back() };
                frameTimestamps = { frameTimestamps.back() };
                pairwiseMatches.clear();
                windowCount++;
            }
        }

        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 'Q') { std::cout << "'q' tusuna basildi, cikiliyor...\n"; break; }
    }
}

int main() {
    // ---- Baglanti ayarlari: arkadasinin makinesinde CALISTIRILIYORSA "127.0.0.1"; bu makineden
    // UZAKTAN baglanmak icin arkadasinin IP adresine degistir (ayni ag/VPN + RPC portu (41451)
    // erisilebilir olmali - arkadasinin makinesinde dogrulanmali) ----
    const std::string airsimHost = "127.0.0.1";
    const std::string vehicleName = ""; // settings.json'da TEK arac varsa bos birak yeterli
    const std::string cameraName = "front_center"; // bkz. pip08_AirSimCalibration.hpp / AirSimMH

    const int minWindowFrames = 3;
    const int maxWindowFrames = 10;
    const double parallaxThresholdPx = 20.0;
    const int minTracksForParallaxTrust = 15;

    AirSimBridge bridge;
    std::cout << "AirSim'e baglaniliyor (" << airsimHost << ")...\n";
    bridge.connect(airsimHost, vehicleName);
    std::cout << "Baglanti kuruldu.\n";

    ORBExtractor extractor = buildExtractor();
    FeatureMatcher matcher = buildMatcher();
    cv::Mat K = AirSimMH::frontCenterK();
    cv::Mat R_BS = cv::Mat(AirSimMH::frontCenterR_BC());
    TrajectoryViewer viewer = buildViewer();

    // ---- Distorsiyon YOK varsayimi: AirSim'in temiz pinhole render'i (bkz. pip08 dosya-basi notu) -
    // main.cpp'deki gibi bir undistort remap adimina GEREK YOK. ----

    // ---- Canli IMU tamponu + tek ImuPreintegrator - main.cpp'deki (dosyadan bir kez yuklenen) imuCam/
    // imuPreint'in canli karsiligi. Front-end (yazar, addSample) ve back-end (okur, vio.optimize()
    // icindeki imuPreint.preintegrate()) AYNI nesneyi paylasir - bu yuzden imuBufferMutex ile korunur
    // (bkz. liveAirSimFrontEndLoop / backEndLoop / pip07_AsyncPipeline.hpp OptionalLockGuard). ----
    TimestampedDataSource liveImuCam;
    ImuPreintegrator imuPreint(liveImuCam);
    std::mutex imuBufferMutex;
    cv::Matx33d R_CB = cv::Matx33d(R_BS).t();

    // ---- Baslangic yercekimi kalibrasyonu - back-end thread BASLAMADAN ONCE, TEK-thread halde
    // yapilir (main.cpp'deki dosya-basi estimateGravityCameraFrame() cagrisinin canli karsiligi).
    // Arac bu sirada KALKISTAN ONCE/hover'da durgun varsayilir - ayni fiziksel varsayim main.cpp'nin
    // "ilk 200 ornek durgun" varsayimiyla AYNI (bkz. pip04_VioWindow.hpp). ----
    std::cout << "Yercekimi kalibrasyonu icin durgun IMU orneği toplaniyor (~1.5s, arac hareketsiz olmali)...\n";
    while ((int)liveImuCam.values.size() < 200) {
        AirSimImuSample s = bridge.pollImu();
        cv::Vec3d aCam = R_CB * cv::Vec3d(s.ax, s.ay, s.az);
        cv::Vec3d gCam = R_CB * cv::Vec3d(s.gx, s.gy, s.gz);
        liveImuCam.addSample(s.timestampNs, {aCam[0], aCam[1], aCam[2], gCam[0], gCam[1], gCam[2]});
    }
    cv::Vec3d gravity = estimateGravityCameraFrame(liveImuCam);
    std::cout << "Kestirilen yercekimi (kamera-0 cercevesinde): "
              << gravity[0] << ", " << gravity[1] << ", " << gravity[2]
              << " (norm=" << cv::norm(gravity) << ")\n";

    VIOBundleAdjustment vio(K, imuPreint, gravity, maxWindowFrames);
    FeatureMatcher matcherBackend = buildMatcher();

    KeyframeDatabase kfDb(0.3, 50);
    std::vector<LoopVerificationResult> foundLoops;

    VIOBundleAdjustment::FrameState initialAnchor;
    initialAnchor.pose = {0,0,0,0,0,0};
    initialAnchor.vel  = {0,0,0};
    initialAnchor.ba   = {0,0,0};
    initialAnchor.bg   = {0,0,0};
    SharedAnchorState sharedAnchor(initialAnchor);
    WindowJobQueue jobQueue(2);
    std::atomic<bool> stopRequested(false);

    int windowFailCount = 0, windowTotalCount = 0, priorSuccessCount = 0, windowDroppedCount = 0;

    // ---- Back-end (pip07'den, DEGISTIRILMEDEN) - main.cpp'deki AYNI backEndLoop, veri kaynagini
    // (dosya mi, AirSim mi) hic bilmiyor (bkz. plan Karar 2). imuBufferMutex DOLU gecirilir (EuRoC'un
    // aksine, buradaki IMU tamponu front-end tarafindan esZAMANLI buyutuluyor). ----
    std::thread beThread(backEndLoop, std::ref(vio), std::ref(jobQueue), std::ref(sharedAnchor),
                          std::ref(stopRequested), std::cref(K), std::ref(matcherBackend),
                          std::ref(kfDb), std::ref(foundLoops),
                          std::ref(windowFailCount), std::ref(priorSuccessCount), &imuBufferMutex);

    liveAirSimFrontEndLoop(bridge, cameraName, extractor, matcher, K, R_BS,
                            minWindowFrames, maxWindowFrames, parallaxThresholdPx, minTracksForParallaxTrust,
                            gravity, viewer, sharedAnchor, jobQueue, stopRequested,
                            imuPreint, liveImuCam, imuBufferMutex,
                            windowTotalCount, windowDroppedCount);

    stopRequested = true;
    jobQueue.notifyStop();
    beThread.join();

    std::cout << "Pencere ozeti: " << (windowTotalCount - windowFailCount - windowDroppedCount) << "/" << windowTotalCount
              << " basarili (" << windowFailCount << " basarisiz, " << windowDroppedCount
              << " kuyruk-dolu nedeniyle atlandi), " << priorSuccessCount
              << " pencerede kovaryans-prior hesaplanabildi\n";
    std::cout << "Keyframe sayisi: " << kfDb.keyframes.size() << ", dogrulanan dongu sayisi: "
              << foundLoops.size() << "\n";
    std::cout << "Bitti.\n";
    return 0;
}
