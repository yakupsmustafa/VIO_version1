    //  VisualOdometry

#include <opencv2/opencv.hpp>
#include <iostream>
#include <optional>
#include "Metrics.h"
#include "pip01_FeatureExtractor.h"
#include "DataLoader.h"
#include "TimestampedDataSource.hpp"
#include "Synchronizer.hpp"
#include "pip02_Matcher.hpp"
#include "Test.hpp"
#include "Imupreintegration.hpp"
#include "Imufactor.hpp"
#include "Viobundleadjustment.hpp"
#include "pip04_VioWindow.hpp"
#include "pip06_LoopClosure.hpp"
#include "pip07_AsyncPipeline.hpp"
#include <thread>

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

// ---- GT'yi goruntu zaman damgalarina hizalar (canli gorsellestirme/ATE icin) ----
// NOT: IMU BURADA hizalanmiyor - ham IMU, ImuPreintegrator uzerinden zaten dogru sekilde (her
// pencerenin gercek kare zaman damgalari arasinda) preintegre ediliyor; bu fonksiyonun eskiden
// urettigi ayrica-hizalanmis IMU dizisi hicbir yerde kullanilmiyordu (gereksiz hesaplama).
std::vector<std::vector<double>> synchronizeGt(
        const ImageLoader& images, const CSVGroundTruthSource& gt) {
    Synchronizer sync(images, {&gt});
    auto aligned = sync.alignAll();
    std::cout << "Hizalanan GT orneği: " << aligned[0].size() << "\n";
    return aligned[0];
}

// ---- ORB extractor'i olusturur ----
ORBExtractor buildExtractor() {
    ORBParams params;
    // 10000 -> 3000: BFMatcher maliyeti O(N*M) oldugundan bu ~11x eslestirme hizlanmasi saglar
    // (izlerin pencerede hayatta kalma oranini fazla dusurmemek icin agresif kisilmadi).
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

// ---- Radial-tangential (radtan) distorsiyon katsayilari (k1,k2,p1,p2) ----
// buildCameraMatrix() ile AYNI EuRoC MH01 cam0 kalibrasyon kaynagindan. Ham EuRoC goruntuleri
// distorsiyonludur; bu katsayilar olmadan findEssentialMat/triangulatePoints/reprojection hatasi
// sistematik olarak yanlis cikar (goruntu kenarlarina dogru gittikce artan bir sapmayla).
cv::Mat buildDistortionCoeffs() {
    return (cv::Mat_<double>(4,1) << -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05);
}

// ---- Kameradan govdeye (body frame) rotasyon - cam0/sensor.yaml'daki GERCEK T_BS'den ----
// VO/VIO, kameranin KENDI koordinat sisteminde ilerliyor; GT ise govde (body) koordinat sisteminde.
// Ayrica IMU olcumlerini kamera cercevesine dondurmek icin de (R_CB = R_BS^T) kullanilir.
cv::Mat buildCameraToBodyRotation() {
    return (cv::Mat_<double>(3,3) <<
        0.0148655429818, -0.999880929698,  0.00414029679422,
        0.999557249008,   0.0149672133247, 0.025715529948,
       -0.0257744366974,  0.00375618835797, 0.999660727178);
}

// ---- TrajectoryViewer'i olusturur ----
TrajectoryViewer buildViewer() {
    ViewerParams vparams;
    vparams.showFrame = true;
    vparams.showTrajectory = true;
    vparams.frameWindowName = "Kamera Goruntusu";
    vparams.trajWindowName = "Yorunge (VO vs GT)";
    vparams.waitKeyDelay = 1;

    vparams.canvasSize = 800;
    vparams.trajScale = 3.0; // autoFit=true iken bu deger otomatik hesaplanir, elle vermene gerek yok
    vparams.voColor = cv::Scalar(0, 255, 0);
    vparams.gtColor = cv::Scalar(0, 0, 255);
    vparams.pointRadius = 2;
    vparams.lineThickness = 2;

    // ---- GT gorsel olcekleme ----
    vparams.autoScaleGT = true;      // false yaparsan manualGtScale kullanilir
    vparams.manualGtScale = 20.0;     // autoScaleGT=false iken GT'ye uygulanacak SABIT olcek
    vparams.minGtPathLength = 1e-6;
    // -75 derece, Adim 1'den ONCEKI (kamera-merkezi hatasi duzeltilmeden once) yanlis konumu gozle
    // telafi etmeye calisan bir kalinti hileydi. rotateVo() SADECE cizimi etkiler, ATE'yi ETKILEMEZ
    // (computeLiveATE ham voPath'i kullanir) - yani bu deger ATE'yi degil, sadece VO/GT cizgilerinin
    // gorsel hizasini bozuyordu. Matematik artik dogru oldugu icin kaldirildi.
    vparams.voRotationDeg = 0.0;

    // ---- Otomatik yakinlastirma ----
    vparams.autoFit = true;
    vparams.fitMargin = 0.15;
    vparams.lateralBoost = 1.0;      // X eksenini bagimsiz buyutmek istersen >1 yap

    // ---- Eksen isareti duzeltmesi: GT'nin X ekseni VO'ya gore ters gozlemlendigi icin aynalandi ----
    vparams.gtAxisSign = cv::Vec3d(1, 1, 1); // artik R_BS geometrik olarak dogru donusumu yapiyor, hileye gerek yok

    // ---- Metrikler ----
    vparams.showMetrics = true;
    vparams.computeLiveATE = true;
    vparams.ateUpdateInterval = 10;
    vparams.minPointsForATE = 3;

    return TrajectoryViewer(vparams);
}

// ---- FRONT-END: kare-basi oku+distorsiyon-duzelt+ORB+eslestir+IMU-seed+canli gorsellestirme.
// Pencere hazir oldugunda BA'yi DOGRUDAN CAGIRMAZ - bir WindowJob paketleyip kuyruga (tryPush) atar
// ve BEKLEMEDEN devam eder. Kuyruk doluysa (back-end geride kaldiysa) pencere ATLANIR - front-end
// hizi hicbir zaman dusmez (bkz. pip07_AsyncPipeline.hpp). ----
void frontEndLoop(const ImageLoader& images, ORBExtractor& extractor, FeatureMatcher& matcher,
                   const cv::Mat& K, const cv::Mat& R_BS,
                   const cv::Mat& undistortMap1, const cv::Mat& undistortMap2,
                   int minWindowFrames, int maxWindowFrames, double parallaxThresholdPx,
                   int minTracksForParallaxTrust,
                   const cv::Vec3d& gravity, ImuPreintegrator& imuPreint,
                   const std::vector<std::vector<double>>& alignedGt,
                   TrajectoryViewer& viewer,
                   SharedAnchorState& sharedAnchor, WindowJobQueue& jobQueue,
                   int& windowTotalCount, int& windowDroppedCount) {

    std::vector<std::vector<cv::KeyPoint>> kpList;
    std::vector<cv::Mat> descList;
    std::vector<int64_t> frameTimestamps;
    std::vector<std::vector<cv::DMatch>> pairwiseMatches; // pairwiseMatches[s]: local kare s -> s+1

    int windowCount = 0;

    for (size_t i = 0; i < images.imageFiles.size(); ++i) {
        cv::Mat imageRaw = cv::imread(images.imageFiles[i], cv::IMREAD_GRAYSCALE);
        if (imageRaw.empty()) {
            std::cerr << "HATA: goruntu okunamadi: " << images.imageFiles[i] << "\n";
            continue;
        }
        double tick = cv::getTickFrequency();
        int64 tDetectStart = cv::getTickCount();
        cv::Mat image;
        cv::remap(imageRaw, image, undistortMap1, undistortMap2, cv::INTER_LINEAR);

        std::vector<cv::KeyPoint> kps;
        cv::Mat descs;
        extractor.detect(image, kps, descs);
        double detectMs = (cv::getTickCount() - tDetectStart) / tick * 1000.0;
        std::cout << "Kare " << i << " -> bulunan kose sayisi: " << kps.size()
                  << " (remap+detect: " << detectMs << " ms)\n";

        kpList.push_back(kps);
        descList.push_back(descs);
        frameTimestamps.push_back(images.timestamps[i]);

        if (kpList.size() > 1) {
            int64 tMatchStart = cv::getTickCount();
            std::vector<cv::DMatch> m = matcher.match(descList[descList.size() - 2], descList.back());
            double matchMs = (cv::getTickCount() - tMatchStart) / tick * 1000.0;
            std::cout << "  Kare " << (i - 1) << " -> " << i << " eslesme sayisi: " << m.size()
                      << " (match: " << matchMs << " ms)\n";
            pairwiseMatches.push_back(m);
        }

        // ---- IMU dead-reckoning ile o ana kadarki pencereyi PAYLASILAN capa uzerinden seed'le
        // (canli gorsellestirme icin) - back-end'in EN SON yayinladigi degeri okur (mutex korumali) ----
        VIOBundleAdjustment::FrameState anchor;
        VIOBundleAdjustment::WindowPrior currentPrior;
        sharedAnchor.get(anchor, currentPrior);

        std::vector<VIOBundleAdjustment::FrameState> windowStates =
            seedWindowStatesFromImu(imuPreint, anchor, frameTimestamps, gravity);

        cv::Point3d camCenter = cameraCenterFromPose(windowStates.back().pose);
        cv::Mat centerMat = (cv::Mat_<double>(3,1) << camCenter.x, camCenter.y, camCenter.z);
        cv::Mat voBody = R_BS * centerMat;
        cv::Point3d voPos(voBody.at<double>(0), voBody.at<double>(1), voBody.at<double>(2));

        std::optional<cv::Point3d> gtPos;
        if (i < alignedGt.size() && alignedGt[i].size() == 3) {
            gtPos = cv::Point3d(alignedGt[i][0], alignedGt[i][1], alignedGt[i][2]);
        }
        viewer.update(image, voPos, gtPos);

        // ---- Pencere kapatma kontrolu: paralaks-tabanli (sabit kare sayisi yerine) ----
        if ((int)kpList.size() >= minWindowFrames) {
            std::vector<std::vector<int>> tracks = buildFullCoverageTracks(pairwiseMatches, (int)kpList[0].size());
            double parallaxPx = computeMedianParallaxPx(kpList, tracks);
            bool reachedMax = (int)kpList.size() >= maxWindowFrames;
            bool enoughParallax = ((int)tracks.size() >= minTracksForParallaxTrust) && (parallaxPx >= parallaxThresholdPx);

            if (enoughParallax || reachedMax) {
                int framesUsed = (int)kpList.size();
                bool lowConfidenceForceClose = reachedMax && !enoughParallax;
                windowTotalCount++;

                double velPriorWeight = lowConfidenceForceClose ? 2.0 : 0.1; // sigma ~0.5 m/s vs ~10 m/s

                std::vector<std::array<double,3>> landmarks;
                triangulateAndFilterWindowTracks(K, windowStates.front().pose, windowStates.back().pose,
                                                  kpList, tracks, landmarks);

                if (!tracks.empty()) {
                    WindowJob job;
                    job.windowIndex = windowCount;
                    job.framesUsed = framesUsed;
                    job.lastFrameGlobalIndex = (int)i;
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

                // ---- Pencereyi kaydir: 1 kare cakismali - sadece son (paylasilan) kareyi tut ----
                kpList = { kpList.back() };
                descList = { descList.back() };
                frameTimestamps = { frameTimestamps.back() };
                pairwiseMatches.clear();
                windowCount++;
            }
        }

        // ---- 'q' tusuna basilirsa dongudden cikip programi duzgunce sonlandir ----
        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 'Q') {
            std::cout << "'q' tusuna basildi, cikiliyor...\n";
            break;
        }
    }
}

int main() {
    // ---- OpenCV'nin ic paralelligi VARSAYILAN (coklu-thread) birakildi ----
    // NOT: cv::setNumThreads(1) ile ayni binarinin CALISTIRMADAN CALISTIRMAYA farkli ATE uretebildigi
    // DOGRULANDI (kare ~300'den itibaren ayrisma) - sebep muhtemelen esit-skorlu kose noktalarinin
    // thread'ler arasi sıralama gurultusu. Ama gercek-zamanli (35-40 FPS) hedefi, bit-bit tekrar-
    // uretilebilirlikten daha oncelikli - gercek VIO sistemleri (VINS-Mono, ORB-SLAM3 vb.) de coklu-
    // thread calisir ve kosudan kosuya bit-bit ayni sonucu VERMEZ. Tek-thread modu, sadece kontrollu
    // A/B parametre karsilastirmalari yaparken GECICI olarak (test build'inde) acilabilir, varsayilan
    // koda konmaz.
    const std::string datasetRoot = "../EUROC_MH01_Easy";
    // 800 -> 2000: ilk 800 karede (~40s) GT sadece ~0.5m'lik alanda hareket ediyor (kalkis sonrasi
    // durgun bolge), bu da kucuk mutlak hatalari gorsel olarak abartili gosteriyordu. 2000 kare
    // (~100s, ~40m gercek yol) VIO dogrulugunu cok daha temsili bir sekilde test eder.
    const int nFrames = 2000;
    // Sabit windowSize=5 yerine paralaks-tabanli degisken pencere (bkz. Adim 6 plani):
    // dusuk-hareketli donemlerde 5 kare yetersiz taban (baseline) verip triangulasyonu dejenere
    // ediyordu (TUM izler cheirality'den aynı anda eleniyordu). Simdi pencere, kamera GERCEKTEN
    // yeterince hareket edene (parallaxThresholdPx) kadar acik kaliyor.
    const int minWindowFrames = 3;
    const int maxWindowFrames = 10;
    const double parallaxThresholdPx = 20.0;
    const int minTracksForParallaxTrust = 15;

    DatasetPaths paths = buildDatasetPaths(datasetRoot);

    ImageLoader images = loadImages(paths.imageFolder, nFrames);
    CSVImuSource imu = loadImu(paths.imuCsvPath);
    CSVGroundTruthSource gt = loadGroundTruth(paths.gtCsvPath);

    const auto alignedGt = synchronizeGt(images, gt);

    ORBExtractor extractor = buildExtractor();
    FeatureMatcher matcher = buildMatcher();
    cv::Mat K = buildCameraMatrix();
    cv::Mat distCoeffs = buildDistortionCoeffs();
    cv::Mat R_BS = buildCameraToBodyRotation();
    TrajectoryViewer viewer = buildViewer();

    // ---- Distorsiyon duzeltme haritalarini bir kez onceden hesapla (her karede yeniden hesaplamaktan ucuz) ----
    cv::Mat firstImage = cv::imread(images.imageFiles[0], cv::IMREAD_GRAYSCALE);
    cv::Mat undistortMap1, undistortMap2;
    cv::initUndistortRectifyMap(K, distCoeffs, cv::Mat(), K, firstImage.size(),
                                 CV_16SC2, undistortMap1, undistortMap2);

    // ---- IMU'yu kamera cercevesine dondur + statik yercekimi kalibrasyonu ----
    TimestampedDataSource imuCam = rotateImuToCameraFrame(imu, R_BS);
    ImuPreintegrator imuPreint(imuCam);
    cv::Vec3d gravity = estimateGravityCameraFrame(imuCam);
    std::cout << "Kestirilen yercekimi (kamera-0 cercevesinde): "
              << gravity[0] << ", " << gravity[1] << ", " << gravity[2]
              << " (norm=" << cv::norm(gravity) << ")\n";

    VIOBundleAdjustment vio(K, imuPreint, gravity, maxWindowFrames);
    // ---- Back-end'in loop-closure eslestirmesi icin AYRI bir FeatureMatcher - front-end'in kendi
    // `matcher`'iyla ayni anda, iki farkli thread'den cagrilma ihtimaline karsi (paylasilan mutable
    // state olmasa da, ekstra kanit yukune GEREK BIRAKMAMAK icin ayri, ucuz bir kopya tercih edildi). ----
    FeatureMatcher matcherBackend = buildMatcher();

    KeyframeDatabase kfDb(0.3, 50); // ~0.3m'den az hareket VE 50 kareden az gecmisse yeni keyframe eklenmez
    std::vector<LoopVerificationResult> foundLoops;

    // ---- Front-end/back-end ayrisimi (bkz. pip07_AsyncPipeline.hpp): BA+kovaryans AYRI bir thread'de
    // (back-end) calisir, front-end (bu thread, ana kare dongusu) HIC BEKLEMEZ. ----
    VIOBundleAdjustment::FrameState initialAnchor;
    initialAnchor.pose = {0,0,0,0,0,0};
    initialAnchor.vel  = {0,0,0};
    initialAnchor.ba   = {0,0,0};
    initialAnchor.bg   = {0,0,0};
    SharedAnchorState sharedAnchor(initialAnchor);
    WindowJobQueue jobQueue(2); // kuyruk derinligi: back-end en fazla 2 pencere geride kalabilir
    std::atomic<bool> stopRequested(false);

    int windowFailCount = 0, windowTotalCount = 0, priorSuccessCount = 0, windowDroppedCount = 0;

    // ---- imuBufferMutex=nullptr: EuRoC IMU tamponu (imuCam) dosyadan bir kez yuklenip BIR DAHA
    // DEGISMEDIGI icin coklu-okuyucu (front-end+back-end) guvenlidir, kilide GEREK YOK (bkz.
    // backEndLoop/OptionalLockGuard yorumu, pip07_AsyncPipeline.hpp) ----
    std::thread beThread(backEndLoop, std::ref(vio), std::ref(jobQueue), std::ref(sharedAnchor),
                          std::ref(stopRequested), std::cref(K), std::ref(matcherBackend),
                          std::ref(kfDb), std::ref(foundLoops),
                          std::ref(windowFailCount), std::ref(priorSuccessCount), nullptr);

    frontEndLoop(images, extractor, matcher, K, R_BS, undistortMap1, undistortMap2,
                 minWindowFrames, maxWindowFrames, parallaxThresholdPx, minTracksForParallaxTrust,
                 gravity, imuPreint, alignedGt, viewer,
                 sharedAnchor, jobQueue, windowTotalCount, windowDroppedCount);

    // ---- Front-end bitti: back-end'e "artik yeni is gelmeyecek" sinyali ver, kuyrukta kalanlari
    // BITIRMESINI bekle (join) - final ozet/ATE ancak back-end tamamen bosaldiktan SONRA dogru olur. ----
    stopRequested = true;
    jobQueue.notifyStop();
    beThread.join();

    std::cout << "Pencere ozeti: " << (windowTotalCount - windowFailCount - windowDroppedCount) << "/" << windowTotalCount
              << " basarili (" << windowFailCount << " basarisiz, " << windowDroppedCount
              << " kuyruk-dolu nedeniyle atlandi), " << priorSuccessCount
              << " pencerede kovaryans-prior hesaplanabildi\n";
    std::cout << "Keyframe sayisi: " << kfDb.keyframes.size() << ", dogrulanan dongu sayisi: "
              << foundLoops.size() << "\n";

    // ---- Loop closure: en az bir dongu dogrulandiysa pose graph coz, duzeltilmis yorungeyle ATE'yi
    // AYRICA raporla (canli/orijinal ATE'ye DOKUNULMAZ - bkz. plan, Karar 6) ----
    // ---- Keyframe-orneklemeli ATE hesaplayan yardimci - once/sonra ADIL (AYNI nokta seti) kiyas icin ----
    auto computeKeyframeAte = [&](const std::vector<std::array<double,6>>& kfPoses) -> double {
        std::vector<cv::Point3d> voPts, gtPts;
        for (size_t k = 0; k < kfPoses.size(); ++k) {
            int fi = kfDb.keyframes[k].frameIndex;
            if (fi < 0 || fi >= (int)alignedGt.size() || alignedGt[fi].size() != 3) continue;
            cv::Point3d c = cameraCenterFromPose(kfPoses[k]);
            cv::Mat cMat = (cv::Mat_<double>(3,1) << c.x, c.y, c.z);
            cv::Mat body = R_BS * cMat;
            voPts.emplace_back(body.at<double>(0), body.at<double>(1), body.at<double>(2));
            gtPts.emplace_back(alignedGt[fi][0], alignedGt[fi][1], alignedGt[fi][2]);
        }
        if (voPts.size() < 3) return -1.0;
        Eigen::MatrixXd vo(3, voPts.size()), gt(3, gtPts.size());
        for (size_t k = 0; k < voPts.size(); ++k) {
            vo(0,k)=voPts[k].x; vo(1,k)=voPts[k].y; vo(2,k)=voPts[k].z;
            gt(0,k)=gtPts[k].x; gt(1,k)=gtPts[k].y; gt(2,k)=gtPts[k].z;
        }
        Eigen::MatrixXd T = umeyamaAlign(vo, gt);
        Eigen::MatrixXd voAligned(voPts.size(), 3), gtMat(gtPts.size(), 3);
        for (size_t k = 0; k < voPts.size(); ++k) {
            Eigen::Vector4d ph(vo(0,k), vo(1,k), vo(2,k), 1.0);
            Eigen::Vector4d pa = T * ph;
            voAligned.row(k) = pa.head<3>().transpose();
            gtMat.row(k) = Eigen::Vector3d(gt(0,k), gt(1,k), gt(2,k)).transpose();
        }
        return computeATE(voAligned, gtMat);
    };

    if (!foundLoops.empty()) {
        const LoopVerificationResult& loop = foundLoops.front();

        std::vector<std::array<double,6>> uncorrectedPoses;
        for (const auto& kf : kfDb.keyframes) uncorrectedPoses.push_back(kf.pose);
        double ateBefore = computeKeyframeAte(uncorrectedPoses);

        auto correctedPoses = solvePoseGraph(kfDb, loop, /*odomRotW*/50.0, /*odomTransW*/50.0, /*loopW*/100.0);
        double ateAfter = computeKeyframeAte(correctedPoses);

        std::cout << "[LOOP CLOSURE] keyframe-bazli ATE (" << kfDb.keyframes.size() << " keyframe, "
                  << "ayni nokta seti): duzeltme-ONCESI=" << ateBefore
                  << " m, duzeltme-SONRASI=" << ateAfter << " m\n";
    }

    std::cout << "Bitti.\n";
    return 0;
}
