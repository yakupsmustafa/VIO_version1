#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/covariance.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "Imufactor.hpp"
#include "Imupreintegration.hpp"
#include <array>
#include <vector>
#include <thread>

// ---- Reprojection error (gorsel hata) - bagimsiz, disaridan dosyaya ihtiyac duymuyor ----
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

// Gorsel (reprojection) + IMU (preintegration) hatasini BIRLIKTE optimize eden BA.
class VIOBundleAdjustment {
public:
    cv::Mat K;
    int maxWindowFrames; // bilgi/kapasite ipucu - optimize() artik gercek frameStates.size()'i kullanir, bunu OKUMAZ
    cv::Vec3d gravityWorld;
    ImuPreintegrator& imuPreint;

    VIOBundleAdjustment(const cv::Mat& K_, ImuPreintegrator& imuPreint_,
                         const cv::Vec3d& gravity_ = cv::Vec3d(0,0,9.81), int maxWindowFrames_ = 5)
        : K(K_), maxWindowFrames(maxWindowFrames_), gravityWorld(gravity_), imuPreint(imuPreint_) {}

    struct FrameState {
        std::array<double,6> pose;  // [angleAxis(3), t(3)]
        std::array<double,3> vel;
        std::array<double,3> ba;    // accel bias
        std::array<double,3> bg;    // gyro bias
    };

    // ---- Bir pencerenin son karesinin poz+hiz belirsizligi (marjinalizasyon yaklasiklamasi) ----
    // Tam Schur-complement marjinalizasyonu yerine: ceres::Covariance ile SADECE bu iki blok'un
    // kovaryansi hesaplanip bir sonraki pencerede yumusak Gauss prior'u olarak enjekte edilir
    // (bkz. optimize() - incomingPrior/outgoingPrior). RowMajor ACIKCA belirtildi: Ceres
    // GetCovarianceBlock row-major ham dizi doner, Eigen'in varsayilani column-major'dir.
    // ---- ba/bg TASINMIYOR (DENENDI, GERI ALINDI) ----
    // Bias'i da ayni sekilde tasimayi denedik (onceki pencerenin bias tahmini+kovaryansini kare-0'a
    // enjekte edip, kare-0'daki mutlak-sifir regularizer'i o durumda kapattik). DETERMINISTIK A/B testte
    // (ayni binary, tek-thread, kosudan kosuya bit-bit ayni sonuc garantili) bu ATE'yi KOTULESTIRDI:
    // 1.643m -> 2.408m, basarisiz pencere sayisi de arttı (25->27). Cok-thread'li ilk testte tersi
    // gorunmustu (2.865->2.766) ama bu YANILTICI cikti - iki ayri calistirma arasindaki thread-sıralama
    // gurultusu (bkz. main.cpp determinizm notu) gercek etkiyi maskeliyordu. Olasi neden: mutlak-sifir
    // regularizer kaba ama HER pencerede bias'i gercekci MEMS araligina geri cekiyordu; onu kare-0'da
    // kapatip yerine onceki pencerenin (kisa pencerede zayif gozlenebilir, potansiyel olarak
    // asiri-guvenli/hatali) tahminini koymak, hatanin duzeltilmeden pencereler arasi birikmesine izin
    // verdi - klasik "duzgun FEJ olmadan naif marjinalizasyon" tuzagi. Sonuc: ba/bg icin mutlak-sifir
    // regularizer TUM karelerde (kare-0 dahil) korunuyor, WindowPrior sadece poz+hiz tasimaya devam
    // ediyor.
    struct WindowPrior {
        bool valid = false;
        Eigen::Matrix<double,6,6,Eigen::RowMajor> poseSqrtInfo;
        Eigen::Matrix<double,6,1> poseX0;
        Eigen::Matrix<double,3,3,Eigen::RowMajor> velSqrtInfo;
        Eigen::Matrix<double,3,1> velX0;
    };

    bool optimize(std::vector<FrameState>& frameStates,
                  const std::vector<int64_t>& frameTimestamps,
                  const std::vector<std::vector<cv::KeyPoint>>& kpList,
                  const std::vector<std::vector<int>>& tracks,
                  std::vector<std::array<double,3>>& landmarks,
                  const WindowPrior* incomingPrior = nullptr,
                  WindowPrior* outgoingPrior = nullptr,
                  double velPriorWeight = 0.1) {

        // ---- Pencere boyutu artik SABIT degil (bkz. paralaks-tabanli pencereleme, main.cpp) -
        // gercek frameStates.size()'e gore calisir, asagidaki tum dongular zaten n uzerinden genel yazilmisti ----
        int n = (int)frameStates.size();
        if (frameStates.empty() || frameTimestamps.size() != frameStates.size()) return false;

        ceres::Problem problem;
        double fx = K.at<double>(0,0), fy = K.at<double>(1,1), cx = K.at<double>(0,2), cy = K.at<double>(1,2);

        // ---- Robust loss (Huber): yanlis eslesen/dinamik-nesne izlerinden gelen buyuk reprojection
        // hatalarinin optimizasyonu domine etmesini engeller (kare yerine ~lineer ceza). Sadece GORSEL
        // hataya uygulaniyor - IMU preintegration hatasi zaten iyi modellenmis Gauss gurultusunden
        // geliyor, standart VIO pratiginde (VINS-Mono/OKVIS) robust loss IMU factor'a uygulanmaz.
        // delta=3.0px: EMPIRIK olarak secildi (bu dataset'te A/B testi, kare 2000 sonundaki ATE):
        // Huber yok -> 2.39m, delta=1.0px -> 4.40m (DAHA KOTU - gercek inlier'lari bile fazla
        // cezalandirip optimizasyonu zayiflatiyor), delta=3.0px -> 1.40m (en iyisi, ayrica basarili
        // pencere sayisi Huber-yok ile ayni kaldi: 438/454). Tek Huber ornegi TUM reprojection
        // bloklari arasinda PAYLASILIYOR (Ceres unique loss-function pointer'larini bir kez siler,
        // paylasim guvenli).
        ceres::LossFunction* huberLoss = new ceres::HuberLoss(3.0);

        for (size_t ti = 0; ti < tracks.size(); ++ti) {
            for (int fi = 0; fi < n; ++fi) {
                cv::Point2f obs = kpList[fi][tracks[ti][fi]].pt;
                ceres::CostFunction* cost = ReprojectionError::create(obs.x, obs.y, fx, fy, cx, cy);
                problem.AddResidualBlock(cost, huberLoss, frameStates[fi].pose.data(), landmarks[ti].data());
            }
        }

        for (int i = 0; i + 1 < n; ++i) {
            cv::Vec3d bg(frameStates[i].bg[0], frameStates[i].bg[1], frameStates[i].bg[2]);
            cv::Vec3d ba(frameStates[i].ba[0], frameStates[i].ba[1], frameStates[i].ba[2]);

            PreintegratedImuData preint = imuPreint.preintegrate(
                frameTimestamps[i], frameTimestamps[i+1], bg, ba);

            if (preint.deltaT <= 0) continue;

            ceres::CostFunction* imuCost = ImuFactor::create(preint, gravityWorld);
            problem.AddResidualBlock(imuCost, nullptr,
                frameStates[i].pose.data(),   frameStates[i].vel.data(),
                frameStates[i].ba.data(),     frameStates[i].bg.data(),
                frameStates[i+1].pose.data(), frameStates[i+1].vel.data(),
                frameStates[i+1].ba.data(),   frameStates[i+1].bg.data());
        }

        // ---- Bias mutlak prior'u: her karenin ba/bg'sini gercekci MEMS sinirlarina dogru ceker ----
        // (bkz. BiasPriorError yorumu - random-walk residual'i TEK BASINA bias'in sinirsizca
        // kacmasini engellemiyor, teshis edilen ıraksama vakasinda bu prior olmadigi icin oldu).
        // accelBiasPriorWeight=20 -> ~0.05 m/s^2, gyroBiasPriorWeight=200 -> ~0.005 rad/s yumusak sinir.
        // TUM karelerde (kare-0 dahil) uygulanir - bkz. WindowPrior yorumu: kare-0'i bundan MUAF tutup
        // yerine tasinan bias kovaryansini koymak DENENDI, deterministik A/B testte ATE'yi kotulestirdi,
        // geri alindi.
        const double accelBiasPriorWeight = 20.0;
        const double gyroBiasPriorWeight = 200.0;
        for (int i = 0; i < n; ++i) {
            problem.AddResidualBlock(BiasPriorError::create(accelBiasPriorWeight), nullptr, frameStates[i].ba.data());
            problem.AddResidualBlock(BiasPriorError::create(gyroBiasPriorWeight), nullptr, frameStates[i].bg.data());
        }

        // ---- Hiz mutlak prior'u: bias duzeltildikten SONRA da |v| milyonlarca m/s'ye kacmaya devam
        // etti (bkz. TESHIS ciktisi) - demek ki bias tek sebep degilmis, vel de HICBIR mutlak sinira
        // bagli degildi. BiasPriorError zaten genel "3B vektoru sifira cek" yapisi, hiz icin de
        // kullaniyoruz. Varsayilan agirlik gevsek (sigma~10 m/s) - gercekci tekli-basamak m/s hizlari
        // neredeyse hic cezalandirmaz, ama milyonlarca m/s'lik kacislara guclu fren olur. Cagiran
        // (main.cpp), zayif-paralaks/dusuk-guven pencerelerde daha SIKI bir deger geçebilir - optimizorun
        // zayif gorsel veriyle hizi kacirma serbestligini daraltmak icin (bkz. Adim 7 devami).
        for (int i = 0; i < n; ++i) {
            problem.AddResidualBlock(BiasPriorError::create(velPriorWeight), nullptr, frameStates[i].vel.data());
        }

        // ---- Kare-0 capasi: yumusak kovaryans-prior'u VARSA onu kullan, yoksa (ilk pencere ya da
        // onceki kovaryans hesaplanamadiysa) bugunku guvenli sert-sabitleme davranisina don ----
        if (incomingPrior && incomingPrior->valid) {
            problem.AddResidualBlock(PosePriorError::create(incomingPrior->poseSqrtInfo, incomingPrior->poseX0),
                                      nullptr, frameStates[0].pose.data());
            problem.AddResidualBlock(VelPriorError::create(incomingPrior->velSqrtInfo, incomingPrior->velX0),
                                      nullptr, frameStates[0].vel.data());
        } else {
            if (problem.HasParameterBlock(frameStates[0].pose.data()))
                problem.SetParameterBlockConstant(frameStates[0].pose.data());
            if (problem.HasParameterBlock(frameStates[0].vel.data()))
                problem.SetParameterBlockConstant(frameStates[0].vel.data());
        }

        ceres::Solver::Options options;
        options.max_num_iterations = 50;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        // ---- Jacobian degerlendirmesi pencere basina binlerce residual blogu (izlenen landmark sayisi x
        // kare sayisi) icerdiginden COK paralellestirilebilir - varsayilan (1 thread) BA suresinin buyuk
        // kismini (bkz. main.cpp'deki "BA+kov" olcumleri, 100ms-1000ms/pencere) aciklayan ana darbogaz.
        // FPS hedefi icin acildi (bkz. main.cpp determinizm notu - coklu-thread kucuk ATE farklarina
        // yol acabilir, kabul edilen bir bedel).
        options.num_threads = std::min(8u, std::max(1u, std::thread::hardware_concurrency()));
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        bool ok = summary.IsSolutionUsable();

        // ---- Marjinalizasyon yaklasiklamasi: son karenin poz+hiz kovaryansini hesaplayip sonraki
        // pencereye tasinacak yumusak prior'u uret. Basarisizligin HER modu guvenli sekilde valid=false'a
        // duser - dusuk-paralaks/dejenere pencerelerde bu BEKLENEN bir durumdur, hata degildir; o
        // pencerelerde sonraki cagri otomatik olarak sert-sabitlemeye geri doner. Bias'i da tasima
        // DENENDI, geri alindi - bkz. WindowPrior yorumu. ----
        if (ok && outgoingPrior) {
            outgoingPrior->valid = false;
            ceres::Covariance::Options covOpts;
            ceres::Covariance covariance(covOpts);
            double* lastPose = frameStates.back().pose.data();
            double* lastVel  = frameStates.back().vel.data();
            std::vector<std::pair<const double*, const double*>> covBlocks =
                { {lastPose, lastPose}, {lastVel, lastVel} };

            if (covariance.Compute(covBlocks, &problem)) {
                double covPoseRaw[36], covVelRaw[9];
                if (covariance.GetCovarianceBlock(lastPose, lastPose, covPoseRaw) &&
                    covariance.GetCovarianceBlock(lastVel,  lastVel,  covVelRaw)) {

                    Eigen::Map<Eigen::Matrix<double,6,6,Eigen::RowMajor>> covPose(covPoseRaw);
                    Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>> covVel(covVelRaw);

                    // Guvenlik: gercekci olmayan (asiri kucuk) varyans = supheli/asiri-guvenli kovaryans
                    const double minPlausibleVariance = 1e-12; // ~1 mikrometre std-sapma altini reddet
                    bool plausible = covPose.diagonal().minCoeff() > minPlausibleVariance &&
                                      covVel.diagonal().minCoeff()  > minPlausibleVariance;

                    if (plausible) {
                        Eigen::LLT<Eigen::Matrix<double,6,6>> lltP(covPose);
                        Eigen::LLT<Eigen::Matrix<double,3,3>> lltV(covVel);
                        if (lltP.info() == Eigen::Success && lltV.info() == Eigen::Success) {
                            Eigen::Matrix<double,6,6> sqrtInfoP = lltP.matrixL().toDenseMatrix()
                                .triangularView<Eigen::Lower>().solve(Eigen::Matrix<double,6,6>::Identity());
                            Eigen::Matrix<double,3,3> sqrtInfoV = lltV.matrixL().toDenseMatrix()
                                .triangularView<Eigen::Lower>().solve(Eigen::Matrix<double,3,3>::Identity());
                            if (sqrtInfoP.allFinite() && sqrtInfoV.allFinite()) {
                                outgoingPrior->poseSqrtInfo = sqrtInfoP;
                                outgoingPrior->velSqrtInfo  = sqrtInfoV;
                                outgoingPrior->poseX0 = Eigen::Map<Eigen::Matrix<double,6,1>>(lastPose);
                                outgoingPrior->velX0  = Eigen::Map<Eigen::Matrix<double,3,1>>(lastVel);
                                outgoingPrior->valid = true;
                            }
                        }
                    }
                }
            }
        }

        return ok;
    }
};
