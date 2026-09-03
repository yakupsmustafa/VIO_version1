#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include "Imupreintegration.hpp"

// ---- Kucuk 3x3 matris/vektor yardimcilari (T=double ya da ceres::Jet olabilir) ----
template <typename T>
inline void mat3Mul(const T A[9], const T B[9], T C[9]) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            C[r*3+c] = T(0);
            for (int k = 0; k < 3; ++k) C[r*3+c] += A[r*3+k] * B[k*3+c];
        }
}
template <typename T>
inline void mat3Transpose(const T A[9], T At[9]) {
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) At[c*3+r] = A[r*3+c];
}
template <typename T>
inline void mat3VecMul(const T A[9], const T v[3], T out[3]) {
    for (int r = 0; r < 3; ++r) {
        out[r] = T(0);
        for (int c = 0; c < 3; ++c) out[r] += A[r*3+c] * v[c];
    }
}
template <typename T>
inline void mat3FromDouble(const cv::Matx33d& M, T out[9]) {
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) out[r*3+c] = T(M(r,c));
}

// ---- IMU Factor: iki ardisik kare (i,j) arasindaki IMU tutarlilik hatasi ----
// Parametre bloklari: pose_i[6], vel_i[3], ba_i[3], bg_i[3], pose_j[6], vel_j[3], ba_j[3], bg_j[3]
// pose = [angleAxis(3), t(3)] - BundleAdjustment.hpp'deki ReprojectionError ile AYNI kural.
// Residual (15 boyutlu): [rotasyon(3), hiz(3), konum(3), gyro_bias_rw(3), accel_bias_rw(3)]
// ---- IMU agirliklari: GERCEK sensor gurultusunden turetme DENENDI, GERI ALINDI ----
// EuRoC MH01 IMU (ADIS16448) resmi/yayinlanmis gurultu parametreleriyle (gyro_noise=1.6968e-4,
// accel_noise=2e-3, vb. - VINS-Mono/OKVIS/Kimera'nin da kullandigi standart degerler) beyaz-gurultu
// integrasyon formulunden (std ~ sigma*sqrt(T), konum icin sigma*sqrt(T^3/3); agirlik=1/std) hesaplanan
// agirliklar DENENDI. Deterministik A/B testte (tek-thread, kosudan kosuya bit-bit ayni sonuc) KATMERLI
// KOTULESTIRDI: ATE 1.395m -> 3.221m, basarisiz pencere orani 18/454 -> 197/454, final |v| 0.01 m/s'den
// 33.7 m/s'ye kacti. Bu, kod tarihindeki eski bulguyla (agirliklari sadece 20x artirmanin bile
// KATMERLI kotulestirdigi) TUTARLI: sorun "IMU az agirlikli" degil, IMU dead-reckoning'in kendisinde
// (preintegration/seed) TESHIS EDILMEMIS baska bir sorun var - agirlik ne kadar "dogru" hesaplanirsa
// hesaplansin, o bozuk sinyale ne kadar cok guvenirsen o kadar kotu sonuc aliyorsun. Eski sabit
// (elle ayarlanmis ama EMPIRIK OLARAK CALISAN) degerlere donduruldu. Asil kok nedeni (dead-reckoning'de
// ne bozuk?) once teshis etmeden agirlik formulu degistirilmemeli.
struct ImuFactor {
    PreintegratedImuData preint;
    cv::Vec3d gravityWorld;
    double rotWeight, velWeight, posWeight, bgWeight, baWeight;

    ImuFactor(const PreintegratedImuData& p, const cv::Vec3d& g)
        : preint(p), gravityWorld(g) {
        rotWeight = 100.0; velWeight = 10.0; posWeight = 50.0; bgWeight = 1.0; baWeight = 1.0;
    }

    template <typename T>
    bool operator()(const T* pose_i, const T* vel_i, const T* ba_i, const T* bg_i,
                     const T* pose_j, const T* vel_j, const T* ba_j, const T* bg_j,
                     T* residual) const {

        T dt = T(preint.deltaT);

        // ---- R_i, R_j (angle-axis -> rotasyon matrisi) ----
        T Ri[9], Rj[9];
        ceres::AngleAxisToRotationMatrix(pose_i, Ri); // column-major dondurur (ceres kurali)
        ceres::AngleAxisToRotationMatrix(pose_j, Rj);
        // ceres::AngleAxisToRotationMatrix column-major yazar; row-major olarak transpoze alalim
        // NOT: "RiT"/"RjT" adina ragmen bunlar sayisal olarak R_i^T/R_j^T DEGIL, R_i/R_j'nin kendisidir
        // (column-major->row-major cevriminin yan etkisi budur - dogrulanmis). rotErr/velErr bu
        // konvansiyonla zaten dogru sonucu veriyor (world->camera parametrizasyonuyla tutarli), o yuzden
        // onlara DOKUNULMADI. Gercek R_i^T/R_j^T'ye ihtiyac duyan (asagidaki Ci/Cj) icin ayrica transpoze
        // alinacak.
        T RiT[9], RjT[9];
        mat3Transpose(Ri, RiT); // aslinda column-major->row-major cevrimi de yapiyor (=R_i, R_i^T degil)
        mat3Transpose(Rj, RjT); // (=R_j, R_j^T degil)

        // ---- Bias farki (linearizasyon noktasindan sapma) ----
        T dbg[3] = { bg_i[0] - T(preint.biasGyroLin[0]),
                     bg_i[1] - T(preint.biasGyroLin[1]),
                     bg_i[2] - T(preint.biasGyroLin[2]) };
        T dba[3] = { ba_i[0] - T(preint.biasAccelLin[0]),
                     ba_i[1] - T(preint.biasAccelLin[1]),
                     ba_i[2] - T(preint.biasAccelLin[2]) };

        // ---- Bias-duzeltilmis preintegre rotasyon: dR_corrected = dR_meas * Exp(dR_dbg * dbg) ----
        T dRdbg[9]; mat3FromDouble(preint.dR_dbg, dRdbg);
        T correction[3]; mat3VecMul(dRdbg, dbg, correction);
        T dRcorrRot[9]; ceres::AngleAxisToRotationMatrix(correction, dRcorrRot);

        T dRmeas[9]; mat3FromDouble(preint.deltaR, dRmeas);
        T dRcorrected[9]; mat3Mul(dRmeas, dRcorrRot, dRcorrected);

        // ---- Rotasyon residual: Log( dRcorrected^T * Ri^T * Rj ) ----
        T dRcorrT[9]; mat3Transpose(dRcorrected, dRcorrT);
        T tmp1[9]; mat3Mul(RiT, Rj, tmp1);
        T Rerr[9]; mat3Mul(dRcorrT, tmp1, Rerr);
        T rotErr[3];
        ceres::RotationMatrixToAngleAxis(Rerr, rotErr);

        // ---- Bias-duzeltilmis preintegre hiz/konum ----
        T dVdba[9]; mat3FromDouble(preint.dV_dba, dVdba);
        T dVdbg[9]; mat3FromDouble(preint.dV_dbg, dVdbg);
        T dPdba[9]; mat3FromDouble(preint.dP_dba, dPdba);
        T dPdbg[9]; mat3FromDouble(preint.dP_dbg, dPdbg);

        T corrVa[3], corrVg[3]; mat3VecMul(dVdba, dba, corrVa); mat3VecMul(dVdbg, dbg, corrVg);
        T corrPa[3], corrPg[3]; mat3VecMul(dPdba, dba, corrPa); mat3VecMul(dPdbg, dbg, corrPg);

        T dVcorrected[3] = { T(preint.deltaV[0]) + corrVa[0] + corrVg[0],
                              T(preint.deltaV[1]) + corrVa[1] + corrVg[1],
                              T(preint.deltaV[2]) + corrVa[2] + corrVg[2] };
        T dPcorrected[3] = { T(preint.deltaP[0]) + corrPa[0] + corrPg[0],
                              T(preint.deltaP[1]) + corrPa[1] + corrPg[1],
                              T(preint.deltaP[2]) + corrPa[2] + corrPg[2] };

        T g[3]  = { T(gravityWorld[0]), T(gravityWorld[1]), T(gravityWorld[2]) };

        // ---- Hiz residual: Ri^T * (v_j - v_i - g*dt) - dVcorrected ----
        T dv[3] = { vel_j[0]-vel_i[0]-g[0]*dt, vel_j[1]-vel_i[1]-g[1]*dt, vel_j[2]-vel_i[2]-g[2]*dt };
        T RiTdv[3]; mat3VecMul(RiT, dv, RiTdv);
        T velErr[3] = { RiTdv[0]-dVcorrected[0], RiTdv[1]-dVcorrected[1], RiTdv[2]-dVcorrected[2] };

        // ---- Kamera merkezleri (FIZIKSEL dunya konumu, pose[3..5]=t_wc DEGIL): C = -R^T * t ----
        // IMU, kameranin gercek dunya konumunu integre eder; pose_i/pose_j[3..5] ise world->camera
        // extrinsic otelemesidir (ReprojectionError'daki ile ayni). Bu ikisi t=-R*C ile iliskili ama
        // ESIT DEGIL (R_i != R_j oldugu surece) - Ci/Cj'yi burada acikca cikarmak gerekiyor.
        T RiTrue_T[9], RjTrue_T[9];
        mat3Transpose(RiT, RiTrue_T); // gercek R_i^T (RiT zaten R_i'nin kendisiydi)
        mat3Transpose(RjT, RjTrue_T); // gercek R_j^T
        T Ci[3], Cj[3];
        {
            T negTi[3] = { -pose_i[3], -pose_i[4], -pose_i[5] };
            mat3VecMul(RiTrue_T, negTi, Ci);
            T negTj[3] = { -pose_j[3], -pose_j[4], -pose_j[5] };
            mat3VecMul(RjTrue_T, negTj, Cj);
        }

        // ---- Konum residual: Ri^T * (C_j - C_i - v_i*dt - 0.5*g*dt^2) - dPcorrected ----
        T half_g_dt2[3] = { T(0.5)*g[0]*dt*dt, T(0.5)*g[1]*dt*dt, T(0.5)*g[2]*dt*dt };
        T dp[3] = { Cj[0]-Ci[0]-vel_i[0]*dt-half_g_dt2[0],
                    Cj[1]-Ci[1]-vel_i[1]*dt-half_g_dt2[1],
                    Cj[2]-Ci[2]-vel_i[2]*dt-half_g_dt2[2] };
        T RiTdp[3]; mat3VecMul(RiT, dp, RiTdp);
        T posErr[3] = { RiTdp[0]-dPcorrected[0], RiTdp[1]-dPcorrected[1], RiTdp[2]-dPcorrected[2] };

        // ---- Bias random-walk residual (bias'in kareler arasi cok degismemesi beklenir) ----
        T bgErr[3] = { bg_j[0]-bg_i[0], bg_j[1]-bg_i[1], bg_j[2]-bg_i[2] };
        T baErr[3] = { ba_j[0]-ba_i[0], ba_j[1]-ba_i[1], ba_j[2]-ba_i[2] };

        // ---- Ciktilari agirliklandirarak yaz ----
        for (int k = 0; k < 3; ++k) residual[k]      = T(rotWeight) * rotErr[k];
        for (int k = 0; k < 3; ++k) residual[3+k]    = T(velWeight) * velErr[k];
        for (int k = 0; k < 3; ++k) residual[6+k]    = T(posWeight) * posErr[k];
        for (int k = 0; k < 3; ++k) residual[9+k]    = T(bgWeight)  * bgErr[k];
        for (int k = 0; k < 3; ++k) residual[12+k]   = T(baWeight)  * baErr[k];

        return true;
    }

    static ceres::CostFunction* create(const PreintegratedImuData& p, const cv::Vec3d& g) {
        return new ceres::AutoDiffCostFunction<ImuFactor, 15, 6,3,3,3, 6,3,3,3>(
            new ImuFactor(p, g));
    }
};

// ---- Bias mutlak prior'u: ba/bg'yi SIFIRA dogru ceker ----
// ImuFactor'deki bias residual'i sadece "iki ardisik kare arasinda bias cok DEGISMESIN" diyor
// (random-walk) - ama hicbir sey bias'in MUTLAK buyuklugunu sinirlamiyor. Teshis edilen ıraksama
// vakasinda (bkz. main.cpp TESHIS ciktisi) bu yuzden |ba| birkaç pencerede 0.9 -> 7.56 -> 1119 m/s^2
// gibi fiziksel olarak imkansiz degerlere kacabiliyordu (gercek MEMS ivmeolcer/gyro bias'i
// tipik olarak <0.1 m/s^2 / <0.01 rad/s civarindadir). Bu prior, bias'i o gercekci araligin
// disina cikmaya calistiginda cezalandirarak kacisi engeller (VINS-Mono/OKVIS'teki bias prior'una
// benzer standart bir VIO teknigi).
struct BiasPriorError {
    double weight;
    explicit BiasPriorError(double w) : weight(w) {}

    template <typename T>
    bool operator()(const T* const bias, T* residual) const {
        residual[0] = T(weight) * bias[0];
        residual[1] = T(weight) * bias[1];
        residual[2] = T(weight) * bias[2];
        return true;
    }

    static ceres::CostFunction* create(double w) {
        return new ceres::AutoDiffCostFunction<BiasPriorError, 3, 3>(new BiasPriorError(w));
    }
};

// ---- Genel kovaryans-agirlikli Gauss prior'u: residual = sqrtInfo * (x - x0) ----
// BiasPriorError'dan farki: izotropik tek bir agirlik yerine TAM (anizotropik/korelasyonlu) bir
// sqrt-bilgi matrisi kullanir - onceki pencerenin ceres::Covariance ile hesaplanan gercek belirsizligini
// tasir (bkz. Viobundleadjustment.hpp WindowPrior). Eigen ctor'u SADECE problem kurulurken (T=double)
// cagrilir - AutoDiff'in Jet tipiyle Eigen matrisi hic karismaz, ImuFactor'daki duz-array yaklasimiyla
// ayni (test edilmis) desen.
template <int N>
struct GaussianPriorError {
    std::array<double, N*N> sqrtInfo; // row-major
    std::array<double, N> x0;         // linearizasyon noktasi (onceki pencerenin optimize edilmis degeri)

    GaussianPriorError(const Eigen::Matrix<double,N,N,Eigen::RowMajor>& S,
                        const Eigen::Matrix<double,N,1>& x0In) {
        Eigen::Map<Eigen::Matrix<double,N,N,Eigen::RowMajor>>(sqrtInfo.data()) = S;
        Eigen::Map<Eigen::Matrix<double,N,1>>(x0.data()) = x0In;
    }

    template <typename T>
    bool operator()(const T* const x, T* residual) const {
        T diff[N];
        for (int i = 0; i < N; ++i) diff[i] = x[i] - T(x0[i]);
        for (int r = 0; r < N; ++r) {
            residual[r] = T(0);
            for (int c = 0; c < N; ++c) residual[r] += T(sqrtInfo[r*N+c]) * diff[c];
        }
        return true;
    }

    static ceres::CostFunction* create(const Eigen::Matrix<double,N,N,Eigen::RowMajor>& S,
                                        const Eigen::Matrix<double,N,1>& x0In) {
        return new ceres::AutoDiffCostFunction<GaussianPriorError<N>, N, N>(
            new GaussianPriorError<N>(S, x0In));
    }
};
using PosePriorError = GaussianPriorError<6>;
using VelPriorError  = GaussianPriorError<3>;
