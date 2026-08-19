#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "ImuPreintegration.hpp"

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
struct ImuFactor {
    PreintegratedImuData preint;
    cv::Vec3d gravityWorld;
    double rotWeight, velWeight, posWeight, biasWeight;

    ImuFactor(const PreintegratedImuData& p, const cv::Vec3d& g,
              double rotW = 100.0, double velW = 10.0, double posW = 50.0, double biasW = 1.0)
        : preint(p), gravityWorld(g), rotWeight(rotW), velWeight(velW), posWeight(posW), biasWeight(biasW) {}

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
        T RiT[9], RjT[9];
        mat3Transpose(Ri, RiT); // aslinda column-major->row-major cevrimi de yapiyor
        mat3Transpose(Rj, RjT);

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

        // ---- Poz farklari (translation kismi, pose[3..5]) ----
        T pi[3] = { pose_i[3], pose_i[4], pose_i[5] };
        T pj[3] = { pose_j[3], pose_j[4], pose_j[5] };
        T g[3]  = { T(gravityWorld[0]), T(gravityWorld[1]), T(gravityWorld[2]) };

        // ---- Hiz residual: Ri^T * (v_j - v_i - g*dt) - dVcorrected ----
        T dv[3] = { vel_j[0]-vel_i[0]-g[0]*dt, vel_j[1]-vel_i[1]-g[1]*dt, vel_j[2]-vel_i[2]-g[2]*dt };
        T RiTdv[3]; mat3VecMul(RiT, dv, RiTdv);
        T velErr[3] = { RiTdv[0]-dVcorrected[0], RiTdv[1]-dVcorrected[1], RiTdv[2]-dVcorrected[2] };

        // ---- Konum residual: Ri^T * (p_j - p_i - v_i*dt - 0.5*g*dt^2) - dPcorrected ----
        T half_g_dt2[3] = { T(0.5)*g[0]*dt*dt, T(0.5)*g[1]*dt*dt, T(0.5)*g[2]*dt*dt };
        T dp[3] = { pj[0]-pi[0]-vel_i[0]*dt-half_g_dt2[0],
                    pj[1]-pi[1]-vel_i[1]*dt-half_g_dt2[1],
                    pj[2]-pi[2]-vel_i[2]*dt-half_g_dt2[2] };
        T RiTdp[3]; mat3VecMul(RiT, dp, RiTdp);
        T posErr[3] = { RiTdp[0]-dPcorrected[0], RiTdp[1]-dPcorrected[1], RiTdp[2]-dPcorrected[2] };

        // ---- Bias random-walk residual (bias'in kareler arasi cok degismemesi beklenir) ----
        T bgErr[3] = { bg_j[0]-bg_i[0], bg_j[1]-bg_i[1], bg_j[2]-bg_i[2] };
        T baErr[3] = { ba_j[0]-ba_i[0], ba_j[1]-ba_i[1], ba_j[2]-ba_i[2] };

        // ---- Ciktilari agirliklandirarak yaz ----
        for (int k = 0; k < 3; ++k) residual[k]      = T(rotWeight)  * rotErr[k];
        for (int k = 0; k < 3; ++k) residual[3+k]    = T(velWeight)  * velErr[k];
        for (int k = 0; k < 3; ++k) residual[6+k]    = T(posWeight)  * posErr[k];
        for (int k = 0; k < 3; ++k) residual[9+k]    = T(biasWeight) * bgErr[k];
        for (int k = 0; k < 3; ++k) residual[12+k]   = T(biasWeight) * baErr[k];

        return true;
    }

    static ceres::CostFunction* create(const PreintegratedImuData& p, const cv::Vec3d& g) {
        return new ceres::AutoDiffCostFunction<ImuFactor, 15, 6,3,3,3, 6,3,3,3>(
            new ImuFactor(p, g));
    }
};
