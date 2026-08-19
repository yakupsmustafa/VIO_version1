#pragma once
#include <opencv2/opencv.hpp>
#include "TimestampedDataSource.hpp"
#include <cmath>

// Iki kare (keyframe) arasindaki IMU olcumlerinin "ozetlenmis" hali.
// Forster et al. 2015 "On-Manifold Preintegration for VIO" makalesindeki
// ayrik (discrete/Euler) preintegration formulasyonu.
struct PreintegratedImuData {
    cv::Matx33d deltaR = cv::Matx33d::eye();  // preintegre edilmis rotasyon (govde_i -> govde_j)
    cv::Vec3d deltaV = cv::Vec3d(0,0,0);      // preintegre edilmis hiz degisimi
    cv::Vec3d deltaP = cv::Vec3d(0,0,0);      // preintegre edilmis konum degisimi
    double deltaT = 0.0;                      // toplam sure (saniye)

    // Bias'in hangi degerinde linearize edildigi (dogrusallastirma noktasi)
    cv::Vec3d biasGyroLin  = cv::Vec3d(0,0,0);
    cv::Vec3d biasAccelLin = cv::Vec3d(0,0,0);

    // Bias degisiminin delta'lara etkisini YAKLASIK (1. derece) veren Jacobianlar.
    // Gercek bias, linearizasyon noktasindan uzaklasirsa bu yaklasim bozulur
    // (gercek sistemler belirli bir esikten sonra "repropagation" yapar - burada atlandi).
    cv::Matx33d dR_dbg = cv::Matx33d::zeros();
    cv::Matx33d dV_dba = cv::Matx33d::zeros();
    cv::Matx33d dV_dbg = cv::Matx33d::zeros();
    cv::Matx33d dP_dba = cv::Matx33d::zeros();
    cv::Matx33d dP_dbg = cv::Matx33d::zeros();
};

inline cv::Matx33d skew(const cv::Vec3d& v) {
    return cv::Matx33d(0, -v[2], v[1],
                        v[2], 0, -v[0],
                        -v[1], v[0], 0);
}

// SO(3) sag Jacobian'i (Jr) - bias duzeltmesinin dogru olcekte uygulanmasi icin gerekli.
inline cv::Matx33d rightJacobianSO3(const cv::Vec3d& phi) {
    double angle = cv::norm(phi);
    if (angle < 1e-8) {
        return cv::Matx33d::eye() - 0.5 * skew(phi);
    }
    cv::Matx33d S = skew(phi);
    double a2 = angle * angle;
    return cv::Matx33d::eye()
         - ((1 - std::cos(angle)) / a2) * S
         + ((angle - std::sin(angle)) / (a2 * angle)) * (S * S);
}

inline cv::Matx33d expSO3(const cv::Vec3d& phi) {
    cv::Mat rvec = (cv::Mat_<double>(3,1) << phi[0], phi[1], phi[2]);
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    return cv::Matx33d(R);
}

class ImuPreintegrator {
public:
    const TimestampedDataSource& imu;
    explicit ImuPreintegrator(const TimestampedDataSource& imu_) : imu(imu_) {}

    // t0,t1: preintegrasyonun yapilacagi zaman araligi (nanosaniye)
    // biasGyro, biasAccel: o anki tahmini sensor bias'lari (linearizasyon noktasi)
    PreintegratedImuData preintegrate(int64_t t0, int64_t t1,
                                        const cv::Vec3d& biasGyro,
                                        const cv::Vec3d& biasAccel) {
        PreintegratedImuData result;
        result.biasGyroLin = biasGyro;
        result.biasAccelLin = biasAccel;

        auto lo = std::lower_bound(imu.timestamps.begin(), imu.timestamps.end(), t0);
        auto hi = std::upper_bound(imu.timestamps.begin(), imu.timestamps.end(), t1);
        size_t i0 = lo - imu.timestamps.begin();
        size_t i1 = hi - imu.timestamps.begin();
        if (i0 + 1 >= i1) return result; // yetersiz ornek

        cv::Matx33d R = cv::Matx33d::eye();
        cv::Vec3d v(0,0,0), p(0,0,0);
        cv::Matx33d JRbg = cv::Matx33d::zeros();
        cv::Matx33d JVba = cv::Matx33d::zeros(), JVbg = cv::Matx33d::zeros();
        cv::Matx33d JPba = cv::Matx33d::zeros(), JPbg = cv::Matx33d::zeros();
        double totalDt = 0;

        for (size_t i = i0; i + 1 < i1; ++i) {
            const auto& va = imu.values[i];
            double dt = (imu.timestamps[i+1] - imu.timestamps[i]) / 1e9;
            if (dt <= 0) continue;

            cv::Vec3d accelRaw(va[0], va[1], va[2]);
            cv::Vec3d gyroRaw(va[3], va[4], va[5]);
            cv::Vec3d accel = accelRaw - biasAccel; // bias duzeltilmis (linearizasyon noktasinda)
            cv::Vec3d gyro  = gyroRaw  - biasGyro;

            cv::Vec3d omega_dt = gyro * dt;
            cv::Matx33d dR = expSO3(omega_dt);
            cv::Matx33d Jr = rightJacobianSO3(omega_dt);

            // ---- Jacobian propagasyonu (Forster et al. eq. 16-18, ayrik/Euler versiyonu) ----
            JPba = JPba + JVba * dt - 0.5 * R * dt * dt;
            JPbg = JPbg + JVbg * dt - 0.5 * R * skew(accel) * JRbg * dt * dt;
            JVba = JVba - R * dt;
            JVbg = JVbg - R * skew(accel) * JRbg * dt;
            JRbg = dR.t() * JRbg - Jr * dt;

            // ---- Delta propagasyonu ----
            p = p + v * dt + 0.5 * (R * accel) * dt * dt;
            v = v + (R * accel) * dt;
            R = R * dR;

            totalDt += dt;
        }

        result.deltaR = R; result.deltaV = v; result.deltaP = p; result.deltaT = totalDt;
        result.dR_dbg = JRbg; result.dV_dba = JVba; result.dV_dbg = JVbg;
        result.dP_dba = JPba; result.dP_dbg = JPbg;
        return result;
    }
};
