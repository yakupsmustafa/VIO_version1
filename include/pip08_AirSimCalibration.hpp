#pragma once
#include <opencv2/opencv.hpp>
#include <cmath>

// ---- AirSim (Cosys-AirSim) settings.json'dan kamera kalibrasyonu turetir ----
// KAYNAK: https://github.com/ethz-asl/unreal_airsim/blob/master/docs/coordinate_systems.md +
// https://cosys-lab.github.io/Cosys-AirSim/settings/ (WebSearch/WebFetch ile dogrulandi, bkz. sohbet).
// ONEMLI KISIT: Bu makinede AirSim/Unreal CALISMIYOR - asagidaki formuller matematiksel olarak
// KENDI KENDINE tutarli ve fiziksel olarak mantikli olacak sekilde turetildi ve TEST EDILDI (bkz.
// dosya sonundaki sanity-check yorumlari), ama GERCEK AirSim ciktisina karsi DOGRULANAMADI - bu,
// arkadasinin makinesinde gercek veriyle yapilmasi gereken bir adim.
//
// ---- Cerceve kurallari ----
// Govde (body, NED): X=ileri(kuzey), Y=sag(dogu), Z=asagi.
// Kamera (standart CV, bu projenin GERI KALANIYLA AYNI): X=sag, Y=asagi, Z=ileri(optik eksen).
// AirSim'in Pitch/Roll/Yaw'i Euler-ZYX (once roll, sonra pitch, sonra yaw; govde eksenlerine gore) -
// ArduPilot/PX4 ile AYNI, standart havacilik kurali.

// ---- Kamera-govde HIZALAMA matrisi (Pitch=Roll=Yaw=0 iken, yani "kamera tam ileri bakiyor" durumu) ----
// Kamera-X(sag) -> govde-Y(sag), Kamera-Y(asagi) -> govde-Z(asagi), Kamera-Z(ileri) -> govde-X(ileri).
// Bu, EKSEN ISIMLERI farkli ama FIZIKSEL yon AYNI oldugu icin gereken sabit bir taban donusumudur -
// EuRoC'un R_BS'sinin (main.cpp buildCameraToBodyRotation) yaptigi isin AYNISI, farkli bir kaynaktan.
inline cv::Matx33d airsimCameraBaseAlignment() {
    return cv::Matx33d(
        0, 0, 1,
        1, 0, 0,
        0, 1, 0
    );
}

// ---- Rx/Ry/Rz (derece) - standart sag-el rotasyon matrisleri ----
inline cv::Matx33d rotX(double deg) {
    double r = deg * CV_PI / 180.0, c = std::cos(r), s = std::sin(r);
    return cv::Matx33d(1,0,0,  0,c,-s,  0,s,c);
}
inline cv::Matx33d rotY(double deg) {
    double r = deg * CV_PI / 180.0, c = std::cos(r), s = std::sin(r);
    return cv::Matx33d(c,0,s,  0,1,0,  -s,0,c);
}
inline cv::Matx33d rotZ(double deg) {
    double r = deg * CV_PI / 180.0, c = std::cos(r), s = std::sin(r);
    return cv::Matx33d(c,-s,0,  s,c,0,  0,0,1);
}

// ---- settings.json'daki Pitch/Roll/Yaw (derece) + taban hizalamadan, kamera->govde rotasyonunu
// (bu projenin R_BS/buildCameraToBodyRotation ile AYNI kullanim) hesaplar ----
// R_BC = Rz(yaw) * Ry(pitch) * Rx(roll) * R_base
inline cv::Matx33d airsimCameraToBodyRotation(double pitchDeg, double rollDeg, double yawDeg) {
    cv::Matx33d extra = rotZ(yawDeg) * rotY(pitchDeg) * rotX(rollDeg);
    return extra * airsimCameraBaseAlignment();
}

// ---- FOV_Degrees (AirSim'de YATAY FOV) + goruntu boyutundan kamera matrisini (K) hesaplar ----
// fx = (W/2) / tan(HFOV/2); kare piksel varsayimiyla fy=fx (AirSim'in render motorunun kendi
// dikey FOV'u en-boy oranindan turetmesiyle TUTARLI).
inline cv::Mat airsimCameraMatrix(int width, int height, double hfovDeg) {
    double fx = (width / 2.0) / std::tan(hfovDeg * CV_PI / 180.0 / 2.0);
    double fy = fx;
    double cx = width / 2.0;
    double cy = height / 2.0;
    return (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
}

// ---- Arkadasindan gelen settings.json'daki front_center/bottom_center icin HAZIR degerler ----
namespace AirSimMH {
    // front_center: X=0.5, Y=0, Z=0, Pitch=0, Roll=0, Yaw=0 (govdenin tam ileri yonune bakiyor)
    inline cv::Mat frontCenterK() { return airsimCameraMatrix(640, 480, 90.0); }
    inline cv::Matx33d frontCenterR_BC() { return airsimCameraToBodyRotation(0, 0, 0); }
    inline cv::Vec3d frontCenterOffsetBody() { return cv::Vec3d(0.5, 0, 0); } // govde cercevesinde (X ileri)

    // bottom_center: X=0, Y=0, Z=0.1, Pitch=-90, Roll=0, Yaw=0 (asagi bakiyor)
    inline cv::Mat bottomCenterK() { return airsimCameraMatrix(640, 480, 90.0); }
    inline cv::Matx33d bottomCenterR_BC() { return airsimCameraToBodyRotation(-90, 0, 0); }
    inline cv::Vec3d bottomCenterOffsetBody() { return cv::Vec3d(0, 0, 0.1); }
}

/*
DOGRULAMA (elle + kod ile, bkz. sohbet):
1) airsimCameraBaseAlignment() determinanti +1 olmali (gercek rotasyon, yansima DEGIL).
2) front_center (aci=0): R_BC'nin 3. sutunu (kameranin ileri/optik ekseni) govde-X (ileri) OLMALI.
3) bottom_center (Pitch=-90): R_BC'nin 3. sutunu govde-Z (asagi) OLMALI - "alt kamera asagi bakiyor"
   fiziksel beklentisiyle TUTARLI (elle dogrulandi: Ry(-90)*[1,0,0]^T = [0,0,1]^T = asagi).
NOT: fx/fy/cx/cy VE R_BC formulleri, GERCEK AirSim ciktisiyla henuz KARSILASTIRILMADI (bu makinede
AirSim yok) - ilk canli testte (arkadasinin makinesinde) MUTLAKA capraz kontrol edilmeli.
*/
