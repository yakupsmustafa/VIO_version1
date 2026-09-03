#pragma once
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>
#include <cstdint>

// AirLib basliklari - SADECE AIRSIM_ROOT tanimliysa derlenir (bkz. CMakeLists.txt, vio_airsim hedefi).
// Bu makinede AirLib YOK - bu dosya burada DERLENEMEZ/TEST EDILEMEZ (bkz. plan, "Kritik kisit").
#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"

// ---- AirSim'den (Cosys-AirSim, ArduCopter/MAVLink SITL kontrolunde) SADECE SENSOR OKUYAN kopru ----
// KAYNAK (WebFetch ile dogrulanan gercek AirLib kaynak kodu, bkz. sohbet):
//   RpcLibClientBase::getImuData(imu_name, vehicle_name) -> msr::airlib::ImuBase::Output
//     (alanlar: linear_acceleration, angular_velocity, orientation, time_stamp)
//   RpcLibClientBase::simGetImages(vector<ImageRequest>, vehicle_name) -> vector<ImageResponse>
//     (ImageRequest: camera_name, image_type, pixels_as_float=false, compress=false, annotation_name)
//     (ImageResponse: image_data_uint8, image_data_float, width, height, time_stamp, ...)
//   MultirotorRpcLibClient::getMultirotorState(vehicle_name) -> MultirotorState
//     (kinematics_estimated.pose.position uzerinden getPosition() -> Vector3r, dunya/NED cercevesinde)
// NOT: enableApiControl/armDisarm KASITLI OLARAK CAGRILMIYOR - settings.json'daki arac tipi "ArduCopter"
// (bkz. pip08_AirSimCalibration.hpp dosya basi), yani ucus MAVLink/ArduPilot SITL tarafindan kontrol
// ediliyor. Bu kopru SADECE SENSOR OKUYOR, ucusa hicbir sekilde KARISMIYOR.

struct AirSimImuSample {
    double ax = 0, ay = 0, az = 0; // m/s^2, govde (body/NED) cercevesinde
    double gx = 0, gy = 0, gz = 0; // rad/s, govde cercevesinde
    int64_t timestampNs = 0;
};

struct AirSimImageSample {
    cv::Mat gray;          // CV_8UC1
    int64_t timestampNs = 0;
    bool valid = false;
};

class AirSimBridge {
public:
    // host: arkadasinin makinesinde AirSim/Unreal calisiyorsa "127.0.0.1"; bu makineden UZAKTAN
    // baglanmak icin arkadasinin IP adresi (ayni ag/VPN'de olmasi + AirSim RPC portunun -varsayilan
    // 41451- gelen baglantilara acik olmasi gerekir - bu, arkadasinin makinesinde dogrulanmali).
    void connect(const std::string& host = "127.0.0.1", const std::string& vehicleName = "") {
        vehicleName_ = vehicleName;
        client_ = std::make_unique<msr::airlib::MultirotorRpcLibClient>(host);
        client_->confirmConnection();
    }

    AirSimImuSample pollImu(const std::string& imuName = "") {
        msr::airlib::ImuBase::Output out = client_->getImuData(imuName, vehicleName_);
        AirSimImuSample s;
        s.ax = out.linear_acceleration.x();
        s.ay = out.linear_acceleration.y();
        s.az = out.linear_acceleration.z();
        s.gx = out.angular_velocity.x();
        s.gy = out.angular_velocity.y();
        s.gz = out.angular_velocity.z();
        s.timestampNs = (int64_t)out.time_stamp;
        return s;
    }

    // cameraName: settings.json'daki kamera adi ("front_center" ya da "bottom_center", bkz. pip08).
    // pixels_as_float=false, compress=false ISTENIYOR - PNG/JPEG cozmekle ugrasmadan dogrudan ham
    // piksel arabellegi almak icin (RPC turu icin de daha az CPU maliyeti).
    AirSimImageSample pollImage(const std::string& cameraName) {
        using ICB = msr::airlib::ImageCaptureBase;
        std::vector<ICB::ImageRequest> req = {
            ICB::ImageRequest(cameraName, ICB::ImageType::Scene, /*pixels_as_float*/false, /*compress*/false)
        };
        auto resp = client_->simGetImages(req, vehicleName_);
        AirSimImageSample out;
        if (resp.empty() || resp[0].image_data_uint8.empty()) return out;

        const auto& r = resp[0];
        size_t n = r.image_data_uint8.size();
        size_t px = (size_t)r.width * (size_t)r.height;
        if (px == 0) return out;

        // NOT: compress=false iken ham arabellegin tam kanal duzeni (BGRA/BGR/gri) AirSim surumune
        // gore degisebilir - bu yuzden BOYUTA gore ayirt ediliyor, sabit bir varsayim YAPILMIYOR.
        // Ilk canli testte (arkadasinin makinesinde) hangi dala girdigi MUTLAKA loglanip dogrulanmali.
        cv::Mat img;
        if (n == px * 4) {
            cv::Mat bgra(r.height, r.width, CV_8UC4, (void*)r.image_data_uint8.data());
            cv::cvtColor(bgra, img, cv::COLOR_BGRA2GRAY);
        } else if (n == px * 3) {
            cv::Mat bgr(r.height, r.width, CV_8UC3, (void*)r.image_data_uint8.data());
            cv::cvtColor(bgr, img, cv::COLOR_BGR2GRAY);
        } else if (n == px) {
            img = cv::Mat(r.height, r.width, CV_8UC1, (void*)r.image_data_uint8.data()).clone();
        } else {
            return out; // beklenmeyen boyut - islenmiyor, cagiran taraf out.valid=false gorur
        }

        out.gray = img.clone();
        out.timestampNs = (int64_t)r.time_stamp;
        out.valid = true;
        return out;
    }

    // Dunya (NED) cercevesinde araç govde konumu - EuRoC'un ground_truth.csv'siyle AYNI anlamda
    // (govde-govde kiyaslamasi, main.cpp'deki alignedGt[i] kullanimiyla BIREBIR ayni).
    cv::Point3d pollGroundTruthPosition() {
        auto state = client_->getMultirotorState(vehicleName_);
        auto p = state.getPosition();
        return cv::Point3d(p.x(), p.y(), p.z());
    }

private:
    std::unique_ptr<msr::airlib::MultirotorRpcLibClient> client_;
    std::string vehicleName_;
};
