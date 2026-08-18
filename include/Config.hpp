#pragma once
#include <nlohmann/json.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
struct Config {
    std::string algorithm, matcherType, imageFolder, gtCsvPath, imuCsvPath, translationMethod;
    double akazeThreshold, ransacThreshold, fx, fy, cx, cy;
    int nFrames, updateEvery;
    bool useImuFusion;
    cv::Mat K;

    static Config load(const std::string& path) {
        std::ifstream f(path);
        nlohmann::json j;
        f >> j;

        Config c;
        c.algorithm = j["algorithm"];
        c.matcherType = j["matcher_type"];
        c.akazeThreshold = j["akaze_threshold"];
        c.ransacThreshold = j["ransac_threshold"];
        c.nFrames = j["n_frames"];
        c.updateEvery = j["update_every"];
        c.imageFolder = j["image_folder"];
        c.gtCsvPath = j["gt_csv_path"];
        c.imuCsvPath = j["imu_csv_path"];
        c.translationMethod = j["translation_method"];
        c.useImuFusion = j["use_imu_fusion"];
        c.fx = j["fx"]; c.fy = j["fy"]; c.cx = j["cx"]; c.cy = j["cy"];

        c.K = (cv::Mat_<double>(3,3) << c.fx, 0, c.cx, 0, c.fy, c.cy, 0, 0, 1);
        return c;
    }
};
