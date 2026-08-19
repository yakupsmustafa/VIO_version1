#pragma once
#include "TimestampedDataSource.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>
#include <algorithm>
#include <iostream>

namespace fs = std::filesystem;

// CSV'yi header'a gore okuyan kucuk yardimci (csv.DictReader karsiligi)
inline std::vector<std::vector<std::string>> readCsvRows(const std::string& path,
                                                            std::vector<std::string>& headerOut) {
    std::ifstream f(path);
    std::string line;
    std::vector<std::vector<std::string>> rows;
    bool first = true;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        // Windows tarzi satir sonu (\r) temizligi
        if (!line.empty() && line.back() == '\r') line.pop_back();

        std::vector<std::string> cols;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) cols.push_back(cell);
        if (first) { headerOut = cols; first = false; }
        else rows.push_back(cols);
    }
    return rows;
}

inline int colIndex(const std::vector<std::string>& header, const std::string& name) {
    for (size_t i = 0; i < header.size(); ++i) if (header[i] == name) return (int)i;
    return -1;
}

class CSVGroundTruthSource : public TimestampedDataSource {
public:
    explicit CSVGroundTruthSource(const std::string& csvPath) {
        std::vector<std::string> header;
        auto rows = readCsvRows(csvPath, header);
        int ci_ts = colIndex(header, "timestamp");
        int ci_x = colIndex(header, "x");
        int ci_y = colIndex(header, "y");
        int ci_z = colIndex(header, "z");

        std::cerr << "[GT] header: ";
        for (auto& h : header) std::cerr << "[" << h << "] ";
        std::cerr << "\n[GT] ci_ts=" << ci_ts << " ci_x=" << ci_x << " ci_y=" << ci_y << " ci_z=" << ci_z << "\n";
        std::cerr << "[GT] toplam satir: " << rows.size() << "\n";

        if (ci_ts < 0 || ci_x < 0 || ci_y < 0 || ci_z < 0) {
            std::cerr << "[GT] HATA: gerekli sutunlardan biri bulunamadi!\n";
            throw std::runtime_error("GT CSV sutun hatasi");
        }

        for (size_t i = 0; i < rows.size(); ++i) {
            auto& r = rows[i];
            if ((int)r.size() <= std::max({ci_ts, ci_x, ci_y, ci_z})) {
                std::cerr << "[GT] HATA satir " << i << " (yetersiz sutun, " << r.size() << " sutun var): ";
                for (auto& c : r) std::cerr << "[" << c << "] ";
                std::cerr << "\n";
                throw std::runtime_error("GT CSV satir hatasi");
            }
            try {
                int64_t ts = std::stoll(r[ci_ts]);
                addSample(ts, {std::stod(r[ci_x]), std::stod(r[ci_y]), std::stod(r[ci_z])});
            } catch (const std::exception& e) {
                std::cerr << "[GT] HATA satir " << i << ": ";
                for (auto& c : r) std::cerr << "[" << c << "] ";
                std::cerr << "\n";
                throw;
            }
        }
    }
};

class CSVImuSource : public TimestampedDataSource {
public:
    explicit CSVImuSource(const std::string& csvPath) {
        std::vector<std::string> header;
        auto rows = readCsvRows(csvPath, header);
        int ci_ts = colIndex(header, "timestamp");
        int ci_ax = colIndex(header, "ax"), ci_ay = colIndex(header, "ay"), ci_az = colIndex(header, "az");
        int ci_gx = colIndex(header, "gx"), ci_gy = colIndex(header, "gy"), ci_gz = colIndex(header, "gz");

        std::cerr << "[IMU] header: ";
        for (auto& h : header) std::cerr << "[" << h << "] ";
        std::cerr << "\n[IMU] ci_ts=" << ci_ts << " ci_ax=" << ci_ax << " ci_ay=" << ci_ay << " ci_az=" << ci_az
                  << " ci_gx=" << ci_gx << " ci_gy=" << ci_gy << " ci_gz=" << ci_gz << "\n";
        std::cerr << "[IMU] toplam satir: " << rows.size() << "\n";

        if (ci_ts < 0 || ci_ax < 0 || ci_ay < 0 || ci_az < 0 || ci_gx < 0 || ci_gy < 0 || ci_gz < 0) {
            std::cerr << "[IMU] HATA: gerekli sutunlardan biri bulunamadi!\n";
            throw std::runtime_error("IMU CSV sutun hatasi");
        }

        for (size_t i = 0; i < rows.size(); ++i) {
            auto& r = rows[i];
            int maxCi = std::max({ci_ts, ci_ax, ci_ay, ci_az, ci_gx, ci_gy, ci_gz});
            if ((int)r.size() <= maxCi) {
                std::cerr << "[IMU] HATA satir " << i << " (yetersiz sutun, " << r.size() << " sutun var): ";
                for (auto& c : r) std::cerr << "[" << c << "] ";
                std::cerr << "\n";
                throw std::runtime_error("IMU CSV satir hatasi");
            }
            try {
                int64_t ts = std::stoll(r[ci_ts]);
                addSample(ts, {std::stod(r[ci_ax]), std::stod(r[ci_ay]), std::stod(r[ci_az]),
                                 std::stod(r[ci_gx]), std::stod(r[ci_gy]), std::stod(r[ci_gz])});
            } catch (const std::exception& e) {
                std::cerr << "[IMU] HATA satir " << i << ": ";
                for (auto& c : r) std::cerr << "[" << c << "] ";
                std::cerr << "\n";
                throw;
            }
        }
    }
};

class ImageLoader : public TimestampedDataSource {
public:
    std::vector<std::string> imageFiles;

    ImageLoader(const std::string& folderPath, int nFrames) {
        for (auto& entry : fs::directory_iterator(folderPath)) {
            if (entry.path().extension() == ".png") imageFiles.push_back(entry.path().string());
        }
        std::sort(imageFiles.begin(), imageFiles.end());
        if ((int)imageFiles.size() > nFrames) imageFiles.resize(nFrames);

        for (auto& f : imageFiles) {
            std::string stem = fs::path(f).stem().string();
            int64_t ts = std::stoll(stem);
            addSample(ts, {});
        }
    }
};
