#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <optional>
#include <deque>
#include "MathUtils.hpp"
#include "Metrics.h"

// TrajectoryViewer icin tum ayarlanabilir parametreler
struct ViewerParams {
    // ---- Pencereler ----
    bool showFrame = true;              // anlik kamera goruntusunu ayri pencerede goster
    bool showTrajectory = true;         // yorunge grafigini ayri pencerede goster
    std::string frameWindowName = "Kamera Goruntusu";
    std::string trajWindowName  = "Yorunge (VO vs GT)";
    int waitKeyDelay = 1;                // ms - imshow'un anlik/non-blocking calismasi icin

    // ---- Yorunge cizimi ----
    int canvasSize = 800;                // yorunge penceresinin piksel boyutu (kare)
    double trajScale = 3.0;              // dunya birimini piksele cevirme olcegi (buyutulmus gorunum icin)
    cv::Scalar voColor = cv::Scalar(0, 255, 0);   // VO tahmini (yesil, BGR)
    cv::Scalar gtColor = cv::Scalar(0, 0, 255);   // ground truth (kirmizi, BGR)
    int pointRadius = 2;
    int lineThickness = 2;

    // ---- Metrikler ----
    bool showMetrics = true;             // FPS/ATE gibi metrikleri yorunge penceresine yaz
    bool computeLiveATE = true;          // umeyamaAlign + computeATE ile ATE'yi canli hesapla
    int ateUpdateInterval = 10;          // kac karede bir ATE yeniden hesaplansin (performans icin)
    int minPointsForATE = 3;             // umeyama icin gereken minimum nokta sayisi
};

class TrajectoryViewer {
public:
    explicit TrajectoryViewer(const ViewerParams& p = ViewerParams())
        : params(p) {
        canvas = cv::Mat::zeros(params.canvasSize, params.canvasSize, CV_8UC3);
        origin = cv::Point2d(params.canvasSize / 2.0, params.canvasSize / 2.0);
        lastTick = cv::getTickCount();
    }

    // Her karede TEK SATIRLA cagrilir. gtPos verilmezse sadece VO cizilir.
    // frame: o anki kamera goruntusu (gri ya da renkli olabilir)
    // voPos: o anki VO tahmini konum (X,Y,Z - dunya/mutlak birimde)
    // gtPos: varsa o anki ground truth konum (ayni birimde, senkronize edilmis)
    void update(const cv::Mat& frame, const cv::Point3d& voPos,
                const std::optional<cv::Point3d>& gtPos = std::nullopt) {

        // ---- FPS hesapla ----
        int64 now = cv::getTickCount();
        double dt = (now - lastTick) / cv::getTickFrequency();
        lastTick = now;
        double fps = (dt > 0) ? (1.0 / dt) : 0.0;

        // ---- Yorunge noktalarini biriktir ----
        voPath.push_back(voPos);
        if (gtPos) gtPath.push_back(*gtPos);
        frameCount++;

        // ---- Yeni segmenti canvas'a ciz (ustune ekleyerek, temizlemeden - hizli) ----
        drawNewSegment(voPath, prevVoPoint, params.voColor);
        if (gtPos) drawNewSegment(gtPath, prevGtPoint, params.gtColor);

        // ---- Canli ATE hesapla (belirli araliklarla, performans icin) ----
        if (params.computeLiveATE && !gtPath.empty() &&
            (int)gtPath.size() >= params.minPointsForATE &&
            frameCount % params.ateUpdateInterval == 0) {
            lastATE = computeLiveATE();
        }

        // ---- Metinlerle birlikte gosterilecek kopya olustur ----
        cv::Mat display = canvas.clone();
        if (params.showMetrics) {
            drawMetrics(display, fps);
        }

        if (params.showFrame && !frame.empty()) {
            cv::imshow(params.frameWindowName, frame);
        }
        if (params.showTrajectory) {
            cv::imshow(params.trajWindowName, display);
        }
        cv::waitKey(params.waitKeyDelay);
    }

private:
    ViewerParams params;
    cv::Mat canvas;
    cv::Point2d origin;
    int64 lastTick;
    int frameCount = 0;

    std::vector<cv::Point3d> voPath, gtPath;
    std::optional<cv::Point2d> prevVoPoint, prevGtPoint;
    double lastATE = -1.0;

    // 3B konumu (X,Z kullanarak - kustan bakis / top-down gorunum) canvas pikseline cevirir
    cv::Point2d toCanvas(const cv::Point3d& p) const {
        return cv::Point2d(origin.x + p.x * params.trajScale,
                            origin.y + p.z * params.trajScale);
    }

    void drawNewSegment(const std::vector<cv::Point3d>& path,
                         std::optional<cv::Point2d>& prevPoint,
                         const cv::Scalar& color) {
        if (path.empty()) return;
        cv::Point2d curr = toCanvas(path.back());
        if (prevPoint) {
            cv::line(canvas, *prevPoint, curr, color, params.lineThickness);
        }
        cv::circle(canvas, curr, params.pointRadius, color, cv::FILLED);
        prevPoint = curr;
    }

    double computeLiveATE() {
        size_t n = std::min(voPath.size(), gtPath.size());
        if (n < (size_t)params.minPointsForATE) return -1.0;

        Eigen::MatrixXd vo(3, n), gt(3, n);
        for (size_t i = 0; i < n; ++i) {
            vo(0,i) = voPath[i].x; vo(1,i) = voPath[i].y; vo(2,i) = voPath[i].z;
            gt(0,i) = gtPath[i].x; gt(1,i) = gtPath[i].y; gt(2,i) = gtPath[i].z;
        }

        Eigen::MatrixXd T = umeyamaAlign(vo, gt); // 4x4 donusum matrisi
        Eigen::MatrixXd voAligned(n, 3), gtMat(n, 3);
        for (size_t i = 0; i < n; ++i) {
            Eigen::Vector4d ph(vo(0,i), vo(1,i), vo(2,i), 1.0);
            Eigen::Vector4d pa = T * ph;
            voAligned.row(i) = pa.head<3>().transpose();
            gtMat.row(i) = Eigen::Vector3d(gt(0,i), gt(1,i), gt(2,i)).transpose();
        }
        return computeATE(voAligned, gtMat);
    }

    void drawMetrics(cv::Mat& display, double fps) {
        int y = 25;
        auto putLine = [&](const std::string& text) {
            cv::putText(display, text, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX,
                        0.6, cv::Scalar(255,255,255), 1, cv::LINE_AA);
            y += 25;
        };
        putLine("FPS: " + std::to_string((int)fps));
        putLine("Kare: " + std::to_string(frameCount));
        if (lastATE >= 0.0) {
            putLine("ATE: " + std::to_string(lastATE));
        }
        putLine("Yesil: VO   Kirmizi: GT");
    }
};
