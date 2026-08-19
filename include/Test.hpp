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

    // ---- GT gorsel olcekleme ----
    // VO essential matrix'ten geldigi icin GERCEK metre olcegini bilmiyor (birim adimlarla ilerliyor),
    // GT ise gercek metre cinsinden. Bu yuzden ikisi ham haliyle CIZILDIGINDE GT gorunmez kalir.
    // autoScaleGT=true iken, GT SADECE CIZIM icin VO'nun kat ettigi mesafeyle orantili buyutulur.
    // NOT: Bu sadece gorsel bir duzeltmedir - ATE hesabi (asagida) HAM (olceksiz) GT kullanmaya devam eder,
    // cunku umeyamaAlign zaten kendi olcek duzeltmesini icinde yapiyor.
    bool autoScaleGT = true;
    double minGtPathLength = 1e-6; // sifira bolme koruma esigi
    double manualGtScale = 1.0; // autoScaleGT=false iken kullanilan SABIT olcek

    // ---- Eksen isareti duzeltmesi ----
    // GT ve VO farkli koordinat kurallarina (sag-el/sol-el, farkli eksen yonu) sahip olabilir.
    // Bu -1 yaparsan o eksen AYNALANIR (ters cevrilir). Hem cizime hem ATE hesabina uygulanir.
    cv::Vec3d gtAxisSign = cv::Vec3d(1, 1, 1); // ornek: X ters gorunuyorsa (-1, 1, 1) yap

    // ---- Otomatik yakinlastirma (auto-fit) ----
    // true iken trajScale/origin ELLE ayarlanmaz - her karede tum yorunge gecmisine
    // gore canvas'a TAM OTURACAK sekilde otomatik hesaplanir.
    bool autoFit = true;
    double fitMargin = 0.15; // kenarlarda birakilacak bosluk orani (0.15 = %15 pay)

    // ---- Yanal (X ekseni) abartma ----
    // 1.0 = normal, oranli. >1.0 = X eksenini Z'ye gore BAGIMSIZ buyutur (oran bozulur,
    // yanal hareketler gorsel olarak daha belirgin hale gelir).
    double lateralBoost = 1.0;

    // ---- VO'yu SADECE CIZIM icin dondurme ----
    // Derece cinsinden - X-Z duzleminde (ekrandaki gorunum) VO yorungesini dondurur.
    // Algoritmanin kendisini/ATE hesabini ETKILEMEZ, sadece gorsel.
    double voRotationDeg = 0.0;

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
    void update(const cv::Mat& frame, const cv::Point3d& voPos,
                const std::optional<cv::Point3d>& gtPos = std::nullopt) {

        // ---- FPS hesapla ----
        int64 now = cv::getTickCount();
        double dt = (now - lastTick) / cv::getTickFrequency();
        lastTick = now;
        double fps = (dt > 0) ? (1.0 / dt) : 0.0;

        // ---- GT'ye eksen isareti duzeltmesi uygula (ATE ve cizim icin TUTARLI olsun diye en basta) ----
        std::optional<cv::Point3d> gtPosCorrected;
        if (gtPos) {
            gtPosCorrected = cv::Point3d(gtPos->x * params.gtAxisSign[0],
                                          gtPos->y * params.gtAxisSign[1],
                                          gtPos->z * params.gtAxisSign[2]);
        }

        // ---- Yorunge noktalarini biriktir (HAM - ATE hesabinda kullanilir) ----
        voPath.push_back(voPos);
        if (gtPosCorrected) gtPath.push_back(*gtPosCorrected);
        frameCount++;

        // ---- Kat edilen toplam mesafeyi takip et (GT gorsel olcegi icin) ----
        if (hasPrevVoWorld) voPathLength += cv::norm(cv::Vec3d(voPos.x,voPos.y,voPos.z) - prevVoWorld);
        prevVoWorld = cv::Vec3d(voPos.x, voPos.y, voPos.z);
        hasPrevVoWorld = true;

        if (gtPosCorrected) {
            cv::Vec3d g(gtPosCorrected->x, gtPosCorrected->y, gtPosCorrected->z);
            if (hasPrevGtWorld) gtPathLength += cv::norm(g - prevGtWorld);
            prevGtWorld = g;
            hasPrevGtWorld = true;
        }

        // ---- GT gorsel olcegini guncelle ----
        if (params.autoScaleGT) {
            if (gtPathLength > params.minGtPathLength) gtVisualScale = voPathLength / gtPathLength;
        } else {
            gtVisualScale = params.manualGtScale; // elle ayarlanan sabit deger
        }

        if (params.autoFit) {
            // ---- Otomatik yakinlastirma: butun gecmisi, guncel olcekle YENIDEN ciz ----
            // Hizalanmis VO verisi varsa (Umeyama calisti) ONU kullan - GT ile ayni cerceve/yon/olcek.
            updateFitScale();
            redrawAll();
        } else {
            // ---- Eski davranis: sabit olcek, sadece yeni segmenti ekle (hizli ama sabit) ----
            drawNewSegment(rotateVo(voPos), prevVoCanvas, params.voColor);
            if (gtPosCorrected) {
                cv::Point3d scaledGt(gtPosCorrected->x * gtVisualScale,
                                      gtPosCorrected->y * gtVisualScale,
                                      gtPosCorrected->z * gtVisualScale);
                drawNewSegment(scaledGt, prevGtCanvas, params.gtColor);
            }
        }

        // ---- Canli ATE hesapla (HAM GT ile, olcek umeyama icinde zaten duzeltiliyor) ----
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

    std::vector<cv::Point3d> voPath, gtPath; // HAM veri, ATE icin
    std::optional<cv::Point2d> prevVoCanvas, prevGtCanvas;
    double lastATE = -1.0;

    // ---- Son hesaplanan Umeyama donusumu - HER KAREDE tum voPath'e yeniden uygulanir ----
    bool hasAlignment = false;
    Eigen::MatrixXd lastAlignT = Eigen::MatrixXd::Identity(4,4);

    // ---- GT gorsel olcekleme durumu ----
    double voPathLength = 0.0, gtPathLength = 0.0;
    cv::Vec3d prevVoWorld, prevGtWorld;
    bool hasPrevVoWorld = false, hasPrevGtWorld = false;
    double gtVisualScale = 1.0;

    // VO noktasini X-Z duzleminde params.voRotationDeg kadar dondurur (sadece cizim icin)
    cv::Point3d rotateVo(const cv::Point3d& p) const {
        if (params.voRotationDeg == 0.0) return p;
        double rad = params.voRotationDeg * CV_PI / 180.0;
        double c = std::cos(rad), s = std::sin(rad);
        double newX = p.x * c - p.z * s;
        double newZ = p.x * s + p.z * c;
        return cv::Point3d(newX, p.y, newZ);
    }

    cv::Point2d toCanvas(const cv::Point3d& p) const {
        return cv::Point2d(origin.x + p.x * params.trajScale * params.lateralBoost,
                            origin.y + p.z * params.trajScale);
    }

    // ---- Tum VO + olceklenmis GT noktalarinin (X,Z) sinirlarina gore
    //      trajScale ve origin'i canvas'a TAM oturacak sekilde hesaplar ----
    void updateFitScale() {
        double minX = 1e18, maxX = -1e18, minZ = 1e18, maxZ = -1e18;
        bool any = false;

        for (auto& raw : voPath) {
            cv::Point3d p = rotateVo(raw);
            minX = std::min(minX, p.x); maxX = std::max(maxX, p.x);
            minZ = std::min(minZ, p.z); maxZ = std::max(maxZ, p.z);
            any = true;
        }
        for (auto& p : gtPath) {
            double gx = p.x * gtVisualScale, gz = p.z * gtVisualScale;
            minX = std::min(minX, gx); maxX = std::max(maxX, gx);
            minZ = std::min(minZ, gz); maxZ = std::max(maxZ, gz);
            any = true;
        }
        if (!any) return;

        // X araligi, lateralBoost UYGULANMIS haliyle hesaba katilir - boylece abartilmis
        // X yine de canvas sinirlarina sigar (tasip kesilmez).
        double rangeX = std::max((maxX - minX) * params.lateralBoost, 1e-6);
        double rangeZ = std::max(maxZ - minZ, 1e-6);
        double usablePixels = params.canvasSize * (1.0 - 2.0 * params.fitMargin);

        params.trajScale = usablePixels / std::max(rangeX, rangeZ);

        double centerX = (minX + maxX) / 2.0;
        double centerZ = (minZ + maxZ) / 2.0;
        origin = cv::Point2d(params.canvasSize / 2.0 - centerX * params.trajScale * params.lateralBoost,
                              params.canvasSize / 2.0 - centerZ * params.trajScale);
    }

    // ---- Canvas'i temizleyip tum yorunge gecmisini GUNCEL olcek/origin ile yeniden cizer ----
    void redrawAll() {
        canvas = cv::Mat::zeros(params.canvasSize, params.canvasSize, CV_8UC3);

        // VO, params.voRotationDeg kadar dondurulerek cizilir (sadece gorsel)
        for (size_t i = 0; i < voPath.size(); ++i) {
            cv::Point2d curr = toCanvas(rotateVo(voPath[i]));
            if (i > 0) cv::line(canvas, toCanvas(rotateVo(voPath[i-1])), curr, params.voColor, params.lineThickness);
            cv::circle(canvas, curr, params.pointRadius, params.voColor, cv::FILLED);
        }
        // GT gorsel olcek ile buyutulerek cizilir (gtVisualScale)
        for (size_t i = 0; i < gtPath.size(); ++i) {
            cv::Point3d sp(gtPath[i].x * gtVisualScale, gtPath[i].y * gtVisualScale, gtPath[i].z * gtVisualScale);
            cv::Point2d curr = toCanvas(sp);
            if (i > 0) {
                cv::Point3d spPrev(gtPath[i-1].x * gtVisualScale, gtPath[i-1].y * gtVisualScale, gtPath[i-1].z * gtVisualScale);
                cv::line(canvas, toCanvas(spPrev), curr, params.gtColor, params.lineThickness);
            }
            cv::circle(canvas, curr, params.pointRadius, params.gtColor, cv::FILLED);
        }
    }

    void drawNewSegment(const cv::Point3d& worldPoint,
                         std::optional<cv::Point2d>& prevPoint,
                         const cv::Scalar& color) {
        cv::Point2d curr = toCanvas(worldPoint);
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

        Eigen::MatrixXd T = umeyamaAlign(vo, gt); // 4x4 donusum matrisi (rotasyon+olcek+oteleme)
        Eigen::MatrixXd voAligned(n, 3), gtMat(n, 3);
        for (size_t i = 0; i < n; ++i) {
            Eigen::Vector4d ph(vo(0,i), vo(1,i), vo(2,i), 1.0);
            Eigen::Vector4d pa = T * ph;
            voAligned.row(i) = pa.head<3>().transpose();
            gtMat.row(i) = Eigen::Vector3d(gt(0,i), gt(1,i), gt(2,i)).transpose();
        }

        // ---- Donusumu SAKLA - her karede TUM voPath'e yeniden uygulanacak (draw icin) ----
        lastAlignT = T;
        hasAlignment = true;

        return computeATE(voAligned, gtMat);
    }

    // ---- Saklanan son donusumu, GUNCEL tum voPath'e uygular (her karede cagrilir, ucuz islem) ----
    std::vector<cv::Point3d> alignedVoForDraw() const {
        if (!hasAlignment) return voPath; // henuz ilk hizalama olmadi, ham goster
        std::vector<cv::Point3d> out(voPath.size());
        for (size_t i = 0; i < voPath.size(); ++i) {
            Eigen::Vector4d ph(voPath[i].x, voPath[i].y, voPath[i].z, 1.0);
            Eigen::Vector4d pa = lastAlignT * ph;
            out[i] = cv::Point3d(pa[0], pa[1], pa[2]);
        }
        return out;
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
        putLine("GT gorsel olcek: " + std::to_string(gtVisualScale));
        putLine("Yesil: VO   Kirmizi: GT (olceklendi)");
    }
};
