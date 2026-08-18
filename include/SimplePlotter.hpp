#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
class SimplePlotter {
public:
    int size = 600;
    cv::Mat canvas;

    SimplePlotter() { canvas = cv::Mat::zeros(size, size, CV_8UC3); canvas.setTo(cv::Scalar(255,255,255)); }

    void updateTrajectory(const std::vector<cv::Point2d>& gt, const std::vector<cv::Point2d>& vo,
                           const std::string& title) {
        canvas.setTo(cv::Scalar(255,255,255));
        double minX=1e9,maxX=-1e9,minY=1e9,maxY=-1e9;
        for (auto& p : gt) { minX=std::min(minX,p.x); maxX=std::max(maxX,p.x); minY=std::min(minY,p.y); maxY=std::max(maxY,p.y); }
        for (auto& p : vo) { minX=std::min(minX,p.x); maxX=std::max(maxX,p.x); minY=std::min(minY,p.y); maxY=std::max(maxY,p.y); }
        double rangeX = std::max(1e-6, maxX-minX), rangeY = std::max(1e-6, maxY-minY);
        double margin = 40;

        auto toPixel = [&](cv::Point2d p) {
            double px = margin + (p.x - minX) / rangeX * (size - 2*margin);
            double py = size - (margin + (p.y - minY) / rangeY * (size - 2*margin));
            return cv::Point((int)px, (int)py);
        };

        for (size_t i = 1; i < gt.size(); ++i) cv::line(canvas, toPixel(gt[i-1]), toPixel(gt[i]), cv::Scalar(0,0,0), 2);
        for (size_t i = 1; i < vo.size(); ++i) cv::line(canvas, toPixel(vo[i-1]), toPixel(vo[i]), cv::Scalar(0,0,255), 1);

        cv::putText(canvas, title, cv::Point(10,20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);
        cv::imshow("VO vs GT", canvas);
    }
};
