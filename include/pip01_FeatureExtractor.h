#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

// ORB icin tum ayarlanabilir parametreleri tutan yapi
struct ORBParams {
    int nfeatures = 1000;
    float scaleFactor = 1.2f;
    int nlevels = 8;
    int edgeThreshold = 31;
    int firstLevel = 0;
    int WTA_K = 2;
    cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
    int patchSize = 31;
    int fastThreshold = 20;
};

class ORBExtractor {
public:
    cv::Ptr<cv::ORB> detector;

    explicit ORBExtractor(const ORBParams& p = ORBParams()) {
        detector = cv::ORB::create(
            p.nfeatures,
            p.scaleFactor,
            p.nlevels,
            p.edgeThreshold,
            p.firstLevel,
            p.WTA_K,
            p.scoreType,
            p.patchSize,
            p.fastThreshold
        );
    }
    void detect(const cv::Mat& grayImage, std::vector<cv::KeyPoint>& kps, cv::Mat& descs) {
        detector->detectAndCompute(grayImage, cv::noArray(), kps, descs);
    }
};


/* 
ORB parametrelerinin tanımı:
- nfeatures: Algılanacak maksimum köşe sayısı.
- scaleFactor: Her seviye arasındaki ölçek faktörü. Örneğin, 1.2, bir sonraki seviyedeki görüntünün önceki seviyeye göre %20 daha küçük olacağı anlamına gelir.
- nlevels: Görüntü piramidindeki seviye sayısı. Daha fazla seviye, daha fazla ölçeklenmiş özellikler anlamına gelir, ancak işlem süresi artar.
- edgeThreshold: Kenar eşiği, köşe algılama sırasında kenarlara yakın bölgeleri filtrelemek için kullanılır. Daha yüksek bir değer, kenarlara daha yakın köşe algılamayı engeller.
- firstLevel: Görüntü piramidindeki ilk seviyeyi belirler. Genellikle 0 olarak ayarlanır.
- WTA_K: Hamming ağacı için kullanılan bit sayısı. 2 veya 3 olabilir. Daha yüksek bir değer, daha fazla bilgi sağlar ancak işlem süresini artırabilir.
- scoreType: Köşe puanlama yöntemi. cv::ORB::HARRIS_SCORE veya cv::ORB::FAST_SCORE olabilir.  -HARRIS_SCORE, Harris köşe algılama yöntemini kullanır ve genellikle daha iyi sonuçlar verir.
- patchSize: Köşe algılama sırasında kullanılan yama boyutu.
- fastThreshold: FAST köşe algılama algoritması için eşik değeri. Daha düşük bir değer, daha fazla köşe algılanmasına neden olur, ancak yanlış pozitifleri artırabilir

*/