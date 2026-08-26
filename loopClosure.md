# Loop Closure — Uygulama Planı

## Context

Bugünkü oturumda VIO sisteminin doğruluğunu artırmak için sırasıyla: Huber robust loss (başarılı, ATE 2.39m→1.40m), bias-marjinalizasyonu (denendi, deterministik testte kötüleşti, geri alındı), IMU ağırlıklarını gerçek sensör gürültüsünden türetme (denendi, katastrofik kötüleşti, geri alındı), ve yerçekimi kestirimini düzeltme (denendi, kötüleşti, geri alındı) test edildi. Dördü de "teorik olarak daha doğru" yaklaşımlardı; üçü ölçülünce başarısız çıktı. Ortak neden: IMU dead-reckoning zincirinde teşhis edilmiş ama düzeltilememiş **sistematik** bir sapma var (rastgele gürültü değil — saf IMU testinde hız neredeyse düz bir çizgi halinde büyüyor), ve bu sapmaya "daha çok güvenen" her değişiklik onu büyütüyor.

Test edilen 2000 karelik kesitte GT verisiyle **gerçek bir döngü doğrulandı**: kamera ~kare 870 civarında başlangıç noktasına 18cm mesafeye kadar geri dönüyor. Loop closure, bu döngüyü kapatarak kare 870-2000 arasında biriken sürüklenmeyi geriye doğru düzeltebilir — altta yatan sistematik sapmayı ÇÖZMESE de, global tutarlılığı ölçülebilir şekilde iyileştirebilir. Kullanıcı önceliği: FPS ve doğruluk, yöntem farketmez; önce loop closure, sonra FPS mimarisi (senkron→asenkron).

**Hedef:** Mevcut, çalışan pencere-bazlı VIO'ya dokunmadan (regresyon riski sıfıra yakın), üstüne bir loop-closure katmanı eklemek: keyframe veritabanı → yer tanıma (descriptor eşleştirme) → geometrik doğrulama (PnP, metrik ölçekli) → pose graph optimizasyonu → düzeltilmiş yörünge/ATE raporu.

## Tasarım Kararları (gerekçeleriyle)

1. **BoW/vocabulary YOK, doğrudan descriptor eşleştirme.** DBoW2/DBoW3 yeni bir bağımlılık gerektirir; bu ölçekte (muhtemelen ~100-150 keyframe) brute-force eşleştirme yeterli ve mevcut `FeatureMatcher` (pip02_Matcher.hpp) altyapısını doğrudan kullanır.

2. **Geometrik doğrulama: PnP, essential-matrix DEĞİL.** `pip03_PoseEstimator.hpp`'deki essential-matrix yaklaşımı sadece yön verir, gerçek ölçeği vermez (monoküler kısıtlaması). Bizim BA'dan çıkan landmark'lar zaten metrik ölçekte (IMU sayesinde) — aday keyframe'in 3D landmark'larına karşı mevcut karenin 2D noktalarıyla `cv::solvePnPRansac` çalıştırmak, ölçek belirsizliği olmadan doğrudan metrik bir poz verir. `pip03_PoseEstimator.hpp`'ye dokunulmayacak (kullanıcı isteğiyle - ileride kendisi silecek).

3. **Loop kısıtı = mevcut `PosePriorError` (GaussianPriorError<6>, Imufactor.hpp) yeniden kullanılır.** PnP zaten MUTLAK bir poz (dünya/kare-0 çerçevesinde) veriyor — bu, tam olarak `PosePriorError`'ın zaten yaptığı "bilinen kovaryanslı mutlak poz prior'u" kalıbı. Yeni bir residual yazmaya gerek yok.

4. **Ardışık-keyframe kenarları için TEK yeni residual: `RelativePoseError`.** Bu, gerçek pose-graph'ın omurgası — düzeltmenin zincir boyunca doğru şekilde "yayılmasını" sağlar (sadece her düğümü kendi eski pozuna geri çeken bağımsız prior'lar YETERSIZ kalır, düzeltme sadece döngü düğümü ile kare-0 arasında sıkışır, zincire yayılmaz). Matematiği ve yardımcı fonksiyonları (`mat3Mul`, `mat3Transpose`, `ceres::AngleAxisToRotationMatrix`/`RotationMatrixToAngleAxis`) **Imufactor.hpp'deki ImuFactor'ın rotasyon-hata deseninden birebir kopyalanır** — yeni/riskli matematik yok, kanıtlanmış kalıbın tekrarı.

5. **Düzeltme "offline" uygulanır (canlı VIO state'ine geri beslenmez).** Pose graph, TÜM keyframe pozlarını (döngüden önce VE sonra) birlikte optimize eder, düzeltmeyi zincire düzgünce yayar — bu, RAPORLANAN/ölçülen yörünge için yeterli. Düzeltmeyi canlı `anchor`/`currentPrior` state'ine geri besleyip devam eden pencerelerin onu kullanmasını sağlamak (tam "online" SLAM davranışı) çok daha riskli ve bugünkü tek-döngülü toplu test için ek fayda sağlamıyor — kapsam dışı bırakıldı.

6. **Canlı görselleştirme/ATE dokunulmaz.** Loop closure sonucu, çalışma bitiminde AYRI bir "loop-closure-düzeltilmiş ATE" olarak raporlanır (mevcut `computeATE`/`umeyamaAlign` ile). Bu, test edilmiş canlı pipeline'a sıfır risk demek.

## Uygulama Adımları (aşamalı, her aşama kendi içinde test edilir)

### Yeni dosya: `include/pip06_LoopClosure.hpp`
(mevcut `pip01`...`pip05` adlandırma kuralına uyar)

- `struct Keyframe { int frameIndex; std::array<double,6> pose; std::vector<cv::KeyPoint> keypoints; cv::Mat descriptors; std::vector<cv::Point3d> landmarks; std::vector<int> landmarkKptIdx; };`
- `class KeyframeDatabase` — `addKeyframe(...)` (mesafe-tabanlı seyreltme: son keyframe'den ~1m'den az hareket edilmişse eklenmez), `keyframes` vektörü.
- `findLoopCandidates(...)` — yeni keyframe'i, zaman/indeks olarak yeterince uzak (ör. >100 kare önce) tüm eski keyframe'lere karşı `FeatureMatcher` ile eşleştirir, eşleşme sayısı eşiği üstündekileri aday olarak döner.
- `verifyLoopPnP(...)` — adayın 3D landmark'ları + yeni karenin 2D noktalarıyla `cv::solvePnPRansac`, yeterli inlier varsa doğrulanmış (R,t) mutlak poz döner.
- `RelativePoseError` (yeni Ceres residual) — `Imufactor.hpp`'deki rotasyon-hata desenini kullanarak iki ardışık keyframe pozu arasındaki ölçülen göreli dönüşümü korur.
- `solvePoseGraph(...)` — tüm keyframe pozlarını parametre bloğu yapar, ardışık kenarlar için `RelativePoseError`, döngü düğümü için `PosePriorError` (Imufactor.hpp'den, PnP sonucunu x0 olarak), kare-0'ı `SetParameterBlockConstant` ile sabitler (VIO'nun kendi optimize() kuralıyla TUTARLI), Ceres ile çözer, düzeltilmiş pozları döner.

### `src/main.cpp` değişiklikleri
- Her başarılı pencere kapanışında (`ok == true` bloğu içinde), `KeyframeDatabase::addKeyframe` çağrısı eklenir (window'un son karesinin zaten hesaplanmış ORB keypoint/descriptor'ları + BA'dan çıkan landmark'lar kullanılır — YENİDEN HESAPLAMA yok).
- Her yeni keyframe eklendiğinde `findLoopCandidates` + `verifyLoopPnP` çağrılır; doğrulanmış bir döngü bulunursa loglanır (konsola: hangi kareler, kaç inlier).
- Çalışma sonunda, en az bir döngü doğrulandıysa `solvePoseGraph` çağrılır, düzeltilmiş keyframe yörüngesiyle `computeATE` tekrar hesaplanıp "loop-closure-düzeltilmiş ATE" olarak orijinal ATE'nin YANINDA raporlanır (ikisi de görünür kalır, karşılaştırma için).

## Aşamalı Doğrulama (her aşama bir öncekini kırmadan eklenir)

1. **Keyframe DB:** Sadece ekleme/sayım — kaç keyframe oluştu, pozisyonları GT'ye göre makul mü (log çıktısı, gözle kontrol).
2. **Aday tespiti (doğrulama YOK):** Adayların kare ~870 civarında çıkıp çıkmadığı, başka yerde YANLIŞ pozitif üretip üretmediği kontrol edilir.
3. **PnP doğrulama:** Doğrulanan döngünün inlier sayısı, kurtarılan pozun GT'ye göre makullüğü kontrol edilir.
4. **Pose graph:** ÖNCE scratch kopyada (bu oturumdaki yerleşik yöntem: `cv::setNumThreads(1)` + `ceres num_threads=1` ile deterministik, GUI kapalı, önce/sonra ATE karşılaştırması) test edilir. Sadece gerçekten iyileştiriyorsa gerçek repoya uygulanır ve gerekçesi kod içine yazılır — kötüleşirse (bugünün 4 denemesindeki gibi) dürüstçe raporlanıp geri alınır.

## Riskler / Notlar
- Ağırlıklar (odometri kenarı, loop prior'u) ilk denemede muhtemelen yanlış olacak — bu oturumun tekrar eden dersi. Ampirik olarak ayarlanacak, "teorik olarak doğru" diye körü körüne güvenilmeyecek.
- Tek bir döngü olayı var (kare ~870) — pose graph'ın tek-döngülü davranışı doğru test edilecek, çoklu-döngü senaryosu bu kapsamda değil.
- `pip03_PoseEstimator.hpp` ve `pip05_KltTracker.hpp`'ye dokunulmayacak.
