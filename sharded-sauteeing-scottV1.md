# Adım 2: Tam VIO Entegrasyonu (Ceres Sliding-Window BA)

## Context

Proje şu an sadece saf monoküler VO çalıştırıyor: `Imupreintegration.hpp`, `Imufactor.hpp`,
`Viobundleadjustment.hpp` dosyaları yazılmış ama `main.cpp`'den hiç çağrılmıyor. Essential-matrix'ten
gelen `t` ölçeksiz olduğundan (Adım 1'de düzeltilen kamera-merkezi hatasından bağımsız olarak) yörünge
gerçek metrik ölçekte değil. Kullanıcı, hafif bir ölçek-düzeltme hilesi yerine **gerçek bir sliding-window
VIO** istedi: IMU preintegration + Ceres BA ile pencere içindeki kareleri birlikte optimize etmek.

Tasarımı çıkarırken mevcut `Imufactor.hpp` içinde, konu bu işe koyulmadan önce düzeltilmesi gereken
**iki bağımsız matematiksel tutarsızlık** bulundu (aşağıda "Bulunan ve Düzeltilecek Hatalar" bölümünde
detaylandırıldı). Bunlar düzeltilmeden pencere BA'sı ya diverge eder ya da anlamsız/rastgele sonuçlar
üretir — bu yüzden "sadece mevcut parçaları birbirine bağlamak" yeterli değil, `Imufactor.hpp`'ye ve
IMU verisinin işlenme şekline hedefli, gerekçeli müdahaleler de gerekiyor.

## Bulunan ve Düzeltilecek Hatalar (Imufactor.hpp + main.cpp'nin IMU'yu okuma şekli)

### Hata A — Konum residual'ı `t_wc`'yi kamera merkezi sanıyor
`Imufactor.hpp:99-115`, konum residual'ında `pose_i[3..5]`/`pose_j[3..5]`'i (yani `ReprojectionError`'daki
ile aynı ham world→camera extrinsic öteleme `t_wc`) doğrudan `pi`/`pj` olarak kullanıp `pj-pi` farkını alıyor.
Ama IMU fiziksel olarak kameranın **gerçek dünya konumunu** (`C = -R_wc^T · t_wc`, Adım 1'de düzeltilen
nicelik) integre ediyor; `t_wc` değil. `t = -R·C` olduğundan `t_j - t_i ≠ R_i·(C_j - C_i)` (R_i≠R_j olduğu
sürece) — somut sayılarla doğrulandı. **Fix:** residual hesaplanmadan önce `Ci = -R_i^T·t_i`,
`Cj = -R_j^T·t_j` hesaplanıp `pj-pi` yerine `Cj-Ci` kullanılacak.

(Not: Dosyadaki `RiT` değişkeni adına rağmen sayısal olarak `R_i^T` DEĞİL `R_i`'nin kendisidir — ceres'in
column-major çıktısını düzeltmenin yan etkisi. Rotasyon ve hız residual'ları bu konvansiyonla hesaplandığında
zaten doğru sonucu veriyor (world→camera parametrizasyonuyla tutarlı standart Forster formülasyonuna denk
geliyor), o yüzden onlara DOKUNULMAYACAK — sadece konum residual'ı, orada `t` ile `C` karıştırıldığı için
düzeltilecek. Kafa karıştırıcı isimlendirme yorum ile belgelenecek, değişken adı değiştirilmeyecek (gereksiz
risk).)

### Hata B — IMU ölçümleri kamera çerçevesine hiç döndürülmüyor
`ImuPreintegrator`, ham `gx,gy,gz,ax,ay,az`'yi CSV'den geldiği gibi (IMU/gövde çerçevesinde) integre ediyor.
Ama `ImuFactor`'ün optimize ettiği `pose`, `ReprojectionError` ile aynı **kamera** çerçevesi. `R_BS`
(kameradan gövdeye rotasyon, `main.cpp:buildCameraToBodyRotation()`) kimlik matrisinden çok uzak (~90°'lik
gerçek bir rotasyon) — yani şu an IMU'nun "yukarısı" ile kameranın "yukarısı" birbirine karışıyor. **Fix:**
`ImuPreintegrator`'a verilmeden önce ham IMU örnekleri `R_CB = R_BS^T` ile kamera çerçevesine döndürülecek
(hem `ax,ay,az` hem `gx,gy,gz` üçlüleri, sabit bir rotasyon olduğu için standart vektör rotasyonu ile).
Bu, `Imupreintegration.hpp`/`Imufactor.hpp` matematiğine dokunmadan, sadece veri girişini düzelterek çözülüyor.

### Statik Yerçekimi Kalibrasyonu (kullanıcı onayladı)
`gravityWorld` sabit `(0,0,9.81)` yerine, sekans başındaki ilk N örnek (kamera-çerçevesine döndürülmüş
ivmeölçer) ortalanarak kestirilecek: `gravity = -normalize(mean(accel[0..N])) * 9.81`. Bu, kare-0'ın
gerçek dünyaya göre nasıl durduğunu bilmeden `(0,0,9.81)` varsaymanın getirdiği sistematik hatayı ortadan
kaldırır.

## Tasarım

### 1. Feature track oluşturma
Pencere içindeki ardışık kare çiftleri zaten `matcher.match()` ile eşleştiriliyor (poz tahmini için de
gerekli). Her çift için `matchMap[s]: queryIdx→trainIdx` kurulur. Kare-0'daki her keypoint için pencere
boyunca zincir takip edilir; zincir bir yerde kopan (eşleşme bulunamayan) track tamamen atılır. Sonuçta
sadece **pencerenin tamamında hayatta kalan** track'ler kalır — bu, `VIOBundleAdjustment::optimize`'ın
gerektirdiği "full coverage" kısıtını inşa yoluyla otomatik sağlar.

### 2. Landmark başlangıç tahmini
Pencerenin ilk ve son karesinin **IMU-seed'lenmiş** pozlarından (bkz. madde 4) `cv::triangulatePoints` ile
3B nokta tahmini çıkarılır (VO zincirinin ölçeksiz pozları yerine — IMU seed zaten gerçek metrik ölçekte).
Sonrasında: homojen bölme, sonlu-olmayan noktaları at, her iki kamera önünde olmayanları at (cheirality),
düşen track'leri `tracks`'ten de eşzamanlı çıkar.

### 3. Pencere gruplama stratejisi
`windowSize = 5` (mevcut varsayılan) korunuyor. Pencereler **1 kare çakışmalı**: pencere k, [a, a+4]
karelerini kapsar; pencere k+1, [a+4, a+8]'i kapsar. Ortak kare (a+4), yeni pencerenin 0. karesi olur —
`VIOBundleAdjustment::optimize`'ın zaten sabitlediği "kare 0" mekanizmasını, önceki pencerenin
BA-düzeltilmiş sonucunu bir sonrakine taşıyan doğal bir çapa olarak kullanır. `optimize()` `false` dönerse
veya sonuç sonlu değilse, çapa BA sonucunu almaz — bir önceki (BA öncesi, saf IMU dead-reckoning) seed
korunur; böylece kötü bir çözüm zincire yayılmaz.

### 4. Frame state başlatma (düzeltilmiş matematikle)
Yeni `seedWindowStatesFromImu(imuPreintCam, anchor, frameTimestamps, gravityCam)` fonksiyonu, `world→camera`
parametrizasyonu için doğru türetilmiş şu tekrarlama bağıntısını kullanır (Forster formülasyonunun
`R_wc = R_wb^T` konvansiyonuna uyarlanmış hali, `ImuFactor`'ün residual'larından ters çözülerek elde edildi):

```
ΔR, ΔV, ΔP = imuPreintCam.preintegrate(t[j-1], t[j], bg[j-1], ba[j-1])   // artik KAMERA cercevesindeki IMU'dan

R[j] = ΔR^T · R[j-1]                                    // R_init[j-1]*ΔR DEGIL — ters!
v[j] = v[j-1] + gravityCam·Δt + R[j-1]^T · ΔV
C[j] = C[j-1] + v[j-1]·Δt + 0.5·gravityCam·Δt² + R[j-1]^T · ΔP     // C = fiziksel kamera merkezi
t[j] = -R[j] · C[j]                                     // FrameState.pose[3..5] icin geri cevir
ba[j] = ba[j-1], bg[j] = bg[j-1]                          // BA tarafindan iyilestirilecek
```

İlk global çapa (hiç pencere optimize edilmemişken): `pose={0,0,0,0,0,0}`, `vel=ba=bg={0,0,0}` — Adım 1
öncesindeki `chainedR=I, chainedT=0` başlangıcıyla birebir aynı, davranış kare-0'da değişmiyor.

### 5. Canlı görselleştirme entegrasyonu
`viewer.update()` her karede sırayla çağrılmaya devam ediyor (API değişmiyor). Değişen: hangi pozun
beslendiği.
- Pencere dolarken: o ana kadar biriken karelerin `seedWindowStatesFromImu` sonucuyla
  `cameraCenterFromPose(states.back().pose)` (Adım 1'deki `-R^T·t` formülüyle, `R_BS` ile gövde çerçevesine
  çevrilerek) çizilir — bugünkü ölçeksiz VO zinciri yerine, zaten hesaplanan gerçek-ölçekli IMU seed'i
  kullanılır.
- Pencere tamamlanıp `optimize()` `true` + sonlu sonuç döndürdüğünde: çapa (pose/vel/ba/bg), o pencerenin
  BA-optimize edilmiş son karesine sıfırlanır. Bu, `TrajectoryViewer`'ın `voPath`'i (sadece `push_back`)
  geçmişi geriye dönük boyayamadığından, **sadece bundan sonra çizilecek kareleri** etkiler — pencere
  sınırlarında (~4 karede bir) küçük konum "sıçramaları" beklenir, bu normal/beklenen davranıştır, hata
  değildir.
- `optimize()` başarısız olursa: çapa BA-öncesi seed'de kalır, konsola net bir uyarı basılır.

### 6. IMU verisinin kamera çerçevesine döndürülmesi + statik yerçekimi
Yeni yardımcılar (bkz. Değişen/Yeni Dosyalar):
- `TimestampedDataSource rotateImuToCameraFrame(const TimestampedDataSource& rawImu, const cv::Mat& R_BS)`
  — her örnekteki `{ax,ay,az}` ve `{gx,gy,gz}` üçlülerini `R_CB = R_BS^T` ile döndürüp yeni bir
  `TimestampedDataSource` döner (timestamp'ler aynen kopyalanır).
- `cv::Vec3d estimateGravityCameraFrame(const TimestampedDataSource& rotatedImu, int nStaticSamples=200)`
  — ilk `nStaticSamples` (döndürülmüş) ivme örneğinin ortalamasını alıp `-normalize(mean)*9.81` döner.

`main()`'de: `imu` yüklendikten hemen sonra `auto imuCam = rotateImuToCameraFrame(imu, R_BS);`,
`ImuPreintegrator imuPreint(imuCam);`, `cv::Vec3d gravity = estimateGravityCameraFrame(imuCam);`,
`VIOBundleAdjustment vio(K, imuPreint, gravity, 5);`. Yaşam süresi: `imu`/`imuCam`/`imuPreint`/`vio` hepsi
`main()` scope'unda, `imu`→`imuCam`→`imuPreint`→`vio` sırasıyla inşa edilir (referans güvenliği için sıra
önemli).

### 7. Imufactor.hpp değişikliği (izole, küçük)
`operator()` içinde, mevcut `RiT`/`RjT` hesaplandıktan hemen sonra gerçek transpozlar ve kamera merkezleri
eklenir:
```cpp
T RiTrue_T[9], RjTrue_T[9];
mat3Transpose(RiT, RiTrue_T);   // gercek R_i^T (RiT zaten R_i'nin kendisiydi)
mat3Transpose(RjT, RjTrue_T);   // gercek R_j^T
T Ci[3], Cj[3];
{ T negTi[3] = {-pose_i[3],-pose_i[4],-pose_i[5]}; mat3VecMul(RiTrue_T, negTi, Ci); }
{ T negTj[3] = {-pose_j[3],-pose_j[4],-pose_j[5]}; mat3VecMul(RjTrue_T, negTj, Cj); }
```
ve konum residual'ındaki `dp[...]` ifadesinde `pj[k]-pi[k]` yerine `Cj[k]-Ci[k]` kullanılır. Rotasyon
(`rotErr`) ve hız (`velErr`) residual'ları **değişmiyor** (doğrulukları yukarıda gerekçelendirildi).

## Değişen / Yeni Dosyalar

- **`include/pip04_VioWindow.hpp`** (yeni, mevcut `pipNN_` adlandırma kuralına uyar):
  - `rotateImuToCameraFrame`, `estimateGravityCameraFrame` (madde 6)
  - `cv::Point3d cameraCenterFromPose(const std::array<double,6>& pose)` — `C=-R^T t` (Adım 1 formülü)
  - `cv::Mat projectionMatrixFromPose(const cv::Mat& K, const std::array<double,6>& pose)` — `K·[R|t]`
  - `std::vector<std::vector<int>> buildFullCoverageTracks(...)` (madde 1)
  - `void triangulateAndFilterWindowTracks(...)` (madde 2)
  - `std::vector<VIOBundleAdjustment::FrameState> seedWindowStatesFromImu(...)` (madde 4, düzeltilmiş formül)
- **`include/Imufactor.hpp`** — madde 7'deki izole `Ci/Cj` ekleme + `dp` düzeltmesi.
- **`src/main.cpp`** — `#include "pip04_VioWindow.hpp"`, `"Imupreintegration.hpp"`, `"Viobundleadjustment.hpp"`
  eklenir; `main()` içine madde 6'daki kurulum eklenir; `processFrames` pencereli hale getirilir (madde 3-5).
- **Değişmeyecek:** `Imupreintegration.hpp`, `Viobundleadjustment.hpp`, `DataLoader.h`, `pip02_Matcher.hpp`,
  `pip03_PoseEstimator.hpp`, `Test.hpp`, `CMakeLists.txt`.

## Doğrulama

1. `cmake --build build -j` — sıfır yeni bağımlılık, mevcut CMake hedefiyle derlenmeli.
2. `./build/vio_cpp` (dataset yolu göreli olduğundan `build/` içinden çalıştırılmalı) ile `EUROC_MH01_Easy`
   üzerinde koş.
3. Her pencere için konsola `"Pencere <k>: <N> tam kapsama izi, BA basarili=<bool>"` bas. `N` sürekli
   0'a yakınsa ORB/ratio-test parametreleri veya `windowSize` ayarı gerekebilir (şimdiden "düzeltmeye"
   çalışma, sadece gözlemle).
4. `optimize()` sonrası her `pose/vel/ba/bg` bileşenini `std::isfinite` ile kontrol et; NaN/Inf varsa
   pencereyi reddet (madde 3'teki "kötü sonuç zincire yayılmasın" kuralı).
5. Yeşil (VIO) yörüngenin kırmızı (GT) ile şeklen örtüşmesi ve `TrajectoryViewer`'ın canlı ATE metriğinin
   bugünkü saf-VO temeline göre görünür şekilde daha iyi/stabil olması beklenen başarı sinyali.
6. Pencere sınırlarında (~4 karede bir) küçük sıçramalar normaldir (madde 5) — hata sanılmamalı.
7. Yörünge ıraksarsa veya `IsSolutionUsable()` sürekli `false` dönerse: önce Hata B'nin (IMU→kamera
   rotasyonu) doğru uygulandığını, sonra statik yerçekimi kestiriminin makul bir değer verdiğini
   (konsola `gravity` vektörünü yazdırarak, `norm≈9.81` olmalı) kontrol et.
