# Adım 6: Paralaks Tabanlı Değişken Pencere Kapatma (Keyframe Yaklaşımı)

## Context

Bugün üç ayrı hipotez (IMU ağırlıkları artırma, KLT izleme, çok-görüşlü triangülasyon) test edildi;
üçü de ölçüldüğünde işe yaramadı ya da kötüleşti — ama bu süreçte kök neden **doğrudan ölçümle**
kesinleşti: sistem her 5 karede bir, kameranın gerçekte ne kadar hareket ettiğine bakmaksızın
BA penceresini zorla kapatıyor. Kamera neredeyse durgunken (EuRoC MH01'in ilk ~1000 karesi, ~0.5m'lik
alan) 2 görüşlü triangülasyonun tabanı (baseline) o kadar küçük kalıyor ki üçgenleme sayısal olarak
dejenere oluyor ve TÜM aday izler cheirality testinden aynı anda elenebiliyor — bir enstrümante
koşuda 160 geçerli iz girip 0'ı hayatta kalmıştı. Bu "0 iz" olayı, sistemi düzeltilmemiş ham IMU
dead-reckoning'e düşürüyor; bu da sabah bulup düzelttiğimiz hız/bias kaçış hatasının ilk
tetikleyicisiydi.

Bu adımda kök nedeni **doğrudan** hedefliyoruz: pencereyi sabit kare sayısına göre değil, izlerin
**medyan piksel yer değiştirmesi** (paralaks) yeterli seviyeye ulaşana kadar açık tutuyoruz —
tıpkı gerçek VIO sistemlerinin (VINS-Mono, ORB-SLAM) keyframe seçiminde yaptığı gibi.

## Tasarım

### 1. `Viobundleadjustment.hpp` — 2 satırlık davranış düzeltmesi + isim değişikliği

`optimize()` içindeki TEK yer, sabit `windowSize`'a bağımlı olan yer:
```cpp
int n = windowSize;
if ((int)frameStates.size() != n || (int)frameTimestamps.size() != n) return false;
```
şuna değişecek:
```cpp
int n = (int)frameStates.size();
if (frameStates.empty() || frameTimestamps.size() != frameStates.size()) return false;
```
Fonksiyonun geri kalanı (reprojection döngüsü, IMU factor döngüsü, bugün eklenen bias/hız prior
döngüleri) zaten `n` üzerinden genel yazılmış — başka HİÇBİR satır değişmiyor.

Üye/parametre adı `windowSize`/`windowSize_` → **`maxWindowFrames`/`maxWindowFrames_`** olarak
değiştirilecek (artık `optimize()` tarafından hiç okunmuyor, sadece bilgi amaçlı tutuluyor — eski adı
bırakmak yanıltıcı olurdu). Yorum eklenecek: "optimize() artik gercek frameStates.size()'i kullanir,
bunu OKUMAZ".

### 2. Yeni yardımcı: `computeMedianParallaxPx` (`pip04_VioWindow.hpp`)

`triangulateAndFilterWindowTracks`'ın hemen üstüne, aynı `tracks`/`kpList` yapılarıyla çalışan:
```cpp
inline double computeMedianParallaxPx(
        const std::vector<std::vector<cv::KeyPoint>>& kpList,
        const std::vector<std::vector<int>>& tracks) {
    if (tracks.empty() || kpList.empty()) return 0.0;
    int lastFrame = (int)kpList.size() - 1;
    std::vector<double> disp;
    disp.reserve(tracks.size());
    for (const auto& tr : tracks) {
        cv::Point2f p0 = kpList[0][tr.front()].pt;
        cv::Point2f pl = kpList[lastFrame][tr.back()].pt;
        disp.push_back(cv::norm(pl - p0));
    }
    size_t mid = disp.size() / 2;
    std::nth_element(disp.begin(), disp.begin() + mid, disp.end());
    return disp[mid];
}
```
Medyan (ortalama değil) tercih edildi — birkaç yanlış-eşleşen/dinamik-nesne izi tek başına sonucu
çarpıtmasın diye.

### 3. `main.cpp::processFrames` — pencere kapatma tetikleyicisi

`buildFullCoverageTracks`, `kpList.size() >= minWindowFrames` olduğu ANDAN itibaren **her karede**
çağrılacak (önceden sadece sabit sınırda 1 kez çağrılıyordu) — maliyeti önemsiz (ORB tespiti +
BFMatcher'ın O(N·M) maliyetinin yanında, hashmap tabanlı zincirleme milisaniyenin altında).
Kapatma anında **aynı `tracks` nesnesi** `triangulateAndFilterWindowTracks`'a geçiriliyor (yerinde
filtreleniyor) — tekrar hesaplama YOK.

```cpp
if ((int)kpList.size() >= minWindowFrames) {
    std::vector<std::vector<int>> tracks = buildFullCoverageTracks(pairwiseMatches, (int)kpList[0].size());
    double parallaxPx = computeMedianParallaxPx(kpList, tracks);
    bool reachedMax = (int)kpList.size() >= maxWindowFrames;
    // Guvenlik: cok az izden hesaplanan medyan gurultulu olabilir - boyle durumda paralaksa
    // GUVENME, sadece maxWindowFrames tavanina gelince kapat (eskisi gibi güvenli geri dönüş).
    bool enoughParallax = ((int)tracks.size() >= minTracksForParallaxTrust) && (parallaxPx >= parallaxThresholdPx);

    if (enoughParallax || reachedMax) {
        int framesUsed = (int)kpList.size();
        std::vector<std::array<double,3>> landmarks;
        triangulateAndFilterWindowTracks(K, windowStates.front().pose, windowStates.back().pose,
                                          kpList, tracks, landmarks);
        std::cout << "Pencere " << windowCount << " (" << framesUsed << " kare, paralaks="
                  << parallaxPx << "px" << (reachedMax && !enoughParallax ? ", MAKS-KARE limiti" : "")
                  << "): " << tracks.size() << " tam kapsama izi, ";
        // ... geri kalan blok (BA cagrisi, basari/basarisizlik, anchor guncelleme, TESHIS,
        //     pencere sifirlama) BUGUNKU MANTIKLA BIREBIR AYNI - sadece tetikleyici degisti.
    }
}
```

`[TESHIS]` satırına `kare=<framesUsed>` ve `paralaks=<parallaxPx>px` eklenecek — mekanizmanın
gerçekten çalıştığını (durgun bölgede pencerenin büyüdüğünü, hareketli bölgede küçüldüğünü)
doğrudan gözlemlemek için.

### 4. Parametreler

- **`minWindowFrames = 3`**: en az 2 sıçrama geçsin diye alt sınır (tek bir gürültülü eşleşme setine
  aşırı tepki vermesin).
- **`maxWindowFrames = 10`**: bugünkünün 2 katı — durgun bölgede gerçek paralaks biriktirmesi için
  iki kat daha fazla şans. Çok büyütülmezse: daha uzun süre düzeltmesiz IMU + ORB eşleşme kalitesinin
  uzun taban'da düşme riski.
- **`parallaxThresholdPx = 20.0`**: fx≈458.65px iken ~2-2.5° paralaksa denk gelir; VINS-Mono/ORB-SLAM
  gibi sistemlerin keyframe eşiklerine yakın, temkinli (yüksek) uçta — sorunumuz "az paralaks"
  olduğu için temkinli tarafta durmak doğru tercih.
- **`minTracksForParallaxTrust = 15`**: bu sayının altında iz varsa medyan güvenilmez sayılır,
  sadece `maxWindowFrames` tavanı kapatma sebebi olabilir.

Not: Bu sayılar deneyle doğrulanacak başlangıç noktaları — kesin/nihai değerler değil.

### 5. `main()` değişiklikleri

```cpp
const int minWindowFrames = 3;
const int maxWindowFrames = 10;
const double parallaxThresholdPx = 20.0;
const int minTracksForParallaxTrust = 15;
...
VIOBundleAdjustment vio(K, imuPreint, gravity, maxWindowFrames);
...
processFrames(images, extractor, matcher, K, R_BS, undistortMap1, undistortMap2,
              imuPreint, vio, minWindowFrames, maxWindowFrames, parallaxThresholdPx,
              minTracksForParallaxTrust, gravity, alignedGt, viewer,
              windowFailCount, windowTotalCount);
```

## Değişen Dosyalar

- **`include/pip04_VioWindow.hpp`**: `computeMedianParallaxPx` eklenir. Başka hiçbir fonksiyon
  değişmez (`triangulateAndFilterWindowTracks`, `buildFullCoverageTracks`, `seedWindowStatesFromImu`
  zaten pencere uzunluğundan bağımsız/genel).
- **`include/Viobundleadjustment.hpp`**: `optimize()`'daki 2 satır + üye adı değişikliği.
- **`src/main.cpp`**: `processFrames` imzası + pencere-kapatma bloğu tetikleyicisi; `main()`'deki
  sabitler ve çağrı yeri.
- **Değişmeyecek**: `Imufactor.hpp`, `Imupreintegration.hpp` (pencere kavramına hiç referans yok,
  grep ile doğrulandı), `pip05_KltTracker.hpp` (kullanılmayan, dokunulmuyor).

## Doğrulama

1. Build: mevcut CMake hedefiyle temiz derlenmeli.
2. Aynı `nFrames=2000` testi (bugünkü HER karşılaştırma bununla yapıldı) çalıştırılıp şu tabloyla
   doğrudan kıyaslanacak:

   | Varyant | Pencere başarısı | ATE (kare 2000) |
   |---|---|---|
   | Temel (sabit windowSize=5) | 484/499 (%97) | ~1.98m |
   | KLT (geri alındı) | %87 | 2.24m |
   | Çok-görüşlü triangülasyon (geri alındı) | %96.4 | 4.01m |
   | **Paralaks-tabanlı (bu plan)** | ? | ? |

3. **Asıl başarı sinyali: ATE'nin 1.98m'nin BELİRGİN ŞEKİLDE altına inmesi.** Bugün iki kez
   öğrendik ki pencere-başarı-oranı tek başına yeterli kanıt değil (iki başarısız denemede de
   pencere sayısı değişmemiş/iyileşmiş gibi görünürken ATE kötüleşmişti) — bu sefer de sadece
   pencere sayısına bakıp zafer ilan etmeyeceğiz.
4. Yeni `kare=`/`paralaks=` alanları doğrudan izlenecek: durgun bölgede pencerenin gerçekten
   5'in üzerine çıkıp çıkmadığı, hareketli bölgede 3'e yakın kalıp kalmadığı — mekanizmanın
   tasarlandığı gibi çalıştığının ilk elden gözlemlenebilir kanıtı.
5. `[TESHIS] |v|/|C|/|ba|` bugünkü düzeltmeden sonraki makul fiziksel aralıkta kalmalı (sabah
   düzeltilen kaçış hatasına karşı regresyon kontrolü).
