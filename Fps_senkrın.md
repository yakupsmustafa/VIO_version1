# FPS — Senkron BA'dan Asenkron Front-end/Back-end Mimarisine Geçiş

## Context

Bugünkü oturumda tekrar tekrar ölçüldü: FPS'i sınırlayan şey ORB/eşleştirme (front-end, ~10-70ms/kare) değil, **BA+kovaryans (back-end)** — pencere başına 100-1000ms, ve şu anki mimaride bu, ana kare döngüsünü DOĞRUDAN bloke ediyor. Bugün erken saatlerde `cv::setNumThreads`/`ceres num_threads` ile ~1.7x hızlanma sağlandı (7.8→13.4 FPS) ama bu, yapısal tavanı kırmadı — kullanıcının hedefi olan 35-40 FPS'e senkron mimariyle ulaşılamaz.

Kullanıcı, kök-neden (sistematik IMU sapması) kovalamayı bırakıp bu FPS işine geçmeye karar verdi — bu, kök nedenden bağımsız, kendi başına değerli bir mimari iyileştirme.

**Hedef:** Front-end'i (kare-başı: oku+distorsiyon-düzelt+ORB+eşleştir+IMU-seed+canlı görselleştirme) back-end'den (pencere-başı: BA optimize+kovaryans) ayırıp iki ayrı thread'de çalıştırmak — böylece BA'nın maliyeti front-end'in hızını bloklamaz.

## Tasarım Kararları

1. **İki thread: front-end (mevcut kare döngüsü) + back-end (BA/kovaryans).** Aralarında iki paylaşılan yapı:
   - `SharedAnchorState` — mutex korumalı `{anchor, currentPrior}`. Front-end her karede OKUR (IMU-seed için), back-end her başarılı pencere sonrası YAZAR.
   - `WindowJobQueue` — mutex+condition_variable korumalı, SINIRLI boyutlu (maks. 2) kuyruk. Front-end pencere hazır olduğunda `WindowJob` (kpList, tracks, landmarks, frameTimestamps, seed states) PUSH eder ve BEKLEMEDEN devam eder. Back-end POP edip `vio.optimize()`'ı (DEĞİŞTİRİLMEDEN, mevcut mantık) çalıştırır.

2. **Kuyruk dolarsa front-end BLOKLANMAZ, o pencere atlanır.** Sınırlı kuyruk + "doluysa gönderme, pencereyi at, IMU-seed'le devam et" kuralı — back-end geride kalsa bile front-end hızı hiç düşmez (gerçek-zamanlı sistemlerin standart davranışı: "geç kalan sonucu atla", "hiç üretmemeyi bekleme"). Kaç pencerenin atlandığı loglanır (back-end'in ne kadar geride kaldığının teşhisi için).

3. **Keyframe DB + loop closure back-end'e taşınır** — optimize edilmiş poz/landmark'lara bağımlı oldukları için mantıksal olarak oraya ait. Front-end bu verilere run sırasında hiç erişmiyor (main() sadece iki thread join olduktan SONRA okuyor) — ek kilitleme GEREKMİYOR.

4. **Görselleştirme (TrajectoryViewer, imshow/waitKey) SADECE front-end thread'inde kalır.** OpenCV highgui çoklu-thread'den güvenli çağrılamayabilir (platform bağımlı) — tek thread'de tutmak bunu bertaraf eder, zaten şu an da orada.

5. **Determinizm bilinçli olarak feda edilir (kullanıcıyla zaten konuşuldu).** Gerçek eşzamanlı çalışma, kuyruk-doluluk zamanlamasına bağlı olduğu için artık sadece OpenCV/Ceres iç thread'lerinden değil, GERÇEK wall-clock zamanlamasından da etkilenir. Bu kabul edilen bir bedel — gerçek VIO sistemleri de böyle.

## Uygulama

### Yeni dosya: `include/pip07_AsyncPipeline.hpp`
(mevcut `pip01`...`pip06` adlandırma kuralına uyar)
- `struct WindowJob` — bir pencerenin BA'ya gönderilecek TÜM girdisi (kpList, descList gerekmez sadece tracks+landmarks+frameTimestamps+seedStates+incomingPrior kopyası).
- `class SharedAnchorState` — `get()` (kilitli kopya döner), `set(anchor, prior)` (kilitli yazar).
- `class WindowJobQueue` — `tryPush(job)` (doluysa false döner, front-end bunu "pencere atlandı" olarak loglar), `pop()` (bosaltilirken blocking wait, stop-flag ile uyandırılabilir).
- `std::atomic<bool> stopRequested` — kapanış sinyali.

### `src/main.cpp` değişiklikleri
- `processFrames()` ikiye ayrılır: `frontEndLoop(...)` (mevcut kare-başı mantık, SADECE `vio.optimize()` çağrısı ve onun sonrası kaldırılıp yerine `queue.tryPush(job)` konur) ve `backEndLoop(...)` (mevcut `vio.optimize()` + TESHIS + keyframe/loop-closure mantığı, `queue.pop()` ile beslenir).
- `main()`: iki `std::thread` başlatır (`std::thread feThread(frontEndLoop, ...); std::thread beThread(backEndLoop, ...);`), kare döngüsü bitince (veya 'q') `stopRequested=true` + `queue` üzerindeki cv'yi uyandırıp **her iki thread'i join eder**, SONRA final özet/ATE/loop-closure raporunu (mevcut sıralamayla) yazdırır.

## Doğrulama

1. **Derleme + çökme/kilitlenme kontrolü**: MH01'de GUI kapalı baştan sona çalıştır, temiz bitişi (thread join, hang yok) doğrula.
2. **Doğruluk kontrolü**: Final ATE'yi mevcut senkron referansla (MH01 ~1.40m) kıyasla — aynı MERTEBEDE olmalı (birebir aynı OLMAYACAK, artık gerçekten zamanlamaya bağlı — ama büyük bir kötüleşme varsa bu bir hata/yarış-durumu işaretidir, kabul edilen varyans değil).
3. **FPS ölçümü**: bugünkü yerleşik yöntemle (60 saniyede işlenen kare sayısı) mevcut ~13.4 FPS senkron referansla kıyasla gerçek kazancı ölç.
4. **Atlanan pencere oranı**: loglanan "pencere atlandı" sayısını kontrol et — çok yüksekse (örn. >%20) back-end front-end'e yetişemiyor demektir, bu durumda BA'nın kendi maliyetini düşürmek (ayrı, gelecekteki bir iş) gerekebilir; bu plan kapsamında sadece TEŞHİS edilir, çözülmez.

## Riskler
- Yarış durumu / kilitlenme riski — kısa, minimal kritik bölgelerle (mutex sadece kopyalama sırasında tutulur) azaltılır.
- Kuyruk atlama oranı yüksek çıkarsa FPS kazancı beklenenden düşük kalabilir — bu, ayrı bir "BA'yı ucuzlaştırma" işinin gerekçesi olur, bu planın kapsamı dışında.
