# چک‌لیست تطابق پیاده‌سازی با مقاله

مقاله: Abdelshaheed and Ashour (2025), *Intelligent Scheduling of AI Tasks*

این سند بین «پیاده‌سازی‌شده در سورس» و «اثبات‌شده روی کلاستر واقعی» تفاوت
می‌گذارد. تا زمانی که موارد Server pending اجرا و آرشیو نشوند، نتایج مقاله
بازتولیدشده تلقی نمی‌شوند.

راهنمای وضعیت:

- `[x]` پیاده‌سازی و اعتبارسنجی محلی کامل است.
- `[ ]` کد آماده است، اما پذیرش به اجرای واقعی Kubernetes نیاز دارد.
- `[!]` انحراف آگاهانه از مقاله است و باید در گزارش نهایی افشا شود.
- `[?]` جزئیات در مقاله منتشر نشده و مقدار پروژه یک فرض بازتولید است.

## الگوریتم و Scheduler

| وضعیت | الزام/ادعای مقاله | تطابق سورس | شاهد پذیرش |
|---|---|---|---|
| [x] | شش ویژگی `T,R,M,G,C,P` | Generator، Trainer، Ranker و Collector یک قرارداد مشترک دارند | تست نام، نوع، بازه، مقدار غیرمتناهی و فیلد گمشده |
| [x] | Min-Max نرمال‌سازی در هر Burst | `scheduler/rank.py` | تست جهت، حالت همه مقادیر برابر و معادله |
| [x] | وزن‌های `0.40,0.35,0.20,0.15,0.10,0.05` | عین ضرایب مقاله؛ جمع 1.25 عمداً تغییر نکرده است | تست parity معادله |
| [x] | کمتر بودن `T,M,G,P` و بیشتر بودن `R,C` مطلوب است | جهت نرمال‌سازی صریح است | تست directionality |
| [x] | اجرای Rank بالاتر زودتر | مرتب‌سازی قطعی نزولی با tie-break پایدار | unit test و رکورد order |
| [x] | آزمایش Reversed | `reverse=true` ترتیب Rank را معکوس می‌کند | تست و Plan ثبت‌شده |
| [x] | جمع‌آوری Burst و صف اولویت | قرارداد `run-id + expected-jobs + quiet-period + timeout` | تست count/timeout/config drift |
| [ ] | Manual Pod Binding | `pods/binding` فقط در پروفایل reproduction | نیازمند RBAC واقعی، Node assignment و رکورد Binding |
| [ ] | اجرای تک‌گره‌ای | `--target-node` اجباری و Node پیش از اجرا بررسی می‌شود | Node باید Ready، schedulable و اختصاصی باشد |
| [x] | تأیید شروع اجرای واقعی | JSON marker بعد از allocation و قبل از اولین compute | parser و timeout تست شده؛ marker زنده Server pending |
| [x] | فاصله‌های 0، 1، 2 و 5 ثانیه | `none/fixed` و eventهای شروع/پایان هر wait | fixed-delay test؛ زمان‌بندی زنده Server pending |
| [x] | اجرای بدون pacing سفارشی | نام canonical برابر `custom-baseline` است | Plan Lock و اسناد با هم یکسان‌اند |

## Workload و سناریوها

| وضعیت | الزام/ادعای مقاله | تطابق سورس | شاهد پذیرش |
|---|---|---|---|
| [x] | چهار گروه workload مصنوعی | Generator چهار category قطعی با seed دارد | تست sampling و ID قطعی |
| [?] | توزیع دقیق ویژگی‌های هر گروه | مقاله پارامتر کامل منتشر نکرده؛ ranges/weights پروژه در `jobs.json` ثبت می‌شود | باید به‌عنوان assumption گزارش شود |
| [x] | Matrix multiplication | Trainer با NumPy/BLAS و seed قطعی اجرا می‌کند | unit test؛ عملکرد واقعی Server pending |
| [x] | BLAS تک‌نخی | `threadpoolctl` تعداد thread واقعی را محدود و در prewarm و هر Job ثبت می‌کند | mismatch باعث رد run می‌شود؛ runtime Server pending |
| [x] | کاهش نمایی Loss با `R` | `loss *= exp(-R)` | تست Trainer |
| [x] | Checkpoint interval `C` | هر `C` گام یک payload محدود روی دیسک نوشته و `fsync` می‌شود | تست اندازه فایل و شمارش checkpoint |
| [x] | Partition count `P` | ردیف‌های ماتریس بدون هم‌پوشانی تقسیم و برای هر peer بافر `G` کپی می‌شود | تست پوشش ردیف‌ها؛ همچنان proxy تک‌نودی است |
| [x] | `T` تخمین مشتق‌شده است | از `R/M/G/C/P` و مدل نسخه‌دار ساخته می‌شود و شرط توقف Trainer نیست | مدل مشترک Generator/Trainer/Simulator و شاهد log |
| [x] | سناریوی 12 Job با load عادی | `12-normal` | 5 repetition در Plan |
| [x] | سناریوی 48 Job با load عادی | `48-normal` | 5 repetition در Plan |
| [x] | سناریوی 48 Job با half load | `48-half`؛ scale سراسری `M` برای رسیدن به ۵۰٪ کار تخمینی کالیبره و `T` بازسازی می‌شود | تست نسبت، paired feature و شاهد `jobs.json` |
| [?] | تعریف دقیق half load | مقاله scaling دقیق را روشن نکرده؛ تعریف عملیاتی ۵۰٪ کار تجمیعی، tolerance و scale در Plan/خروجی ثبت می‌شود | باید به‌عنوان assumption گزارش شود |
| [x] | workload یکسان بین default/custom | seed و featureها در هر block/repetition عیناً مشترک‌اند | تست paired materialization |
| [x] | پنج تکرار مستقل | برچسب canonical برابر `0..4` است | Plan و Analyzer آن را enforce می‌کنند |

## ماتریس آزمایش

| وضعیت | بخش | سناریو/پیکربندی | تعداد |
|---|---|---|---:|
| [x] | Pacing مقاله | `48-half-pacing` × default/custom-baseline/1s/2s/5s × 5 | 25 |
| [x] | Main مقاله | سه سناریو × default/custom-baseline/reversed × 5 | 45 |
| [x] | کل مقاله | Plan Lock `article-70.json` | **70** |
| [x] | RFAP پیشنهادی پروژه | adaptive در Pacing و سه سناریوی Main | 20 |
| [x] | کل Extended | Plan Lock `extended-90.json` | **90** |
| [!] | Adaptive RFAP | در مقاله وجود ندارد و نباید به نویسندگان نسبت داده شود | گزارش و نمودار جداگانه الزامی است |

Planها با ترتیب deterministic-randomized-blocks قفل شده‌اند:

- Article 70 SHA-256:
  `6076e7863c7b2507baf856c3f78750f594f834590847a0d1f403758b8bac4566`
- Extended 90 SHA-256:
  `a383708fc9cecaec48d573716687a51d9ef9d379d5e9a932029e2487f688b52c`

## اندازه‌گیری و تحلیل

| وضعیت | معیار مقاله | تطابق سورس | شرط پذیرش |
|---|---|---|---|
| [x] | JCT هر Job | creation تا completion | timestamp کامل و مرتب؛ مقدار منفی رد می‌شود |
| [x] | Avg/Min/Max JCT | summary از Jobهای کامل | run ناقص وارد aggregate نمی‌شود |
| [x] | p95 JCT | Hyndman/Fan type 7 | روش در schema ثبت شده است |
| [x] | Makespan | اولین submission تا آخرین completion | count کامل اجباری است |
| [x] | ILT | فاصله execution-startهای مرتب‌شده | marker همه Jobها اجباری است |
| [x] | ECDF میانگین | ECDF هر run روی grid مشترک | pooling بین repetitionها ممنوع است |
| [x] | IQR | percentileهای 25/75 بین runها | داده و نمودار تولید می‌شود |
| [x] | Confidence interval | Student-t 95% روی metric سطح run و اختلاف‌های paired بر اساس scenario/repetition/seed | CSV و تست آماری paired |
| [x] | رد داده ناقص | Collector fail-closed و failure artifact | missing/failed/duplicate/mismatch رد می‌شود |
| [x] | اثبات ترتیب واقعی | scheduler record در نتیجه custom embed و جداگانه archive می‌شود | rank/order/job set/timestamp/pacing دقیقاً validate می‌شود |
| [x] | متادیتای بازتولید | Plan/artifact hash، context، Helm، Kubernetes، Node و image IDs | نبود یا mismatch باعث رد resume/analysis می‌شود |
| [x] | قرارداد Runtime زمان‌بند | quiet/burst/poll/execution/API/adaptive از Helm به آرگومان Deployment و رکورد هر Run قفل شده‌اند | drift زنده یا mismatch رکورد با Helm باعث رد Run می‌شود |
| [x] | کنترل cache/carryover | prewarm همان digest روی Node و cooldown تمیز ۳۰ ثانیه‌ای قبل هر run | شاهد Minikube Docker، BLAS، pressure و continuity داخل snapshot hash |

## Docker، امنیت و Kubernetes

| وضعیت | کنترل | پیاده‌سازی | پذیرش نهایی |
|---|---|---|---|
| [x] | Image غیر-root | UID/GID 10001 در هر دو Dockerfile | build/runtime inspect Server pending |
| [x] | Pod workload امن | token/service-links خاموش، read-only، no capabilities، seccomp و `/tmp` محدود | schema محلی پاس شده؛ runtime Server pending |
| [x] | Dependency pinning | requirements نقش‌محور + constraints + build backend pin | import و `pip check` پاس |
| [x] | Image immutable | Helm و Runner digest را پشتیبانی/اجبار می‌کنند | digest واقعی Registry pending |
| [x] | RBAC کمینه | Role namespace و ClusterRole شرطی Node/Metrics | `kubectl auth can-i` pending |
| [x] | Health/metrics | `/livez`، `/readyz` و `/metrics` | پاسخ زنده pending |
| [x] | Observability | JSON logs، Prometheus metrics و eventهای pacing | scrape/log زنده pending |
| [x] | Rolling update | یک replica و `maxSurge=0` بدون overlap | rollout/rollback زنده pending |
| [x] | NetworkPolicy | ingress محدود و egress فقط به پورت‌های API با CIDR اختیاری | اجرای CNI pending |
| [x] | Secret hygiene | فقط reference به Secret؛ هیچ credential در سورس نیست | pull واقعی pending |
| [x] | Profile production | gate-controller، Binding را به kube-scheduler واگذار می‌کند | smoke جداگانه pending |
| [x] | Recovery production | state اتمیک روی PVC، تطبیق Pod UID و resume بدون release تکراری | تست replacement زنده pending |
| [!] | Profile production | numerically معادل reproduction مقاله نیست | باید جدا گزارش شود |

## چک‌لیست Go/No-Go برای ادعای بازتولید مقاله

- [x] الگوریتم، workload، Plan، schema، تست و manifests محلی کامل‌اند.
- [x] 145 تست خودکار بدون skip یا failure پاس شده‌اند.
- [x] Helm lint و schema سخت‌گیرانه Kubernetes 1.36 پاس شده است.
- [x] Plan Lock دقیق 70 و 90 آماده است.
- [ ] هر دو Docker image روی سرور build و smoke شوند.
- [ ] imageها push و digest واقعی Registry ثبت شود.
- [ ] server preflight و `kubectl apply --dry-run=server` پاس شود.
- [ ] reproduction scheduler Ready و RBAC آن تأیید شود.
- [ ] smoke سه‌Job شامل order/binding/marker/result پاس شود.
- [ ] برای Extended، Metrics timestamp cadence و fail-closed تست شود.
- [ ] یک اجرای pilot با `--limit 1` پذیرفته و archive شود.
- [ ] سپس دقیقاً یکی از Planهای 70 یا 90 بدون overlap اجرا شود.
- [ ] Analyzer بدون `--allow-partial` پاس شود.
- [ ] نتایج خام، scheduler records، Helm/cluster metadata و digestها backup شوند.

تا تکمیل همه موارد `[ ]` مرتبط، وضعیت صحیح پروژه «آمادهٔ ورود به فاز
راه‌اندازی/اعتبارسنجی Kubernetes» است، نه «مقاله با موفقیت بازتولید شد».

ممیزی عددبه‌عدد ادعاهای منتشرشده و مرز دقیق شواهد اجرای فرانسه در
`docs/ARTICLE_CLAIM_STATUS_FA.md` ثبت شده است. راهنمای ورود و snapshot معماری
فرانسه نیز در `docs/FRANCE_ACCESS_AND_TOPOLOGY_FA.md` قرار دارد.
