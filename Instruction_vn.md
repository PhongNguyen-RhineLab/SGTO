# Hướng dẫn phần code dự án SGTO (Quy hoạch trạm sạc EV)

Tài liệu này mô tả toàn bộ mã nguồn thí nghiệm cho bài toán quy hoạch trạm sạc xe điện dựa trên rủi ro, thuật toán **SGTO (Scenario-Based Global Trajectory Optimization)**, chạy trên hai bộ dữ liệu thực: **UrbanEV (Thâm Quyến)** — instance chính, và **Smarter Mobility (Paris Belib')** — instance phụ kiểm tra tính tổng quát.

Tài liệu gồm hai phần:

- Phần A dành cho người muốn **chạy lại thí nghiệm**: cài đặt, câu lệnh, đọc kết quả.
- Phần B dành cho người muốn **hiểu và phát triển tiếp code**: kiến trúc, từng module, công thức hàm mục tiêu, chi tiết thuật toán.

---

## PHẦN A. CHẠY LẠI THÍ NGHIỆM

### A.1. Yêu cầu môi trường

- Python 3.10 trở lên
- Bắt buộc: `numpy`, `pandas`
- Tùy chọn: `pandapower` (cho `--grid ieee33`), `osmnx` (cho `--roads osmnx`), `matplotlib` (cho `make_plots.py`)
- Kết nối internet để tải dữ liệu (một lần)

Không cần GPU. Toàn bộ tính toán là numpy trên CPU.

### A.2. Cài đặt

```bash
pip install -r requirements.txt
python setup_data.py all        # tải UrbanEV -> UrbanEV/data và Paris -> smarter-mobility/data
pip install pandapower osmnx matplotlib   # tùy chọn
```

Nếu mạng chặn GitLab, tải `train.csv` thủ công từ https://gitlab.com/smarter-mobility-data-challenge/tutorials và đặt vào `smarter-mobility/data/train.csv`. Có thể trỏ thư mục khác bằng `--data-dir`.

### A.3. Các lệnh chạy

Chạy một thí nghiệm đơn:

```bash
python run_experiment.py --quick                        # chạy thử nhanh, khoảng 2 phút
python run_experiment.py                                # UrbanEV, lưới synthetic, đủ 7 phương pháp
python run_experiment.py --dataset paris                # instance Paris
python run_experiment.py --grid ieee33                  # lưới IEEE 33-bus (cần pandapower)
python run_experiment.py --dataset paris --roads osmnx  # khoảng cách đường bộ OSM
python run_experiment.py --methods sgto cost_aware_greedy
python run_experiment.py --list                         # liệt kê dataset và phương pháp
```

Các cờ chính (cờ do người dùng truyền luôn thắng default per-dataset trong `registry.py`):

| Cờ | Ý nghĩa |
|---|---|
| `--dataset urbanev\|paris` | chọn instance; tự đặt data_dir, budget, split_date mặc định |
| `--grid synthetic\|ieee33` | mô hình sức chứa lưới (xem B.3c) |
| `--roads auto\|osmnx\|geodesic` | nguồn khoảng cách đường bộ |
| `--budget`, `--rho`, `--split-date` | ghi đè tham số bài toán |
| `--seed` | seed cho CẢ kịch bản lẫn thuật toán |
| `--algo-seed` | chỉ đổi seed thuật toán, giữ nguyên bộ test — dùng cho ablation đa seed |
| `--no-risk-in-weights` | trọng số semi-gradient chỉ theo mean (hành vi cũ, xem B.10 bài học 4) |
| `--demand-growth` | (Paris) nhân nhu cầu theo hệ số tăng trưởng EV |
| `--tag`, `--out` | đặt tên file kết quả, thư mục ra |
| `--quick` | thu nhỏ để kiểm tra pipeline; **không dùng số này cho paper** |

Chạy cả bộ thí nghiệm bằng `run_all.sh` (chia stage, lỗi một run không dừng cả bộ):

```bash
bash run_all.sh                 # tất cả (vài giờ)
bash run_all.sh main            # chỉ 4 bảng chính dataset x lưới
bash run_all.sh main weights    # một tập stage tùy chọn
QUICK=1 bash run_all.sh         # smoke test toàn bộ
bash run_all.sh --list          # xem tên stage
```

Stage: `setup main rho_seeds weights paris_calib rho_sweep`. Ghi đè tham số quét qua biến môi trường `SEEDS`, `GROWTHS`, `RHOS`.

### A.4. Đọc kết quả

Kết quả ghi vào `results/<dataset>/results_<tag>.json` (không có `--tag` thì là `results.json`). File mở đầu bằng block `_config` lưu toàn bộ tham số của run (dataset, grid, budget, rho, seed, risk_in_weights...) để tái lập. Với mỗi phương pháp, file chứa:

- Bảng **metrics** (toàn bộ chỉ số của paper, xem mục B.7)
- **Lời giải**: danh sách cặp `(zone id, level)` — tức là đặt trạm ở vùng nào, mức công suất nào
- **Lịch sử vòng lặp** của SGTO (giá trị validation ở từng vòng, vòng nào được chấp nhận)

Ví dụ cách đọc nhanh bằng Python:

```python
import json
r = json.load(open("results/urbanev/results_main.json"))
for name in r:
    if name.startswith("_"): continue
    m = r[name]["metrics"]
    print(name, round(m["F_rob"], 2), round(m["F_rob_gain"], 2))
```

Hai công cụ gộp kết quả thành bảng LaTeX và hình vẽ cho paper:

```bash
# bảng một run / mean±std đa seed / quét tham số
python make_tables.py table results/urbanev/results_main.json
python make_tables.py seeds results/urbanev/results_main.json \
    results/urbanev/results_rho_seed*.json --methods sgto sgto_risk_neutral
python make_tables.py sweep "results/urbanev/results_rho[0-9]*.json" --param rho --methods sgto

# hình: bar so sánh phương pháp / đường cong sweep / hội tụ SGTO
python make_plots.py methods results/urbanev/results_main.json -o figures
python make_plots.py sweep "results/urbanev/results_rho[0-9]*.json" --param rho --methods sgto -o figures
python make_plots.py sweep "results/paris/results_growth*.json" --param demand_growth \
    --methods cost_aware_greedy sgto sgto_risk_neutral -o figures
python make_plots.py history results/urbanev/results_main.json --methods sgto -o figures
```

Lưu ý quan trọng khi đọc số quét rho: `F_rob` và `F_rob_gain` được tính bằng chính rho của run đó nên **không so sánh được giữa các mức rho**; hai công cụ trên mặc định dùng các cột độc lập với rho (`F_mean`, `CVaR_loss`, overload, cost) cho mode sweep.

Một điểm cần lưu ý khi đọc số: `F_rob` mang một hằng số âm rất lớn vì luôn có một khối nhu cầu không thể phục vụ (unmet demand) ngay cả với lời giải tốt nhất. Vì vậy code bổ sung chỉ số `F_rob_gain` (phần cải thiện so với lời giải rỗng) để so sánh giữa các phương pháp cho trực quan hơn.

### A.5. Kết quả tham chiếu (để đối chiếu)

UrbanEV, lưới synthetic, cấu hình đầy đủ, seed 42 (budget 100000, rho 0.3, trọng số thẳng hàng — xem B.10 bài học 4):

| Phương pháp | F_rob(test) | CVaR |
|---|---|---|
| Random Search | -2978.67 | 2914 |
| Simulated Annealing | -1921.96 | 2717 |
| Cost-Aware Greedy | -1901.59 | 2808 |
| SGTO không exchange | -1886.23 | 2767 |
| Greedy + One-Exchange | -1863.48 | 2737 |
| SGTO risk-neutral | -1745.93 | 2547 |
| SGTO (đầy đủ) | -1699.88 | 2426 |

Thứ tự đơn điệu theo từng thành phần thuật toán. Hai lưu ý khi diễn giải:

1. **Cặp SGTO / SGTO risk-neutral ở rho 0.3 là hòa thống kê.** Qua 4 seed thuật toán trên cùng bộ test: SGTO -1772.9 ± 48.7 so với risk-neutral -1742.5 ± 35.7 (F_rob); CVaR 2578 ± 102 so với 2500 ± 73. Nguyên nhân: CVaR trên 12 kịch bản test xấp xỉ trung bình của ~1 kịch bản tệ nhất — estimator phương sai rất lớn. Trong paper, luận điểm risk-aware nên đứng trên đường quét rho, không đứng trên cặp so sánh này.
2. **Quét rho cho đường cong risk-return sạch:** rho đi 0 -> 2 thì CVaR(test) giảm đơn điệu 3153 -> 2617 -> 2530 -> 2482 -> 2415, overload tổng giảm 209.5k -> 131.5k kW, trong khi FR gần phẳng (0.44-0.47) và chi phí giảm.

Paris (demand-growth 3, budget 7500): greedy đạt F_rob 70.15 với CVaR 9.86 và chi phí 4620; SGTO đạt 76.69 với CVaR 4.58 và chi phí 2646 — greedy đuổi coverage mà xây quá tay vào vùng phạt lưới, SGTO tiêu ít hơn 43% và an toàn gấp đôi ở đuôi rủi ro.

---

## PHẦN B. HIỂU VÀ PHÁT TRIỂN CODE

### B.1. Triết lý kiến trúc

Nguyên tắc cốt lõi: **thuật toán chỉ nhìn thấy một đối tượng `ProblemInstance`, không bao giờ chạm vào file thô.** Mọi việc đọc file, biến đổi dữ liệu đều nằm trong tầng `data_processing/`. Nguyên tắc này đã được kiểm chứng thực tế: thêm instance Paris chỉ cần một loader mới (`paris.py`) và một entry trong `registry.py`, không sửa một dòng thuật toán nào. Mọi loader tuân theo cùng một giao diện duck-typed: `zone_ids`, `dist_km`, `adj`, `district_of_zones()`, `days()`.

Chuỗi phụ thuộc một chiều:

```
file csv thô  ->  data_processing/  ->  ProblemInstance  ->  algorithms/  ->  metrics.py
```

### B.2. Cấu trúc thư mục

```
config.py                  toàn bộ giả định và siêu tham số gom về một chỗ
data_processing/
  registry.py              đăng ký dataset: tên -> loader + default per-dataset
  common.py                đường tải ngày chuẩn, haversine, SyntheticGrid
  urbanev.py               đọc csv UrbanEV, dựng ProblemInstance
  paris.py                 đọc train.csv Smarter Mobility (Belib Paris)
  scenarios.py             dựng tập kịch bản train/val/test từ các ngày thực
  grid_ieee33.py           grid provider IEEE 33-bus qua pandapower
  roads_osmnx.py           ma trận khoảng cách đường bộ OSM, có cache
model/
  instance.py              các dataclass ProblemInstance và Scenario
  reward.py                F_omega, CVaR, F_rob, IncrementalState (tính gain nhanh)
algorithms/
  base.py                  giao diện Solver, kiểu SolveResult
  greedy.py                greedy nhận biết chi phí (cũng là bộ khởi tạo cho SGTO)
  local_search.py          one-exchange, drop-and-refill, baseline greedy+exchange
  sgto.py                  SGTO đầy đủ: semi-gradient, knapsack DP, exchange,
                           validation acceptance; các cờ ablation
  random_search.py         baseline ngẫu nhiên khả thi
  annealing.py             baseline simulated annealing
metrics.py                 tất cả chỉ số của paper trên tập test giữ riêng
run_experiment.py          điểm vào chương trình
setup_data.py              tải dữ liệu một lệnh
run_all.sh                 chạy cả bộ thí nghiệm theo stage
make_tables.py             gộp kết quả thành bảng LaTeX
make_plots.py              vẽ hình cho paper (bar, sweep, hội tụ)
```

Thêm một dataset mới = một module loader có hàm `build_instance(cfg)` cộng một entry trong `data_processing/registry.py` (kèm default riêng: data_dir, split_date, budget, và hook `configure` nếu cần scale mức công suất).

### B.3. Ánh xạ dữ liệu UrbanEV sang mô hình

Đây là bảng quan trọng nhất để hiểu code khớp với paper thế nào.

| Đối tượng trong mô hình | Nguồn UrbanEV |
|---|---|
| Vùng nhu cầu U (275) | các cột của `volume.csv` (các zone giao thông) |
| Ứng viên V | cùng các zone đó, quy hoạch từ đầu (green-field) |
| Tập nền E (825) | zone x 3 mức công suất, định nghĩa trong `config.py` |
| d_{u,t} | lượng sạc theo giờ (kWh), một kịch bản = một ngày, T = 24 |
| Độ phủ a_{u,e} | suy giảm hàm mũ theo khoảng cách đường (`distance.csv`) x hệ số mức |
| Cặp synergy | zone kề nhau (`adj.csv`) nằm trong bán kính D_max |
| Vùng lưới điện Z (11) | nhóm quận, lấy từ `TAZID // 100` |
| Kịch bản (scenario) | ngày thường/cuối tuần/cao điểm thực + nhiễu cắt lưới và tăng đột biến |

Về phân chia thời gian: các ngày **trước 2023-01-15** dùng làm tập train; validation và test lấy từ các ngày **sau đó và rời nhau**. Nhờ vậy bước chấp nhận theo validation và bước đánh giá cuối cùng **không bao giờ nhìn thấy ngày train**. Đây là điểm để tránh overfit và cần nêu rõ trong paper.

Ví dụ minh họa quy mô: 275 zone x 3 mức (nhỏ/vừa/lớn) = 825 phần tử trong tập nền E. Một lời giải là chọn tối đa một mức cho mỗi zone, sao cho tổng chi phí không vượt ngân sách.

### B.3b. Ánh xạ dữ liệu Paris (Smarter Mobility) sang mô hình

Nguồn: challenge "Smarter Mobility Data Challenge" (Amara-Ouali et al., arXiv:2306.06142), file `train.csv`: occupancy 15 phút của 91 trạm Belib', giai đoạn 2020-07-03 đến 2021-02-18, cột `date, Station, Available, Charging, Passive, Other, Latitude, Longitude, Postcode, area`.

| Đối tượng mô hình | Nguồn Smarter Mobility |
|---|---|
| Vùng nhu cầu U (91) | mỗi trạm Belib' là một zone |
| Tập nền E (273) | trạm x 3 mức công suất scale theo trạm (xem dưới) |
| Nhu cầu d_{u,t} | trung bình giờ của số plug đang sạc x `plug_power_kw` (kWh), có nhân `demand_growth` |
| Khoảng cách đường bộ | OSMnx nếu bật hoặc đã cache; không thì great-circle x 1.3 |
| Vùng lưới Z | arrondissement theo `Postcode` (NaN rơi về `area`); fallback 4 area của challenge |
| Kịch bản | ngày thực, chia tại 2020-12-01 |

Điểm dễ vấp nhất: **mức công suất phải scale theo trạm, không dùng mức của Thâm Quyến.** Zone UrbanEV là TAZ gộp hàng trăm kWh/h; trạm Belib' chỉ vài chục. Giữ mức Thâm Quyến làm instance suy biến (mọi phương pháp hội tụ về all-small, ngân sách không ràng buộc). `registry.py` xử lý việc này qua hook `_paris_configure`: mức 22/110/360 kW, bán kính coverage hẹp hơn (decay 1 km, radius 3 km). Tương tự, nhu cầu thời đại dịch quá thấp — cờ `--demand-growth` (khuyến nghị 3, cần trích dẫn dự phóng tăng trưởng EV) mới tạo được tradeoff mức công suất thật.

### B.3c. Hai mô hình sức chứa lưới

**`--grid synthetic` (mặc định)** — giả định phát biểu trong paper: tải nền mỗi quận theo đường feeder ngày chuẩn scale theo nhu cầu sạc của quận; `g_z = margin x (đỉnh tải nền + tải trạm tham chiếu)`. Cài trong `data_processing/common.py` (`SyntheticGrid`).

**`--grid ieee33`** — thay giả định trên bằng giới hạn vật lý từ hệ thống thử nghiệm IEEE 33-bus qua pandapower (`data_processing/grid_ieee33.py`):

1. 32 bus tải của `case33bw` được chia dọc feeder thành các nhóm liền kề, khớp tỉ trọng tải nền của nhóm với tỉ trọng nhu cầu sạc của quận.
2. Sức chứa mỗi quận = **hosting capacity theo ràng buộc điện áp**: hệ số nhân tải lớn nhất trên các bus của quận sao cho điện áp bus nhỏ nhất còn >= `ieee33_vmin` (0.90 pu), tìm bằng binary search trên power flow AC. Dùng điện áp vì `max_i_ka` trong case33bw là placeholder — điện áp mới là giới hạn vật lý thật.
3. Mỗi ngày, mạng 3.715 MW được calibrate về độ lớn thành phố qua tổng đỉnh nhu cầu quận.

Mọi power flow chạy một lần lúc dựng; mỗi kịch bản chỉ còn phép nhân mảng. Kết quả đáng chú ý: hệ số headroom dao động 1.37x (quận cuối feeder) đến 25x (gần trạm biến áp) — tính không đồng nhất theo topology mà mô hình synthetic không có; đáng một câu trong phần thực nghiệm.

### B.4. Module `config.py`

Mọi hằng số mà paper coi là **giả định** đều nằm ở đây, để có thể trích dẫn và thay đổi trong ablation. Các nhóm chính:

- `CapacityLevelConfig`: định nghĩa từng mức công suất (nhỏ/vừa/lớn) — chi phí c_e, công suất q_e. Quy ước 1 đơn vị = 1000 USD.
- Trọng số hàm mục tiêu: alpha (coverage), beta (synergy), gamma (grid penalty), eta (unmet demand), rho (mức ngại rủi ro của CVaR).
- Tham số thuật toán: ngân sách B, số vòng lặp K, số kịch bản lấy mẫu m, cùng các tham số của phần iterated local search.

Ba tham số ILS đáng chú ý (đây là phần mở rộng so với Algorithm 1 gốc):

```python
patience: int = 3          # số lần validation từ chối liên tiếp trước khi dừng
                           # patience = 1 khôi phục đúng luật "dừng khi bị từ chối lần đầu" của paper
perturb_frac: float = 0.34 # khi bị từ chối, bỏ ngẫu nhiên tỉ lệ này số phần tử rồi refill
                           # đặt 0.0 để tắt, quay về resample thuần
final_polish: bool = True  # sau vòng lặp, chạy một lượt exchange trên toàn bộ tập train
```

Đặt `patience=1, perturb_frac=0.0, final_polish=False` sẽ **khôi phục chính xác Algorithm 1 trong paper (ver 1)**.

### B.5. Module `model/instance.py`

Chứa hai dataclass, đều là dữ liệu thuần (không có logic thuật toán).

**`Scenario`** — một kịch bản omega:

```python
name: str
prob: float
demand:   np.ndarray   # (T, U)  d_{u,t}, kWh
grid_cap: np.ndarray   # (T, Z)  g_{z,t}, kW
bg_load:  np.ndarray   # (T, Z)  tải nền l_{z,t}, kW
zeta:     np.ndarray   # (T,)    hồ sơ mức sử dụng zeta_t trong [0,1]
```

**`ProblemInstance`** — toàn bộ dữ liệu bài toán, độc lập với dataset. Các trường then chốt:

- `zone_of`, `level_of` (mỗi phần tử e là cặp zone-mức), `cost`, `qcap`, `budget`
- `A` (U x E): ma trận độ phủ a_{u,e}, giá trị trong [0,1], là hình học tĩnh nhân hệ số mức
- `B_zone` (n_zones x n_zones): trọng số synergy đối xứng giữa các zone, đã gộp hệ số và ngưỡng cắt D_max
- `grid_of` (E,): phần tử e thuộc vùng lưới nào
- `w_t`: trọng số thời gian
- `scen_train`, `scen_val`, `scen_test`: ba tập kịch bản

Hai phương thức tiện ích quan trọng:

- `feasible(X)`: kiểm tra ràng buộc partition (mỗi zone tối đa một mức) và ràng buộc ngân sách.
- `solution_cost(X)`: tổng chi phí của lời giải.

Quy ước chỉ số dùng xuyên suốt code: `u` là vùng nhu cầu, `e` là phần tử tập nền (zone, mức), `z` là vùng lưới, `t` là mốc thời gian trong một ngày, `omega` là chỉ số kịch bản.

### B.6. Module `model/reward.py` — trái tim của mô hình

Đây là nơi hiện thực hóa toàn bộ hàm mục tiêu của paper. Bốn thành phần:

**1. Coverage (độ phủ) — submodular, cần tối đa:**

```
C_omega(X) = sum_t w_t sum_u d_{u,t} * [1 - prod_{e in X} (1 - a_{u,e})]
```

Trực giác: mỗi trạm phủ một vùng với xác suất a_{u,e}; phần `1 - tích các (1 - a)` là xác suất vùng u được phủ bởi ít nhất một trạm. Càng thêm trạm càng phủ tốt hơn nhưng lợi ích biên giảm dần (diminishing returns) — đó là tính submodular.

Ví dụ: vùng u có hai trạm ứng viên phủ với a = 0.6 và a = 0.5. Một mình trạm đầu phủ 0.6. Thêm trạm hai, độ phủ thành 1 - (1-0.6)(1-0.5) = 1 - 0.4*0.5 = 0.8, tức lợi ích biên của trạm hai chỉ còn 0.2 chứ không phải 0.5, vì phần chồng lấn đã được trạm đầu phủ rồi.

**2. Synergy (cộng hưởng tuyến) — supermodular, cần tối đa:**

```
Y(X) = sum_{e < f in X} b_{ef}
```

Thưởng cho các cặp trạm nằm trên cùng hành lang giao thông (zone kề nhau trong bán kính D_max). Đây là tính supermodular: giá trị của một trạm **tăng** khi đã có trạm bạn đồng hành gần đó.

**3. Grid overload (quá tải lưới) — phạt, supermodular:**

```
P_omega(X) = sum_{t,z} max(0, l_{z,t} + sum_{e in X_z} zeta_t * q_e - g_{z,t})^2
```

Với mỗi vùng lưới z và mỗi giờ t: nếu tải nền cộng công suất các trạm trong vùng vượt quá sức chứa g_{z,t}, phần vượt bị phạt theo bình phương. Bình phương khiến phạt tăng rất nhanh khi quá tải nặng, tức là code rất "sợ" vi phạm lớn.

**4. Unmet demand (nhu cầu không đáp ứng) — phạt, supermodular:**

```
U_omega(X) = sum_{t,u} max(0, d_{u,t} - s_{u,t}(X))
s_{u,t}(X) = min(d_{u,t}, serve_eff * zeta_t * sum_e a_{u,e} * q_e)
```

Lượng phục vụ được s bị chặn trên bởi nhu cầu thực và bởi năng lực khả dụng. Dạng `min` này chính là điều làm cho U thỏa mãn giả định submodularity của phần lý thuyết (Assumption 1). Nếu mô hình phục vụ thay đổi, kết quả lý thuyết về supermodularity của unmet demand cũng cần phát biểu lại giả định tương ứng.

**Tổng hợp thành hàm mục tiêu một kịch bản và hàm robust:**

```
F_omega = alpha*C + beta*Y - gamma*P - eta*U
F_rob   = sum_omega p_omega * F_omega - rho * CVaR_delta(L_omega)
L_omega = gamma*P_omega + eta*U_omega
```

`F_rob` là kỳ vọng của F trừ đi một số hạng CVaR đo phần đuôi xấu nhất của tổn thất L. Đặt `rho = 0` sẽ tắt phần nhận biết rủi ro (đây chính là baseline "Risk-Neutral SGTO").

**Thiết kế hiệu năng — lớp `IncrementalState`:**

Đây là chi tiết cài đặt (không phải đóng góp thuật toán, nên trong paper chỉ nên ghi ở đoạn "computational remarks"). Ý tưởng: cache trạng thái phụ thuộc lời giải (tích độ phủ Q_u, năng lực phục vụ S_u, tải lưới thêm vào mỗi vùng) và **xếp chồng toàn bộ dữ liệu kịch bản thành tensor**:

```
D    (M, T, U)   nhu cầu
WD   (M, U)      sum_t w_t * demand   (coverage gộp lại theo t!)
ZETA (M, T)      hồ sơ mức sử dụng
BG   (M, T, Z)   tải nền
GCAP (M, T, Z)   sức chứa lưới
```

Nhờ đó một lần đánh giá đầy đủ chỉ là vài phép numpy vector hóa thay vì vòng lặp Python qua từng kịch bản. Coverage đặc biệt gọn: vì w_t và demand tách được khỏi số hạng phụ thuộc lời giải, nó thu về một phép nhân ma trận duy nhất. Toàn bộ được kiểm chứng cho kết quả **trùng khớp bit-by-bit** với bản cài đặt cũ (giữ lại làm `reward_reference.py`). Kết quả: nhanh khoảng 6 lần, đủ để chạy được cấu hình ngân sách lớn mà trước đó bị timeout.

### B.7. Module `metrics.py` — các chỉ số của paper

Tính toàn bộ chỉ số trên tập test **giữ riêng** (không phải tập train hay validation). Danh sách chỉ số:

1. Global reward F_rob(X)
2. Tỉ lệ đáp ứng nhu cầu (demand fulfillment ratio)
3. Tỉ lệ phủ giờ cao điểm (peak-hour coverage)
4. Tổng quá tải lưới
5. Quá tải lưới lớn nhất
6. Tổng nhu cầu không đáp ứng
7. Điểm kết nối tuyến (route connectivity)
8. Tổng chi phí đầu tư
9. Reward trường hợp xấu nhất (worst-case)
10. Tổn thất CVaR
11. Thời gian chạy
12. Số lần đánh giá global reward

Có thêm hàm `describe_solution(inst, X)` để in lời giải ra dạng đọc được (zone id gốc, mức công suất).

### B.8. Thuật toán SGTO — `algorithms/sgto.py`

Mỗi vòng lặp k gồm **năm pha** đúng như Algorithm 1:

**Pha 1 — Lấy mẫu kịch bản.** Rút m kịch bản Omega_k từ tập train.

**Pha 2 — Ước lượng semi-gradient.** Tại điểm tuyến tính hóa hiện tại (lời giải X_cur), tính trọng số cho mỗi phần tử:

- Phần tử **chưa chọn** (e không thuộc X): trọng số thêm vào = trung bình lợi ích biên khi thêm e, qua m kịch bản.

  ```
  w_k^+(e) = (1/m) sum_omega [ F_omega(X_k ∪ {e}) - F_omega(X_k) ]
  ```

- Phần tử **đã chọn** (e thuộc X): trọng số gỡ bỏ = trung bình lợi ích biên khi gỡ e.

  ```
  w_k^-(e) = (1/m) sum_omega [ F_omega(X_k) - F_omega(X_k \ {e}) ]
  ```

Các trọng số này tạo ra một hàm thay thế modular (tuyến tính) quanh X_k — đây là ý tưởng "semi-gradient" của paper.

**Pha 3 — Bài toán knapsack modular.** Giải:

```
max sum_{e in X} w_k(e)   ràng buộc:  một mức mỗi zone (partition)  và  tổng chi phí <= B
```

bằng quy hoạch động theo nhóm zone (`solve_modular_knapsack`). Vì mỗi zone chỉ được chọn tối đa một mức, đây là knapsack theo nhóm chuẩn.

**Pha 4 — Tìm kiếm cục bộ.** Chạy one-exchange (và drop-and-refill, xem B.9) trên lời giải knapsack để tinh chỉnh.

**Pha 5 — Cổng validation.** So lời giải mới với lời giải tốt nhất trên tập Omega_val:

- Nếu **tốt hơn** ngưỡng epsilon: cập nhật X_best, tiếp tục từ đó.
- Nếu **không**: (bản paper) dừng; (bản mở rộng) nhiễu loạn X_best rồi tiếp tục, dừng sau `patience` lần từ chối liên tiếp.

Điểm mấu chốt về tính đúng đắn: **X_best chỉ bị thay khi validation cải thiện**, nên lời giải trả về luôn đơn điệu theo điểm validation bất kể có bật phần mở rộng hay không.

**Ba cờ ablation** (để tạo các baseline trong paper):

```python
use_exchange = False        # -> "SGTO without Local Exchange"
risk_aware   = False        # -> "Risk-Neutral SGTO" (rho = 0 ở mọi đánh giá)
risk_in_weights = False     # (AlgoConfig / --no-risk-in-weights) trọng số
                            # semi-gradient chỉ theo mean — hành vi cũ,
                            # dùng cho ablation; xem B.10 bài học 4
# patience=1, perturb_frac=0.0, final_polish=False  -> khôi phục đúng Algorithm 1
```

### B.9. Ba đóng góp thuật toán thật sự

Đây là phần cần phân biệt rạch ròi: chỉ **thay đổi thuật toán thật** mới đưa vào paper; tối ưu cài đặt (như vector hóa, cache) chỉ ghi ở đoạn computational remarks.

**1. Dual-rule greedy (greedy hai luật) — `algorithms/greedy.py`.**
Greedy theo tỉ lệ lợi-ích-trên-chi-phí một mình có thể tệ tùy ý khi một phần tử đắt lại đáng giá hơn nhiều phần tử rẻ. Cách khắc phục: chạy **song song** greedy theo tỉ lệ và greedy theo lợi ích tuyệt đối, rồi giữ lời giải tốt hơn. Đây là bảo đảm chuẩn từ tài liệu budgeted coverage.

Ví dụ: ngân sách 900. Một trạm lớn gain 300. Các trạm rẻ gain 25, chi phí 60 mỗi cái. Luật tỉ lệ chọn 15 trạm rẻ được 375; nhưng nếu độ phủ của chúng chồng lấn nặng thì trạm lớn thực tế lại đáng hơn. Chạy cả hai luật và giữ cái tốt hơn mới an toàn.

**2. Drop-and-refill (bỏ và nạp lại ngân sách) — `algorithms/local_search.py`.**
One-exchange thông thường chỉ đổi một-đổi-một, không bao giờ đổi được **một trạm lớn lấy nhiều trạm nhỏ**. Move mới này bỏ những phần tử yếu nhất và greedy tiêu lại phần ngân sách vừa giải phóng, chạy một lần sau khi các lượt exchange đã hội tụ. Đây là kiểu move 1-thành-nhiều mà exchange chuẩn thiếu.

**3. Iterated local search (SGTO như tìm kiếm cục bộ lặp).**
Bản gốc dừng ngay khi validation từ chối, dễ kẹt ở cực trị cục bộ. Bản mới, khi bị từ chối, **nhiễu loạn** lời giải tốt nhất (bỏ ngẫu nhiên một tỉ lệ phần tử rồi greedy refill trên mẫu mới) và tiếp tục, đồng thời **theo dõi lời giải validated tốt nhất** (best-so-far), và tùy chọn chạy một lượt polish cuối trên toàn bộ tập train.

Bằng chứng ở budget 12000 (một seed): greedy đạt gain 1446.2, SGTO theo luật paper 1449.3 (dừng ở vòng 1), SGTO mở rộng 1457.8 với quá tải lớn nhất giảm từ 2607 xuống 1841 kW. Các cải thiện được chấp nhận ở vòng 2/4/6/9 đều đến từ các lần khởi động lại nhờ nhiễu loạn — tức phần mở rộng thực sự có tác dụng.

### B.10. Bốn bài học đã trả giá (ba lỗi và một phát hiện)

Các mục này từng làm sai lệch kết quả, đã sửa. Ghi lại để tránh lặp và để trích dẫn giả định.

**Lỗi 1 — Kịch bản cắt lưới làm CVaR bị vô hiệu.**
Kịch bản cắt lưới scale tổng sức chứa lưới xuống **dưới cả tải nền**, khiến số hạng CVaR luôn bị chi phối bởi một khối tổn thất cố định và không còn phản ánh rủi ro thực. Số hạng CVaR rất nhạy với cách scale sức chứa lưới trong lúc dựng kịch bản; scale sai làm ẩn hoàn toàn phần phạt rủi ro.

**Lỗi 2 — Effectiveness không nhận biết tắc nghẽn.**
Do độ hiệu quả không phụ thuộc mức tắc nghẽn, thuật toán suy biến về nghiệm "toàn trạm nhỏ": coverage trên mỗi đồng chi phí luôn thắng, nên quyết định mức công suất trở nên tầm thường và phạt lưới không bao giờ ràng buộc (kiểm chứng: ngay cả gamma = 0 cũng cho quá tải bằng 0 ở budget 5000). Sau khi làm effectiveness nhận biết tắc nghẽn, nâng eta lên khoảng 5 bắt đầu trộn các trạm mức vừa vào lời giải.

**Lỗi 3 — Ánh xạ quận sai.**
TAZID chạy từ 102 đến 1173. Lấy **ký tự đầu** của TAZID gộp nhầm zone 1011 vào quận 1. Cách đúng là **chia nguyên cho 100** (`TAZID // 100`), cho ra **11 quận**. Đây là logic ánh xạ đúng, không phải tiền tố chuỗi.

**Bài học 4 — Trọng số semi-gradient phải cùng mục tiêu với cổng validation.**
Phiên bản đầu tính trọng số w_k bằng marginal của **trung bình** F_omega (đúng theo Eq. trong bản nháp paper) trong khi cổng validation chấp nhận theo mục tiêu **risk-aware**. Hệ quả: DP đề xuất nghiệm khá theo mean nhưng đuôi rủi ro nổ (trên chính mẫu train của nó: F_rob -3158 so với incumbent -2029), bị gate phủ quyết sạch, và `sgto_no_exchange` suy biến về đúng nghiệm greedy khởi tạo trong mọi cấu hình. Nó cũng giải thích vì sao bản risk-neutral từng thắng bản risk-aware trên chính chỉ số rủi ro: pipeline của risk-neutral thẳng hàng (trọng số mean, gate mean), còn bản risk-aware tự đánh nhau. Sửa: trọng số dùng cùng mục tiêu với gate (`risk_in_weights = True`, mặc định); marginal của mean + rho x CVaR tính được tự nhiên vì dạng Rockafellar-Uryasev là trung bình trên các kịch bản đuôi. Giá phải trả: bước tính trọng số đắt gấp đôi (~40% tổng thời gian SGTO). Định nghĩa w_k trong paper cần cập nhật tương ứng, và bản thân hiện tượng này đáng một nhận xét trong phần thực nghiệm.

### B.11. Các giả định cần trích dẫn trong paper

Những giả định này nằm trong `config.py` và `scenarios.py`, cần một dòng trong bảng dữ liệu của paper:

1. **Sức chứa lưới.** Mặc định là mô hình tổng hợp (`SyntheticGrid`, xem B.3c); tùy chọn `--grid ieee33` thay bằng hosting capacity theo điện áp của IEEE 33-bus với `ieee33_vmin = 0.90 pu` — bản thân v_min cũng là một giả định cần nêu.
2. **Chi phí mỗi mức dùng con số tham khảo từ tài liệu** (`CapacityLevelConfig`, 1 đơn vị = 1000 USD). Cần một dòng trích dẫn trong bảng dữ liệu. Đây cũng là điểm chưa nguồn nào phủ — chi phí lắp đặt phải dựa trên giả định từ tài liệu.
3. **Nhu cầu phục vụ** `s_{u,t} = min(d, serve_eff * zeta_t * sum a_ue q_e)`, thỏa mãn giả định submodularity (Assumption 1) của phần lý thuyết.
4. **Mức sử dụng** zeta_t theo hình dạng nhu cầu thành phố, rescale về [zeta_min, zeta_max].
5. **Paris — chuyển occupancy sang năng lượng** bằng một công suất trung bình mỗi plug (`plug_power_kw = 7.4`, mạng Belib' giai đoạn đó chủ yếu AC).
6. **Paris — khoảng cách đường bộ** khi không dùng OSMnx: great-circle nhân hệ số vòng 1.3 (giá trị đô thị phổ biến trong tài liệu routing).
7. **Paris — hệ số tăng trưởng nhu cầu** (`demand_growth`, khuyến nghị 3): dữ liệu 2020-2021 là thời đại dịch, utilization thấp bất thường; quy hoạch cho nhu cầu dự phóng cần một trích dẫn tăng trưởng EV châu Âu.

### B.12. Những việc còn mở (định hướng phát triển)

- **Độ ổn định của CVaR.** CVaR trên 12 kịch bản test xấp xỉ trung bình của ~1 kịch bản tệ nhất nên rất nhiễu (xem A.5). Hướng xử lý: tăng `n_test` lên 24-30 (thêm kịch bản nhiễu loạn) để CVaR trung bình trên >= 3 kịch bản, và/hoặc chạy >= 5 seed thuật toán cho bảng ablation.
- **Ngân sách Paris chưa ràng buộc** (tiêu tối đa 4620/7500 ở growth 3) trong khi khung NP-hardness dựa trên budgeted max coverage. Cân nhắc hạ budget mặc định Paris xuống ~3000 và kiểm tra separation còn giữ không.
- **Chốt trích dẫn cho `demand_growth = 3`** (dự phóng tăng trưởng EV châu Âu).
- **Baseline còn thiếu:** genetic algorithm, và một tham chiếu MILP cho instance nhỏ. Giao diện `Solver` trong `algorithms/base.py` đã sẵn sàng để cắm thêm.
- **Cập nhật paper:** định nghĩa w_k theo mục tiêu robust (bài học 4, B.10) và đoạn nhận xét thực nghiệm tương ứng; câu về heterogeneity của lưới IEEE 33-bus (B.3c).

---

## Phụ lục: quy trình làm việc khuyến nghị

1. Chạy `QUICK=1 bash run_all.sh` (hoặc `--quick` cho một lệnh đơn) trước để chắc pipeline thông.
2. Chạy `bash run_all.sh main` và kiểm tra thứ tự phương pháp khớp bảng A.5 (Random < SA < Greedy < SGTO-no-exch < Greedy+Exch < SGTO-RN < SGTO).
3. Gộp bảng bằng `make_tables.py`, vẽ hình bằng `make_plots.py`; với sweep rho chỉ dùng các cột độc lập với rho.
4. Khi sửa mô hình reward, luôn đối chiếu với `reward_reference.py` để đảm bảo không phá vỡ tính đúng đắn.
5. Kết quả đưa vào paper luôn chạy đủ (không `--quick`), ghi `--tag` rõ ràng, và giữ nguyên file json để tái lập.
4. Khi thêm đóng góp mới: tự hỏi đó là **thay đổi thuật toán** hay **tối ưu cài đặt**. Chỉ loại đầu mới đưa vào phần thuật toán của paper; loại sau ghi ở computational remarks.