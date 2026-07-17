# Hướng dẫn phần code dự án SGTO (Quy hoạch trạm sạc EV)

Tài liệu này mô tả toàn bộ mã nguồn thí nghiệm cho bài toán quy hoạch trạm sạc xe điện dựa trên rủi ro, thuật toán **SGTO (Scenario-Based Global Trajectory Optimization)**, chạy trên dữ liệu thực **UrbanEV (Thâm Quyến)**.

Tài liệu gồm hai phần:

- Phần A dành cho người muốn **chạy lại thí nghiệm**: cài đặt, câu lệnh, đọc kết quả.
- Phần B dành cho người muốn **hiểu và phát triển tiếp code**: kiến trúc, từng module, công thức hàm mục tiêu, chi tiết thuật toán.

---

## PHẦN A. CHẠY LẠI THÍ NGHIỆM

### A.1. Yêu cầu môi trường

- Python 3.10 trở lên
- Hai thư viện: `numpy`, `pandas`
- Kết nối internet để tải dữ liệu UrbanEV (một lần)

Không cần GPU. Toàn bộ tính toán là numpy trên CPU.

### A.2. Cài đặt

```bash
pip install numpy pandas
git clone --depth 1 https://github.com/IntelligentSystemsLab/UrbanEV.git
```

Thư mục `UrbanEV/` phải nằm cạnh code (hoặc trỏ đường dẫn trong `config.py`). Loader chỉ đọc các file csv trong `UrbanEV/data/`.

### A.3. Các lệnh chạy

```bash
python run_experiment.py --quick                        # chạy thử nhanh, khoảng 2 phút
python run_experiment.py                                # chạy đầy đủ
python run_experiment.py --methods sgto cost_aware_greedy   # chỉ chạy một vài phương pháp
```

- `--quick` dùng ít kịch bản (scenario) và ít vòng lặp, chỉ để kiểm tra pipeline có chạy trơn tru không. Kết quả ở chế độ này **không dùng để báo cáo trong paper**.
- Chạy đầy đủ (không cờ) mới cho ra số liệu dùng được.
- `--methods` nhận danh sách tên phương pháp để chạy chọn lọc thay vì tất cả.

### A.4. Đọc kết quả

Kết quả ghi vào `results/results.json`. Với mỗi phương pháp, file chứa:

- Bảng **metrics** (toàn bộ chỉ số của paper, xem mục B.7)
- **Lời giải**: danh sách cặp `(zone id, level)` — tức là đặt trạm ở vùng nào, mức công suất nào
- **Lịch sử vòng lặp** của SGTO (giá trị validation ở từng vòng, vòng nào được chấp nhận)

Ví dụ cách đọc nhanh bằng Python:

```python
import json
r = json.load(open("results/results.json"))
for name in r:
    m = r[name]["metrics"]
    print(name, round(m["F_rob"], 2), round(m["F_rob_gain"], 2))
```

Một điểm cần lưu ý khi đọc số: `F_rob` mang một hằng số âm rất lớn vì luôn có một khối nhu cầu không thể phục vụ (unmet demand) ngay cả với lời giải tốt nhất. Vì vậy code bổ sung chỉ số `F_rob_gain` (phần cải thiện so với lời giải rỗng) để so sánh giữa các phương pháp cho trực quan hơn.

### A.5. Kết quả tham chiếu (một seed, để đối chiếu)

Sau khi sửa các lỗi (xem mục B.9), thứ tự các phương pháp khớp với kỳ vọng của paper:

| Phương pháp | F_rob |
|---|---|
| Random Search | khoảng -3652 |
| Cost-Aware Greedy | khoảng -3448 |
| Greedy + One-Exchange | khoảng -3442 |
| SGTO (đầy đủ) | khoảng -3400 |

Với SGTO: tỉ lệ đáp ứng nhu cầu (fulfillment ratio) khoảng 0.144 đến 0.153, quá tải lưới lớn nhất giảm còn 640 đến 807 kW. Đây là số một seed, dùng để bạn biết mình đang ở đúng khoảng giá trị.

Lưu ý: ở ngân sách nhỏ (budget 5000) bài toán gần như bão hòa — tìm kiếm mạnh và cả simulated annealing độc lập đều không vượt qua được SGTO, nên khác biệt giữa các phương pháp ở đó nhỏ một cách tự nhiên. Để bảng kết quả chính có ý nghĩa, nên chạy thêm ít nhất một cấu hình ngân sách lớn hơn (ví dụ budget 12000).

---

## PHẦN B. HIỂU VÀ PHÁT TRIỂN CODE

### B.1. Triết lý kiến trúc

Nguyên tắc cốt lõi: **thuật toán chỉ nhìn thấy một đối tượng `ProblemInstance`, không bao giờ chạm vào file thô.** Mọi việc đọc file, biến đổi dữ liệu đều nằm trong tầng `data_processing/`. Nhờ đó, muốn thêm dữ liệu Paris sau này chỉ cần viết **một loader mới**, không phải sửa thuật toán.

Chuỗi phụ thuộc một chiều:

```
file csv thô  ->  data_processing/  ->  ProblemInstance  ->  algorithms/  ->  metrics.py
```

### B.2. Cấu trúc thư mục

```
config.py                  toàn bộ giả định và siêu tham số gom về một chỗ
data_processing/
  urbanev.py               đọc csv UrbanEV, dựng ProblemInstance
  scenarios.py             dựng tập kịch bản train/val/test từ các ngày thực
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
```

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
use_exchange = False   # -> "SGTO without Local Exchange"
risk_aware   = False   # -> "Risk-Neutral SGTO" (rho = 0 ở mọi đánh giá)
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

### B.10. Ba lỗi đã sửa (bài học quan trọng)

Ba lỗi này từng làm sai lệch kết quả, đã sửa. Ghi lại để tránh lặp và để trích dẫn giả định.

**Lỗi 1 — Kịch bản cắt lưới làm CVaR bị vô hiệu.**
Kịch bản cắt lưới scale tổng sức chứa lưới xuống **dưới cả tải nền**, khiến số hạng CVaR luôn bị chi phối bởi một khối tổn thất cố định và không còn phản ánh rủi ro thực. Số hạng CVaR rất nhạy với cách scale sức chứa lưới trong lúc dựng kịch bản; scale sai làm ẩn hoàn toàn phần phạt rủi ro.

**Lỗi 2 — Effectiveness không nhận biết tắc nghẽn.**
Do độ hiệu quả không phụ thuộc mức tắc nghẽn, thuật toán suy biến về nghiệm "toàn trạm nhỏ": coverage trên mỗi đồng chi phí luôn thắng, nên quyết định mức công suất trở nên tầm thường và phạt lưới không bao giờ ràng buộc (kiểm chứng: ngay cả gamma = 0 cũng cho quá tải bằng 0 ở budget 5000). Sau khi làm effectiveness nhận biết tắc nghẽn, nâng eta lên khoảng 5 bắt đầu trộn các trạm mức vừa vào lời giải.

**Lỗi 3 — Ánh xạ quận sai.**
TAZID chạy từ 102 đến 1173. Lấy **ký tự đầu** của TAZID gộp nhầm zone 1011 vào quận 1. Cách đúng là **chia nguyên cho 100** (`TAZID // 100`), cho ra **11 quận**. Đây là logic ánh xạ đúng, không phải tiền tố chuỗi.

### B.11. Các giả định cần trích dẫn trong paper

Những giả định này nằm trong `config.py` và `scenarios.py`, cần một dòng trong bảng dữ liệu của paper:

1. **Sức chứa lưới là tổng hợp (synthetic).** Tải nền mỗi quận theo một đường cong feeder ngày chuẩn, scale theo nhu cầu sạc của quận; `g_{z,t} = margin * (đỉnh tải nền + tải trạm tham chiếu)`. Có thể thay bằng IEEE 33-bus qua pandapower sau; các hook đã có sẵn trong `scenarios.py`.
2. **Chi phí mỗi mức dùng con số tham khảo từ tài liệu** (`CapacityLevelConfig`, 1 đơn vị = 1000 USD). Cần một dòng trích dẫn trong bảng dữ liệu. Đây cũng là điểm chưa nguồn nào phủ — chi phí lắp đặt phải dựa trên giả định từ tài liệu.
3. **Nhu cầu phục vụ** `s_{u,t} = min(d, serve_eff * zeta_t * sum a_ue q_e)`, thỏa mãn giả định submodularity (Assumption 1) của phần lý thuyết.
4. **Mức sử dụng** zeta_t theo hình dạng nhu cầu thành phố, rescale về [zeta_min, zeta_max].

### B.12. Những việc còn mở (định hướng phát triển)

- **Ablation rho (risk-aware vs risk-neutral).** Hiện ở rho 0.3 hai biến thể hội tụ về cùng lời giải vì CVaR vẫn bị khối unmet demand chi phối. Cần chọn một trong: rho lớn hơn, định nghĩa loss trừ đi baseline lời giải rỗng, hoặc kịch bản lưới chặt hơn. Quyết định này ảnh hưởng công thức trong paper nên cần người chọn, không nên tự đổi L_omega.
- **Baseline còn thiếu:** genetic algorithm, và một tham chiếu MILP cho instance nhỏ. Giao diện `Solver` trong `algorithms/base.py` đã sẵn sàng để cắm thêm.
- **Tuyến đường:** hiện dùng zone adjacency; nếu reviewer muốn tuyến đường thực, thay bằng hành lang OSMnx.
- **Instance thứ hai:** thêm loader Paris (Smarter Mobility) — chỉ cần viết một loader mới nhờ kiến trúc tách biệt.

---

## Phụ lục: quy trình làm việc khuyến nghị

1. Chạy `--quick` trước để chắc pipeline thông.
2. Chạy đầy đủ, kiểm tra thứ tự phương pháp trong `results.json` có khớp mục A.5 không (Random < Greedy < Greedy+Exchange < SGTO).
3. Khi sửa mô hình reward, luôn đối chiếu với `reward_reference.py` để đảm bảo không phá vỡ tính đúng đắn.
4. Khi thêm đóng góp mới: tự hỏi đó là **thay đổi thuật toán** hay **tối ưu cài đặt**. Chỉ loại đầu mới đưa vào phần thuật toán của paper; loại sau ghi ở computational remarks.