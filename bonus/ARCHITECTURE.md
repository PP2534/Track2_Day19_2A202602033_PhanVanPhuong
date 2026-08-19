# Bonus — HybridMemoryAgent: Thiết Kế Bộ Nhớ Cho Trợ Lý AI Tiếng Việt

**Contributor:** Phan Văn Phương · **Repo:** [Day19-Track2-VectorFeatureStore-Lab](https://github.com/PP2534/Track2_Day19_2A202602033_PhanVanPhuong)
**POC:** `bonus/agent.py` + `bonus/demo.py` — chạy bằng `python bonus/demo.py` (exit 0)

---

## 1. Tổng quan

Trợ lý cá nhân cho người dùng Việt Nam cần **ba loại bộ nhớ**, và mỗi loại có
vòng đời + cơ chế truy xuất riêng. Không ép cả ba vào một hệ thống:

| Loại | Ví dụ | Lưu trữ | Vòng đời |
|---|---|---|---|
| **Episodic memory** | tài liệu đã đọc, ghi chú, hội thoại | Vector store (Qdrant) | giờ → tháng |
| **Stable profile** | ngôn ngữ ưa thích, topic affinity, tốc độ đọc | Feature store (Feast) | tuần → tháng |
| **Recent activity** | query 1 giờ qua, tần suất đọc đêm | Streaming feature view | phút → giờ |

Sơ đồ kiến trúc:

```
                          ┌──────────────────────────────┐
                          │         user action          │
                          │  (đọc doc / hỏi / ghi chú)    │
                          └──────────────┬───────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
        ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────┐
        │ EPISODIC (Qdrant)│  │ PROFILE (Feast)  │  │ ACTIVITY (stream)  │
        │ collection       │  │ user_profile_    │  │ query_velocity_    │
        │ "bonus_memory"   │  │ features (batch) │  │ features (1h TTL)  │
        │ chunk + embed +  │  │ materialize      │  │ materialize-       │
        │ upsert (user_id) │  │ daily            │  │ incremental 5 min  │
        └────────┬─────────┘  └────────┬─────────┘  └─────────┬──────────┘
                 │                     │                       │
                 │   recall(query): hybrid RRF (vector+keyword)│
                 └─────────────┬───────┴───────────┬───────────┘
                               ▼                   ▼
                    ┌─────────────────────────────────────────┐
                    │  HybridMemoryAgent._assemble_context()  │
                    │  "User likes <affinity>, reads <wpm>wpm,│
                    │   <qph> queries/h. Top memories: <top-3>"│
                    └──────────────────┬──────────────────────┘
                                       ▼
                              ┌─────────────────┐
                              │  LLM (production)│ → trả lời
                              │  (POC: in context│   dựa trên context
                              │   string, no LLM)│
                              └─────────────────┘
```

**Data flow:** một hành động của user rẽ thành 3 nhánh ghi (vector / profile /
activity). Lúc recall, ba nhánh đọc hội tụ thành **một context string** — thứ
LLM nhìn thấy. POC không gọi LLM thật; `recall()` trả đúng context đó để thấy
rõ phần nào của câu trả lời đến từ loại memory nào.

---

## 2. Quyết định kiến trúc (3 quyết định, mỗi cái có tradeoff)

### Quyết định 1 — Chunking: cắt theo **semantic break** (đoạn văn → câu), không theo số token cố định

Có 3 lựa chọn: **(a)** cắt theo số token cố định (200–500 token), **(b)** mỗi
message/hội thoại là một chunk, **(c)** semantic break (đoạn văn, hết câu).

- **Chọn (c).** `_chunk()` tách đoạn văn trước, đoạn quá dài (>400 ký tự) mới
  cắt tiếp ở biên câu.
- **Tradeoff vs (a):** token-fixed dễ index đều (chi phí predictable) nhưng cắt
  ngang giữa câu làm *retrieval quality* tệ — query không khớp nửa chunk nào.
  Semantic break đắt hơn chút (chunk không đều) nhưng mỗi chunk là một ý trọn
  vẹn, embedding khớp hơn.
- **Tradeoff vs (b):** per-message quá ngắn (embedding nghèo), per-conversation
  quá dài (tràn context window, embedding "pha loãng"). Chunk theo ý nằm giữa —
  tối ưu *context window* vì chỉ đưa top-3 chunk liên quan, không đưa cả hội thoại.

### Quyết định 2 — Feature schema: **tabular features** làm trục chính, không dùng embedding features cho profile

- **Chọn pattern tabular** trong Feast: `user_profile_features` (topic_affinity
  STRING, preferred_language STRING, reading_speed_wpm INT64, TTL 30 ngày, nguồn
  batch daily) + `query_velocity_features` (queries_last_hour, TTL 1 giờ, nguồn
  streaming).
- **Tradeoff vs embedding features** (lưu vector latent-preference của user):
  embedding bắt được sở thích ngầm (implicit) mà tabular bỏ sót, nhưng mỗi
  profile vector cần **re-index khi user đổi ý** — chu kỳ cập nhật không
  khớp với nhịp "đọc xong một tài liệu". Tabular đơn giản, **debug được**
  (nhìn thẳng giá trị feature), và phù hợp quy mô cá nhân: 100 user thì vài
  trường STRING rẻ hơn hẳn một cụm vector 384-d. Embedding features chỉ đáng
  giá ở quy mô triệu user có hành vi dày.

### Quyết định 3 — Freshness: **một tầng bất đồng bộ, ba mức làm mới**

Một cấu hình TTL không thể phục vụ cả ba câu hỏi khác nhau:

| Use case | Yêu cầu | Cơ chế |
|---|---|---|
| "Tôi vừa đọc gì đây?" (ngay sau khi đọc) | **sub-second** | stream Push API → ép ngay chunk mới vào Qdrant, `remember()` trả về trước khi user kịp hỏi |
| "Tôi đang quan tâm gì?" | **5 min** | batch refresh `query_velocity` bằng `feast materialize-incremental` mỗi 5 phút |
| "Hồ sơ của tôi thế nào?" | **daily** | batch `user_profile` làm mới mỗi đêm (profile ít đổi) |

- **Tradeoff:** càng nhanh càng đắt + phức tạp (Push API, offset commit, exactly-
  once). Chỉ *episodic* cần sub-second vì nó là thứ người dùng vừa tạo ra bằng
  tay; profile 30 ngày không nên trả tiền cho streaming. Đây chính là lý do
  POC **tách** vector store khỏi feature store (xem §3).

---

## 3. Lựa chọn bị loại bỏ (rejected alternative)

**"Lưu episodic memory ngay trong feature store dưới dạng embedding feature view"** —
tôi đã cân nhắc phương án này vì nó gộp được mọi thứ vào một Feast repo (đơn
giản hạ tầng). Nhưng **loại bỏ** vì chu kỳ re-index khác hẳn nhau: memory mới
xuất hiện **từng giờ** còn profile thay đổi **theo tuần**. Embedding feature
view buộc cả hai đi cùng một nhịp materialize — hoặc profile bị đẩy lên tần
suất phí phạm, hoặc memory phải chờ batch rồi mới recall được. Tách thành
Qdrant (vector, sub-second upsert) + Feast (tabular, daily) cho phép mỗi loại
đi đúng nhịp của nó.

---

## 4. Vietnamese-context considerations

- **Code-switching (vi/en mix):** người Việt hỏi "tutorial về cloud security"
  lẫn "tài liệu bảo mật đám mây". POC giữ cả hai bằng **hybrid RRF**: keyword
  bắt nghĩa đen (vi), vector bắt nghĩa gần (en) — đúng bài học NB2.
- **Phonetic typo ("hạ tầng" gõ thành "ha tang"):** whitespace tokenizer chết
  ngay. Nâng cấp đúng là `pyvi`/`underthesea` để tách từ tiếng Việt (còn
  tokenizer tách theo dấu cách gán sai nghĩa cho "hạ tầng" vs "hà tầng"). POC
  dùng whitespace vì quy mô 5 memories — ghi nhận là giới hạn, không phải lựa
  chọn.
- **Quyền riêng tư (Nghị định 13/2023/NĐ-CP):** memory là dữ liệu cá nhân —
  mọi point phải có `user_id` payload và recall **bắt buộc** filter theo nó
  (bài học NB7/OWASP LLM08). POC làm đúng điều này.

---

## 5. Liên kết với lab concepts

| Lab concept | Ở đâu trong bonus |
|---|---|
| **RRF (NB2)** | `_hybrid_top_k()` fuse vector rank + keyword rank, k=60 |
| **Filtered search / namespace (NB5, NB7)** | mọi query_point đều có `query_filter` theo `user_id` |
| **Feast online lookup (NB4)** | `_profile()` gọi `get_online_features` cho affinity/language/wpm/queries_last_hour |
| **PIT join (NB4, NB8)** | memory dùng `ts` riêng; profile/activity tách khỏi episodic để không bao giờ dùng giá trị tương lai |
| **TTL (NB4, NB7)** | profile TTL 30 ngày, activity TTL 1 giờ, episodic cần decay riêng (§6) |

---

## 6. What this POC doesn't handle yet

- **Privacy isolation thật:** filter theo `user_id` là isolation *mềm* — chưa có
  mã hoá per-user hay per-tenant collection; một bug quên filter vẫn rò như NB7.
- **Encryption at rest** cho vector store và online store chưa bật.
- **CRUD trên memory:** chưa có delete/sửa một chunk đã lưu (chỉ upsert mới).
- **Memory decay:** chưa có TTL/archive cho episodic ("30 ngày không truy cập →
  archive") — profile có TTL, episodic thì chưa.
- **Multi-device sync / vector DB server thật:** dùng Qdrant in-memory của lite
  path, không phải deployment có payload index thật.
- **LLM generation:** recall() dừng ở context string; phần sinh câu trả lời và
  re-ranking theo profile (boost topic_affinity) là bước tiếp theo.

---

## 7. Vibe coding workflow log (optional)

Prompt hiệu quả nhất: *"tạo class HybridMemoryAgent dùng Qdrant in-memory +
app.embeddings.Embedder, recall phải filter theo user_id và fuse vector+keyword
bằng RRF k=60"* — code chạy đúng ngay vòng đầu vì pattern đã có sẵn trong NB2.
Prompt fail: *"so sánh agentic retrieval với single-shot"* trong bonus context —
không có ground truth nên kết quả vô nghĩa; phải tự định nghĩa đúng câu hỏi
trước khi hỏi AI (bài học NB6 lặp lại).
