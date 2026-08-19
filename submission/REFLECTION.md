# Reflection — Lab 19

**Tên:** Phan Văn Phương
**Cohort:** A20-K2
**Path đã chạy:** lite

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Trên 50 golden queries, hybrid (RRF, k=60) thắng trung bình: 78.6% vs BM25 77.8% và vector 73.2%. Theo loại query: **mixed** (câu ghép nhiều ý) hybrid thắng rõ nhất — 100% vs 97.0/98.5% — vì RRF gộp hai danh sách hạng, mỗi vế của câu được một bên bắt. **exact** (keyword chuẩn) BM25 và hybrid ngang 96.7% — lexical đã đủ, vector không thêm giá trị. **paraphrase** cả ba đều yếu (24–33%) vì bge-small là model tiếng Anh trên văn bản tiếng Việt — đây là giới hạn model, không phải thuật toán; đổi sang model đa ngữ (bge-m3) sẽ cải thiện.

Tôi **không dùng hybrid** khi: (1) query là keyword/technical chính xác — BM25 rẻ và nhanh hơn nhiều (P50 5.6ms vs hybrid 202ms ở NB3); (2) ứng dụng đòi hỏi latency thấp ổn định — mỗi query hybrid phải embed thêm vector; (3) câu hỏi đơn giản một ý định — theo NB6, classic retrieval đã đủ, agentic/hybrid chỉ đáng giá với câu nhiều phần.

---

## Điều ngạc nhiên nhất khi làm lab này

Điều bất ngờ nhất là các thất bại **im lặng**: post-filter sập recall về 0% ở filter chặt ~4% và latest-value join rò 98% số dòng mà hệ thống không báo lỗi nào — chỉ là trả kết quả sai trơn tru. Nếu không đo bằng ground truth đúng (PIT join / brute-force trên subset), không ai phát hiện ra.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`) — `bonus/agent.py` (HybridMemoryAgent) + `bonus/demo.py` + `bonus/ARCHITECTURE.md`
- [ ] Pair work với: _<tên đồng đội nếu có>_
