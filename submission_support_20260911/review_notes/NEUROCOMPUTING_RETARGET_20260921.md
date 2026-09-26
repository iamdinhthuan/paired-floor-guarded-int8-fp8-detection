# Chuyển sang Neurocomputing — 21/09/2026

Editor CVIU từ chối vì scope/readership và đề nghị cân nhắc Neurocomputing. Đây không phải chuyển bài tự động hoặc đảm bảo acceptance. Bản sửa định vị đóng góp là quy trình đánh giá và phân tích thực nghiệm neural inference sau quantization, không phải quantizer hoặc lý thuyết mới.

Title: **Separating Clean Fidelity from Corruption Sensitivity in Quantized Neural Detectors: A Paired Evaluation**.

Một nguồn chỉnh sửa: `../source/main.tex`, `../source/supplement.tex`, `../source/references.bib`. Không sửa các nhánh lịch sử `paper/` hay `Thuan_paper_3_CVIU_revised/`.

## Thay đổi

- Viết lại abstract/introduction/contributions/conclusion theo câu hỏi đánh giá compressed neural networks; vẫn giới hạn bằng chứng vào detector đã đo.
- Thêm related work và một hàng bảng prior art; bốn nghiên cứu mới không được trình bày như baseline đã chạy.
- Thêm discussion hướng áp dụng cho calibration/rounding/mixed-precision. QRT-DETR chỉ là bối cảnh recipe portability, không giải thích được nguyên nhân collapse của engine hiện tại.
- Đồng bộ title main/S1/PDF metadata/cover letter; sáu keywords và năm highlights dưới 85 ký tự.
- Cover letter nói rõ CVIU từ chối vì scope; không gọi là endorsement/transfer đã được duyệt.
- AI declaration đồng bộ Methods và lịch sử dùng ChatGPT, Codex, Claude Code. Công cụ không là tác giả hoặc CRediT contributor; tác giả phải duyệt khai báo trước nộp.
- Giữ dữ liệu, bảng generated, hình khoa học và bootstrap schedules; không train/inference mới.
- DOI giữ nguyên; evidence archive v2.2.0 có title/filename cũ. Không tuyên bố hashes cũ bind PDF mới, không push GitHub hoặc Zenodo.

## Bốn citation Neurocomputing

Metadata đối chiếu Crossref (publisher-deposited); nội dung đối chiếu abstract/highlights của publisher, không suy diễn ngoài phần truy cập được.

| Bài | DOI | Ngữ cảnh |
|---|---|---|
| Diao et al., Attention Round, 565 (2024), 127012 | https://doi.org/10.1016/j.neucom.2023.127012 | Rounding và mixed precision |
| Hao et al., Stabilized activation scale estimation, 569 (2024), 127120 | https://doi.org/10.1016/j.neucom.2023.127120 | Activation-scale calibration |
| Feng et al., PLMQ, 651 (2025), 131000 | https://doi.org/10.1016/j.neucom.2025.131000 | Piecewise/mixed-precision quantization |
| Huo et al., QRT-DETR, 661 (2026), 131957 | https://doi.org/10.1016/j.neucom.2025.131957 | Detector-specific PTQ |

Năm trong DOI có thể khác năm volume. Các references nền tảng hiện có được giữ, không xóa chỉ vì thuộc journal khác.

## Chưa xác minh được toàn bộ guide

Trang chính thức trả HTTP 403 khi kiểm tra:
https://www.sciencedirect.com/journal/neurocomputing/about/aims-and-scope
https://www.sciencedirect.com/journal/neurocomputing/publish/guide-for-authors

Giữ Elsevier CAS, abstract dưới 250 từ, sáu keywords, năm highlights dưới 85 ký tự như ràng buộc đóng gói thận trọng; không khẳng định đã xác minh toàn bộ yêu cầu hiện hành. Tác giả cần đối chiếu article type, template/page limits, anonymization, định dạng highlights, graphical abstract, biography/photos nếu yêu cầu, reviewer suggestions và declarations trên cổng chính thức. Đi từ trang publisher, không dùng link giả mạo Editorial Manager trên kết quả tìm kiếm.

## Trước submit

1. Toàn bộ tác giả duyệt bản cuối, AI disclosure, CRediT, affiliation và cover letter.
2. Xác nhận CVIU đã kết thúc xử lý, không nộp đồng thời; không tự điền manuscript ID chưa được cung cấp.
3. Dùng bộ Neurocomputing gọn trong `submission_package/`, không upload cả thư mục hỗ trợ.
4. Nếu cần archive bind chính xác PDF mới, phải tạo version mới khi được cho phép; DOI hiện tại vẫn là experimental release đã xuất bản.
5. Kiểm tra PDF do portal dựng trước khi xác nhận submit.

Đổi framing và citation không tự tạo novelty mới. Rủi ro novelty của evaluation paper và giới hạn primary YOLO vẫn được trình bày thẳng.
