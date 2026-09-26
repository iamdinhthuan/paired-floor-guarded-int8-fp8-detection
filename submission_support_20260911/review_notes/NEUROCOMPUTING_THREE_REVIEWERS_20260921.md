# Phản biện độc lập bản Neurocomputing — 21/09/2026

## Phạm vi

Ba reviewer AI độc lập, sau đó đối chiếu bởi agent chính; đây không phải phản biện chính thức của journal. Đọc bản hiện tại trong `submission_support_20260911/source/`, không dùng các nhánh CVIU cũ. Bộ nộp được xét là main 14 trang và Supplementary Information 19 trang trong `submission_package/`.

Vòng này chỉ review: không sửa manuscript, không chạy inference/training mới, không push repository hoặc thay đổi Zenodo. Báo cáo này nằm ngoài thư mục nộp bài.

## Kết luận tổng hợp

**Có cơ sở nộp Neurocomputing sau một vòng sửa nhỏ. Chưa nên upload nguyên bộ hiện tại vì graphical abstract có chữ chồng nhau.** Trong phạm vi kiểm tra, chưa phát hiện lỗi khoa học làm vô hiệu kết quả trung tâm. Điều này không phải chứng nhận mọi implementation đều đúng hoặc bảo đảm được accept.

Reviewer 1 đề nghị sửa nhỏ; Reviewer 2 đánh giá đủ cơ sở về phương pháp; Reviewer 3 phát hiện một lỗi production nhỏ nhưng cần sửa trước upload. Rủi ro khoa học/biên tập lớn nhất còn lại là novelty mang tính tăng thêm và phạm vi thực nghiệm tập trung vào YOLO11/TensorRT, không phải phép tính interaction sai.

Hai việc nên hoàn thành trước upload: sửa chồng chữ graphical abstract và phân loại lại PD-Quant. Nhân cùng vòng sửa, làm mềm câu khẳng định verification, bỏ caption thừa, và đồng bộ câu mô tả archive ở S1. Không cần train thêm chỉ để xử lý các phát hiện này.

## Reviewer 1 — novelty, ý nghĩa và định vị journal

**Đề nghị: sửa nhỏ rồi nộp.** Đóng góp tăng thêm có thật nhưng không phải bước đột phá về quantization. Điểm mạnh là kết hợp điều kiện đầu vào/thực thi được kiểm soát, common-image uncertainty, clean-control rõ ràng và bằng chứng cho thấy các lựa chọn đánh giá thay đổi diễn giải như thế nào.

Introduction, bảng so sánh công trình trước, measurement contract và bảng hệ quả của từng thành phần đã làm rõ đóng góp này. Không có căn cứ bắt buộc bổ sung một quantizer mới, chạy SOTA PTQ sweep hoặc đạt kết quả tốt trên một họ detector thứ hai chỉ để bảo vệ những claim hiện tại.

Các lỗi nhỏ xác nhận được:

1. `source/main.tex:139`: PD-Quant bị xếp vào nhóm “detector-specific” gắn với classification/localization. Cần chuyển sang nhóm PTQ tổng quát. [Bài CVPR gốc](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_PD-Quant_Post-Training_Quantization_Based_on_Prediction_Difference_Metric_CVPR_2023_paper.html) trình bày PTQ tổng quát và các thực nghiệm classification, không phải phương pháp chuyên biệt cho detector.
2. `source/main.tex:141`: “These components are recorded and verified” rộng hơn bằng chứng được phân cấp ở Methods/Supplement. Nên viết theo nghĩa các thành phần được ghi nhận và kiểm tra trong phạm vi artifact còn lưu, đồng thời công khai phần chưa xác minh.
3. `source/main.tex:155`: caption Table 1 giải thích “Not established” nhưng bảng không còn dùng nhãn này. Bỏ câu giải thích thừa.

Bốn citation Neurocomputing mới phù hợp với nội dung abstract của nguồn xuất bản và metadata nhà xuất bản: [Attention Round](https://www.sciencedirect.com/science/article/pii/S0925231223011359), [Stabilized activation-scale estimation](https://www.sciencedirect.com/science/article/pii/S0925231223012432), [PLMQ](https://www.sciencedirect.com/science/article/pii/S0925231225016728), [QRT-DETR](https://www.sciencedirect.com/science/article/pii/S0925231225026293). Không coi chúng là baseline đã chạy. Không tuyên bố đã kiểm tra toàn văn của cả bốn bài.

Rủi ro còn lại là editor có đánh giá đóng góp quy trình này đủ significant hay không. Việc thêm citation đúng journal không tự giải quyết novelty. Các phép đo absolute/relative degradation của quantized YOLO đã có tiền lệ, được bài hiện tại thừa nhận qua [Karimov et al.](https://arxiv.org/html/2508.19600v3).

## Reviewer 2 — phương pháp, thống kê và bằng chứng

**Đề nghị: đủ cơ sở nộp về phương pháp; không phát hiện P0/P1 khoa học được chứng minh trong phạm vi kiểm tra.**

Các kiểm tra độc lập trên ledger/draw lưu sẵn:

| Đại lượng | Tính lại | Đối chiếu manuscript |
| --- | --- | --- |
| Holdout interaction | −0.549807 AP; interval [−0.784330, −0.290816] | Khớp |
| Holdout corrupted FP8–INT8 gap | +1.050221 AP; interval [0.910530, 1.155391] | Khớp |
| Controlled clean substitution | +0.0573566 AP; interval [−0.183926, +0.251766] | Khớp |
| Covariance audit | 127/144 cells tăng SD khi bỏ covariance; median ratio 1.123531 | Khớp |
| TT100K fixed-universe | Đủ 36 cells, 10.000 draws | Không còn là one-cell diagnostic |

Trong từng draw, identity `Gc = deltaE + Gclean` khớp với sai số số học tối đa khoảng 1.7e−18 trên thang AP gốc. Identity variance/covariance khớp đến độ chính xác floating point. Những phép kiểm tra này không chứng minh percentile interval có coverage 95% trong mọi population.

Các nguồn chính: `outputs/analysis/cviu_v4/holdout_synthesis/recovered/`, `outputs/analysis/cviu_v4/clean_control/summary.json`, `outputs/analysis/pairing_covariance_v1/`, và `outputs/bootstrap/tt100k_fixed_universe_full_v1_20260905/joint_macro.json`.

Bài phân biệt đúng: estimand với accuracy ranking; image uncertainty với seed/corruption uncertainty; original-source holdout với JPEG-95 exploratory control; recipe portability với bản chất datatype. S1 giải thích việc phục hồi clean AP vectors và phần thiếu metadata môi trường lịch sử, không giả vờ đó là chứng thực đầy đủ lịch sử execution.

Một bổ sung nhỏ nên làm: ở phần uncertainty/limitations, nối rõ việc KITTI chưa được xác minh scene/drive-disjoint với hệ quả thống kê: image-bootstrap hiện tại không xét khả năng phụ thuộc giữa ảnh cùng scene. Đây là điều chưa xác minh, không phải bằng chứng đã phát hiện leakage hoặc interval sai.

Reviewer không chạy lại toàn bộ AP từ predictions hay detector inference. Agent chính chạy ba nhóm regression tests: `test_v4_protocol_contract.py`, `test_v4_publication_tables.py`, `test_v4_holdout_table.py`; kết quả **44 passed, 6 skipped**. Sáu test bị skip vì môi trường Python hiện tại thiếu `pycocotools`; không được báo thành toàn bộ prediction-level tests đã pass trong vòng này.

## Reviewer 3 — production, source package và ranh giới tái lập

**Đề nghị: sửa nhỏ trước upload; có một lỗi bố cục đã xác nhận.**

### Lỗi phải sửa

`submission_package/04_graphical_abstract_Neurocomputing.pdf`, trang 1, khung bên phải: dòng “six-block means” chạm/chồng dòng “Clean FP8–INT8 gap”. Nguồn tại `submission_support_20260911/graphical_abstract_Neurocomputing.tex:20`, hai node đặt quá gần nhau. Agent chính render độc lập ở tỷ lệ 2x và xác nhận lỗi này. Sửa khoảng cách hoặc cho chú thích một dòng, rồi compile lại PDF và cập nhật cả hai source ZIP từ nguồn duy nhất. Không sửa trực tiếp PDF.

Contact sheet thu nhỏ trước đó không bộc lộ lỗi rõ; không được suy từ compile sạch rằng mọi graphical element đều không overlap.

### Các kiểm tra đạt trong phạm vi audit

- Đã kiểm tra contact sheets toàn bộ main 14 trang và S1 19 trang, chưa thấy clipping hoặc overlap tương tự. Đối chiếu raster PDF hiện tại với bản build trích từ flat ZIP khớp.
- Flat source ZIP có 49 file nguồn/assets, không kèm cache/build logs hoặc dữ liệu/token không liên quan. Thành phần ZIP khớp bản build được kiểm tra.
- Final logs không có unresolved references/citations hoặc overfull warnings. Vẫn có tám cảnh báo BibTeX thiếu trường pages ở tài liệu cũ; không báo “mọi warning bằng zero” và không tự bịa số trang.
- Titles, tác giả, declarations, funding không có tài trợ và DOI statement nhìn chung nhất quán. Agent chính kiểm tra SHA-256 của cả bảy file trong `submission_package/`: khớp `FINAL_VERIFICATION.json`; vòng review chưa thay chúng.
- Source/PDF identity không đồng nghĩa mọi phép đo đã được tái tạo từ inference. Archive, compact submission ZIP và prediction-level example là ba phạm vi khác nhau.

### Chỉnh wording không phải blocker

S1 §14 (`source/supplement.tex:739`) nói v2.2.0 “archives the revised manuscript”, trong khi main, đầu S1 và cover letter đã nói rõ archive chứa title/văn bản cũ. Có thể thay bằng “archives the preceding manuscript version and the experimental evidence” để tránh mơ hồ nếu đọc riêng section này. Đây không phải bằng chứng DOI sai hoặc archive thiếu toàn bộ evidence.

## Phân xử các phản biện dễ bị lặp lại sai

- “TT100K chỉ kiểm tra một cell”: không còn đúng; S1 đã có full grid.
- “56 ô đổi dấu là 56 phát hiện có ý nghĩa thống kê”: manuscript không khẳng định như vậy và đã báo độ lớn nhỏ của nhiều hiệu ứng.
- “Interaction âm nghĩa FP8 kém hơn INT8 trên ảnh corrupt”: sai estimand; holdout vẫn có corrupted gap dương.
- “Cross-family collapse chứng minh INT8 vốn kém hơn”: bài hiện tại đã bác bỏ cách diễn giải này.
- “DOI phải chứa đúng PDF mới thì toàn bộ evidence vô hiệu”: không đúng. Main công khai archive là phiên bản văn bản cũ với bằng chứng thực nghiệm không đổi. Cần nhất quán cách mô tả, không thay thế nội dung archive lịch sử.
- “Không có quantizer mới thì bắt buộc reject”: không phải tiêu chí khoa học hợp lệ cho mọi evaluation study. Tuy nhiên, mức significance vẫn là quyền đánh giá của editor.

## Giới hạn xác nhận journal

Các URL chính thức của Neurocomputing cho aims/scope và Guide for Authors trả HTTP 403 trong lượt kiểm tra. Không dùng website mô phỏng Elsevier hoặc hướng dẫn thứ cấp để chứng nhận quy định bắt buộc. Do đó, kết luận khoa học không đồng nghĩa xác nhận đầy đủ mọi yêu cầu hiện hành của cổng submission.

## Cập nhật sau review — đã sửa theo yêu cầu tác giả

Ngày 21/09/2026, lượt triển khai tiếp theo đã sửa cả sáu điểm: phân loại PD-Quant, mức độ verification, caption Table 1, hệ quả within-scene dependence trên uncertainty, wording về archive cũ ở S1, và chồng chữ graphical abstract. Bố cục workflow và take-home trong graphical abstract cũng được căn lại; các headline và bảng/figure thực nghiệm không thay đổi.

Main vẫn 14 trang, S1 vẫn 19 trang. Đã compile lại cả bốn tài liệu từ ZIP Overleaf và ZIP flat, đối chiếu từng trang PDF giữa hai đường build, xem contact sheets main/S1 và graphical abstract ở tỷ lệ 2x. Không thấy chồng chữ graphical abstract sau sửa. Kết quả build/hash chi tiết nằm trong `FINAL_VERIFICATION.json` cạnh báo cáo này. Bản trước sửa được giữ trong `PRE_REVIEW_FIXES_20260921.zip`, ngoài thư mục nộp bài. Nhận xét “chưa nên upload nguyên bộ hiện tại” ở phần trên mô tả bản trước sửa, không phải bộ đã export sau lượt này.

Không có training/inference mới, không thay số liệu, không cập nhật repository/Zenodo. Tác giả vẫn cần xác nhận đồng thuận/exclusive submission và kiểm tra PDF do portal journal tạo ra.

### Cập nhật wording theo tác giả

Declaration về AI trong writing process đã được thay nguyên văn bằng đoạn do tác giả cung cấp, chỉ nêu OpenAI's ChatGPT. Disclosure coding assistance trong Methods được giữ riêng. Tác giả đã được nhắc xác nhận đoạn declaration mới phản ánh đầy đủ công cụ thực tế đã dùng; lượt sửa này không xác minh lại lịch sử sử dụng công cụ.

Các đại từ `we/our/us` đã được loại khỏi phần nguồn main và supplement; abstract, contributions, mô tả thao tác và nhiều đoạn diễn giải đã được chuyển sang cấu trúc bị động. Declaration được giữ nguyên văn dù có các câu chủ động. Tên công trình trong bibliography không được sửa: cụm “Are we ready for autonomous driving?” là tên gốc của bài KITTI, không phải cách xưng của tác giả. Khi trích text từ PDF, một hit `us-` còn xuất hiện do ngắt dòng từ `using`, không phải đại từ `us`.

Đã đối chiếu multiset biểu thức toán inline và numeric tokens trong cả hai nguồn: không thay đổi. Các bảng và figure thực nghiệm cũng giữ nguyên byte. Đã rebuild cả hai source ZIP và PDF; main 14 trang, S1 19 trang. Backup trước lượt wording nằm trong `PRE_LANGUAGE_EDIT_20260921.zip`.

### Cập nhật khả năng đọc bảng theo yêu cầu tác giả

Table 5 được tách thành Panel A (bốn AP) và Panel B (clean gap, corrupted gap, interaction và các interval ở cột riêng). Không xếp chồng point estimate và interval trong từng ô; toàn bộ số lẻ được giữ như trước. Caption được rút gọn, định nghĩa và provenance được đặt thành notes có nhãn dưới bảng. Hai dataset được phân nhóm bằng khoảng cách hàng và dòng equal-block mean được phân biệt rõ.

Table 6 được mở theo chiều ngang trang, tách rõ primary/diagnostic và bổ sung cách đọc shift. Table 3 được bỏ thu nhỏ bằng resizebox, các ngưỡng size lặp lại được chuyển xuống note; scope exploratory được ghi rõ. Table 4 được dùng header hai dòng không resize; Table 7 được nhóm theo all-corruptions/severity-5 và theo dataset, bỏ phóng font quá lớn. Table 10 được ngăn nhóm exploratory/final, nghĩa của sign differences được giải thích; bảng runtime được dùng header hai dòng và ghi rõ median-of-ratios khác ratio-of-medians. Khoảng cách hàng được tăng ở các bảng văn bản dài. Ví dụ tính AP được giữ trong cùng một đoạn không ngắt trang; liên kết Table 5 Panel A/B được giải thích trong Results.

Đã chạy 34 regression tests cho bảng holdout/publication: 34 passed, không skip. Decimal values trong thân sáu bảng đổi layout và các hàng số liệu dataset được so với backup: không thay đổi. Các input hash/paired draws được kiểm tra bằng generator hiện có; không chạy inference mới. Main sau chỉnh là **15 trang**, S1 **19 trang**. Hai source ZIP được compile và đối chiếu PDF. Backup trước lượt này: `PRE_TABLE_LAYOUT_20260921.zip`.
