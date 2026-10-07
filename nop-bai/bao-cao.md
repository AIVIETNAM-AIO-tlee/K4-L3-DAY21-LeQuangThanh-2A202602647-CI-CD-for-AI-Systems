# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

<!--
HƯỚNG DẪN - đọc rồi XÓA TOÀN BỘ các khối chú thích này sau khi điền xong:

  - Giới hạn: KHÔNG QUÁ 1 TRANG A4, tương đương khoảng 450 - 550 từ nội dung.
  - Chỉ điền vào các chỗ ___ và các ô trong bảng. Không thêm mục mới.
  - Viết bằng câu hoàn chỉnh, không gạch đầu dòng cụt lủn.
  - Kiểm tra độ dài sau khi đã xóa hết chú thích:
        wc -w nop-bai/bao-cao.md
    và xem trước bản in bằng cách mở file trên GitHub rồi Ctrl+P / Cmd+P.
-->

| | |
|---|---|
| Họ và tên | Lê Quang Thành |
| MSSV | 2A202602647 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/AIVIETNAM-AIO-tlee/K4-L3-DAY21-LeQuangThanh-2A202602647-CI-CD-for-AI-Systems |
| Ngày nộp | 07/10/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

<!-- Khoảng 120 - 150 từ. Điền kết quả thật từ MLflow UI ở Bước 1, tối thiểu 3 lần chạy. -->

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Bộ siêu tham số này cho f1-score cao nhất trong 3 lần chạy thử nghiệm, đạt 0.7149, cao hơn đáng kể so với lần 1 (0.7109) và lần 2 (0.6051). Mặc dù lần 1 có accuracy cao nhất là 0.8780 nhưng không chênh lệch quá nhiều so với lần 3, nhưng f1-score thì cho ra đánh giá trực quan hơn về hiệu năng của mô hình vì thấy accuracy có thể gây hiểu nhầm. Quan sát thấy khi tăng số lượng cây (n_estimators) từ 50 lên 200, f1-score tăng lên, nhưng khi giảm learning_rate xuống 0.05, mô hình không cải thiện hiệu năng.

<!--
Trả lời trong phần Lý do:
  - Vì sao bộ này tốt hơn các bộ còn lại (dựa trên f1_score, không phải accuracy)?
  - Lần chạy có accuracy cao nhất có trùng với lần có f1_score cao nhất không?
    Nếu không, điều đó nói lên điều gì?
  - Bạn quan sát thấy đánh đổi nào giữa n_estimators và learning_rate?
-->

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập dữ liệu Census Income có sự mất cân bằng lớp rõ rệt khi lớp dương (thu nhập > 50K USD) chỉ chiếm khoảng 24,8% tổng số mẫu. Nếu một mô hình đơn giản luôn dự đoán nhãn "thu nhập thấp" cho mọi đối tượng, độ chính xác (accuracy) vẫn đạt tới 75,2%, tạo ra ảo tưởng về một mô hình hiệu quả nhưng thực chất hoàn toàn vô dụng vì không phát hiện được bất kỳ trường hợp thu nhập cao nào. Điểm F1-score của lớp dương (trung bình điều hòa giữa precision và recall) đo lường chính xác năng lực nhận diện lớp thiểu số này, phản ánh cân bằng giữa việc tránh dự đoán nhầm và tránh bỏ sót. Do đó, ngưỡng chất lượng bắt buộc phải tính trực tiếp trên lớp dương mà không sử dụng `average="weighted"` hay `average="macro"`, bởi các phương pháp lấy trung bình này sẽ bị lớp đa số lấn át và làm mất đi ý nghĩa đánh giá thực tế.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

<!-- Nêu 2 - 3 khó khăn thật, mỗi ô một câu ngắn. -->

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| Lỗi cài đặt gói `scikit-learn==1.4.2` khi khởi tạo môi trường trên máy. | Môi trường mặc định dùng Python 3.13 chưa có bản wheel pre-built, việc build từ source yêu cầu numpy 2.0.0rc1 không tương thích. | Tạo lại môi trường ảo `.venv` bằng phiên bản Python 3.10 theo đúng chuẩn bài lab và CI/CD. |
| Mô hình ở lần chạy 2 có accuracy tương đối cao (84.6%) nhưng F1-score lại tụt dốc (0.6051). | Dữ liệu bị mất cân bằng lớp khiến accuracy không phản ánh đúng chất lượng phân loại của lớp thiểu số. | Sử dụng MLflow UI để đối chiếu và quyết định lựa chọn bộ tham số dựa trên F1-score thay vì accuracy. |
| Nguy cơ phát sinh lỗi đường dẫn khi tự động xuất file model và report. | Thư mục `outputs/` và `models/` có thể chưa tồn tại trước khi chạy script huấn luyện. | Sử dụng `os.makedirs(..., exist_ok=True)` trong script `src/train.py` trước khi lưu file. |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

<!-- Lấy số liệu từ bảng ở mục 3.6 của tasks/buoc-3.md. -->

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7149 | 0.8740 |
| Bước 3 (thêm `train_batch2`) | 0.7354 | 0.8820 |

**Nhận xét:** Khi bổ sung thêm 22.361 mẫu từ `train_batch2`, tập dữ liệu huấn luyện tăng gấp đôi (đạt 44.722 mẫu), giúp F1-score của mô hình trên tập holdout tăng từ 0.7149 lên 0.7354 (tăng ~2.05%) và accuracy tăng từ 0.8740 lên 0.8820. Việc bổ sung lượng lớn dữ liệu cùng phân phối giúp mô hình Gradient Boosting khái quát hóa ranh giới quyết định cho lớp thiểu số tốt hơn mà không đánh đổi độ chính xác tổng thể. Quan trọng nhất, toàn bộ chu trình từ thêm dữ liệu, phiên bản hóa bằng DVC đến huấn luyện lại và tái triển khai lên server suy luận đã được thực hiện hoàn toàn tự động bởi pipeline CI/CD mà không cần bất kỳ can thiệp thủ công nào.

<!--
Một câu trả lời trung thực kiểu "f1 giảm 0,01 vì dữ liệu mới cùng phân phối, không mang
thêm thông tin mới" được đánh giá cao hơn kết luận sai rằng thêm dữ liệu luôn tốt hơn.
-->

---
