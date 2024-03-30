# Image-Restoration

## Data mining
### data_dryrun
Mỗi file đều là numpy có shape (100, 271, 271). Chúng đại diện cho 100 frame ảnh ở dạng 12-bit integer raw format (do đó có thể xem 1 file ứng với một video).

"Mục tiêu của người tham gia là tái tạo một khung ảnh chất lượng cao duy nhất từ mỗi trong 100 chuỗi khung hình. Bạn có thể sử dụng tất cả 100 khung hình hoặc một phần của chúng, nhưng cấm nghiêm ngặt việc lựa chọn thủ công các khung hình đầu vào. Việc lựa chọn mẫu con từ tập dữ liệu đầu vào theo cách thuật toán là chấp nhận được." - trích dẫn từ rule.

"Quy tắc quan trọng: Chúng tôi mong đợi dữ liệu sẽ rất khó khăn để khôi phục. Một phần của thách thức đến từ việc tập dữ liệu được chụp dưới một loạt các điều kiện nhiễu khác nhau. Do đó, chúng tôi nới lỏng ràng buộc về số lượng mô hình được đào tạo. Bạn có thể tùy chọn đào tạo các mô hình khác nhau cho các mức độ nhiễu khác nhau. Nếu bạn làm như vậy, bạn sẽ cần nộp tất cả các mô hình trong giai đoạn xác minh người chiến thắng."
### data_final
Mỗi file đều là numpy có shape (271, 271, 100)

### Symbol
- high: mean very wrong
- low: mean a bit good

## Related work
[Xia_2023_ICCV] 