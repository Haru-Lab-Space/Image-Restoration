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
[Xia_2023_ICCV] bao gồm ba thành phần: mạng trích xuất trước IR nhỏ gọn (CPEN), dynamic IR transformer và mạng khử nhiễu. Trainging processing consists of two  stages: pretraining and training. In pretraining stage, we input ground-truth image into CPEN to capture the compact IR prior representation (IRP) to guide DIRformer. And in the training stage, we training the deffusion model to directly estimate the same IRP in CPEN only using LQ images.

Diffusion model thường được sử dụng trong nhiệm vụ tổng hợp hình ảnh. Dưới góc nhìn của nhiệm vụ khôi phục hình ảnh, diffusion trở nên quá phức tạp và tốn kém, đôi khi, chính sự phức tạp đó khiến cho hiệu suất của mô hình giảm. 

- Sử dụng Transformer vì nó có thể mô hình hóa sự phụ thuộc pixel dài hạn (knowledge base). Và tại đó, họ sử dụng Transformer tại quá trình xây dựng UNet.
- Sử dụng Diffusion Model vì các kết quả mô hình thực tế cho thấy nó hiệu quả hơn các phương pháp truyền thống.
- Compact IR prior extraction network (CPEN)
- Dynamic Gated Feed-Forward Network (DGFN) 
- Dynamic Multi-Head Transposed Attention (DMTA)


SRCNN
DnCNN
ARCNN
RePaint


## Motivation
Nguyên nhân gây ra sự mờ ảnh chính là máy ảnh không theo kịp tốc độ thay đổi hình ảnh. Do đó, chúng ta sẽ cố gắng tái tạo lại thực tế này bằng cách gộp k ảnh liền kề thành một ảnh duy nhất từ đó phân tích độ mờ do vấn đề không bắt kịp frame của máy ảnh.

