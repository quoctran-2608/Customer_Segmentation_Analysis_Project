# --- 1. Import các thư viện cần thiết ---
import streamlit as st
import pandas as pd
import numpy as np
from utils import load_data

# --- 2. Cấu hình trang ---
st.set_page_config(
    page_title="Phân tích Phân khúc Khách hàng",
    page_icon="📊",
    layout="wide"
)

# --- 3. Tải dữ liệu ---
data_path = 'https://raw.githubusercontent.com/riodev1310/rio_datasets/refs/heads/main/marketing_data_with_missing_values.csv'
df = load_data(data_path)

# --- 4. Nội dung Trang Chính ---
if df is not None:
    st.title("📊 Phân tích Phân khúc Khách hàng")
    st.markdown("---")

    st.header("📝 Giới thiệu")
    st.markdown("""
    Chào mừng đến với dashboard phân tích dữ liệu khách hàng từ chiến dịch marketing.
    Mục tiêu của dự án này là:
    * **Hiểu rõ** đặc điểm nhân khẩu học và hành vi mua sắm của khách hàng.
    * **Khám phá** các mối quan hệ ẩn giữa các yếu tố khác nhau.
    * **Phân khúc** khách hàng thành các nhóm chiến lược bằng mô hình RFM.
    * Xây dựng một **công cụ dự đoán** cơ bản về khả năng phản hồi marketing.

    Sử dụng thanh điều hướng bên trái để khám phá các phần phân tích chi tiết.
    """)
    st.markdown("---")

    st.header("🔢 Tổng quan Dữ liệu")
    # **(Bổ sung)** Nhấn mạnh dữ liệu đã được xử lý
    st.info("Lưu ý: Dữ liệu hiển thị trên dashboard này đã được **tiền xử lý và làm sạch** từ file gốc.")

    st.write(f"Bộ dữ liệu sau xử lý gồm **{df.shape[0]:,}** dòng (khách hàng) và **{df.shape[1]}** cột (thuộc tính).")

    # **(MỚI)** Phần liệt kê các bước tiền xử lý
    st.subheader("Các thao tác Tiền xử lý & Feature Engineering chính đã áp dụng:")
    st.markdown("""
        1.  **Xử lý định dạng:**
            * `Income`: Loại bỏ ký tự `$` và `,`, chuyển sang dạng số.
            * `Marital_Status`: Gom nhóm các giá trị ('Married', 'Together' -> 'Partnered'; 'Single', 'Divorced', 'Widow', 'Alone' -> 'Single'; 'YOLO', 'Absurd' -> 'Other').
            * `Dt_Customer`: Chuyển sang định dạng datetime.
        2.  **Xử lý Giá trị thiếu (NULL):**
            * `Education`: Fill bằng giá trị phổ biến nhất ('Graduation').
            * `Income`: Fill bằng giá trị trung vị (median).
            * `Dt_Customer`: Fill bằng ngày phổ biến nhất (mode).
            * `NumWebVisitsMonth`: Fill bằng giá trị phổ biến nhất (mode).
        3.  **Xử lý Ngoại vi (Outliers):**
            * Loại bỏ các hàng có `Year_Birth` < 1920 (3 hàng).
        4.  **Feature Engineering (Làm giàu Dữ liệu):**
            * Tạo cột `Age` và `Age_Group`.
            * Tạo cột `Total_Children` = `Kidhome` + `Teenhome`.
            * Tạo cột `Income_Group` (phân nhóm thu nhập).
            * Tạo cột `Total_Spending` (tổng chi tiêu 6 danh mục Mnt).
            * Tạo cột `Total_NumberOfPurchases` (tổng số lần mua hàng trên các kênh).
            * Tạo cột `Customer_Tenure` (thâm niên khách hàng) và `Tenure_Group`.
        5.  **Xử lý Lỗi Logic:**
            * Loại bỏ 4 hàng có chi tiêu > 0 nhưng số lần mua = 0.
        """)

    st.subheader("Thống kê Mô tả (Các cột số chính sau xử lý)")
    numeric_cols_to_describe = ['Income', 'Age', 'Total_Children', 'Customer_Tenure', 'Recency', 'Total_Spending', 'Total_NumberOfPurchases', 'NumWebVisitsMonth']
    # Đảm bảo chỉ chọn các cột tồn tại trong df
    cols_exist = [col for col in numeric_cols_to_describe if col in df.columns]
    if cols_exist:
        st.dataframe(df[cols_exist].describe().T, use_container_width=True)
    else:
        st.warning("Không tìm thấy các cột số chính để hiển thị thống kê.")


    st.subheader("Xem lướt Dữ liệu sau Xử lý")
    st.write("Hiển thị 10 dòng dữ liệu đầu tiên (đã xử lý)")
    st.dataframe(df.head(10), use_container_width=True)

else:
    st.warning("Không thể tải hoặc xử lý dữ liệu. Vui lòng kiểm tra lại file hoặc đường dẫn.")