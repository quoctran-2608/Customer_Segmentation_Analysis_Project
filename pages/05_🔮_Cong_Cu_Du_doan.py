# --- 1. Import các thư viện cần thiết ---
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from utils import load_data
from config import DATA_PATH

# --- 2. Cấu hình trang ---
st.set_page_config(
    page_title="Công cụ Dự đoán  Chi tiêu",
    page_icon="💰",
    layout="wide" # Sử dụng layout rộng
)

# --- 3. Tải Dữ liệu, Xử lý & Huấn luyện Mô hình OLS (Sử dụng Cache) ---
@st.cache_data # Cache toàn bộ quá trình
def load_train_ols(df_clean):
    """Tải dữ liệu, xử lý, huấn luyện OLS và trả về model, scaler, features."""
    try:
        # --- Chuẩn bị dữ liệu CHO MÔ HÌNH OLS ---
        cols_for_ols = ['Income', 'Total_Children', 'Customer_Tenure', 'Total_Spending']
        df_model_ols = df_clean[cols_for_ols].dropna()

        # --- Xử lý Ngoại vi (Outliers) ---
        Q1_inc = df_model_ols['Income'].quantile(0.25)
        Q3_inc = df_model_ols['Income'].quantile(0.75)
        IQR_inc = Q3_inc - Q1_inc
        lower_bound_inc = Q1_inc - 1.5 * IQR_inc
        upper_bound_inc = Q3_inc + 1.5 * IQR_inc
        
        Q1_spend = df_model_ols['Total_Spending'].quantile(0.25)
        Q3_spend = df_model_ols['Total_Spending'].quantile(0.75)
        IQR_spend = Q3_spend - Q1_spend
        lower_bound_spend = Q1_spend - 1.5 * IQR_spend
        upper_bound_spend = Q3_spend + 1.5 * IQR_spend

        df_ols_no_outliers = df_model_ols[
            (df_model_ols['Income'] >= lower_bound_inc) & (df_model_ols['Income'] <= upper_bound_inc) &
            (df_model_ols['Total_Spending'] >= lower_bound_spend) & (df_model_ols['Total_Spending'] <= upper_bound_spend)
            # Không cần lower bound vì thường >= 0
        ]

        # Tách X và y từ dữ liệu đã làm sạch outlier
        feature_cols_ols = ['Income', 'Total_Children', 'Customer_Tenure']
        target_col_ols = 'Total_Spending'
        X_ols_clean = df_ols_no_outliers[feature_cols_ols]
        y_ols_clean = df_ols_no_outliers[target_col_ols]

        # Chuẩn hóa Biến Độc lập
        scaler_ols = StandardScaler()
        X_scaled_ols = scaler_ols.fit_transform(X_ols_clean)
        X_scaled_df_ols = pd.DataFrame(X_scaled_ols, columns=feature_cols_ols, index=X_ols_clean.index)

        # Thêm cột Hằng số
        X_final_ols = sm.add_constant(X_scaled_df_ols)

        # Huấn luyện mô hình OLS trên toàn bộ dữ liệu sạch
        ols_model = sm.OLS(y_ols_clean, X_final_ols)
        ols_results = ols_model.fit()

        print("OLS model trained.")
        # Trả về kết quả fit, scaler và danh sách features
        return ols_results, scaler_ols, feature_cols_ols

    except Exception as e:
        st.error(f"Lỗi trong quá trình huấn luyện mô hình OLS: {e}")
        return None, None, None

# --- Tải Dữ liệu, hạy hàm để lấy kết quả OLS, scaler --- ---
df_clean = load_data(DATA_PATH)
ols_model, scaler, features = load_train_ols(df_clean)

# --- 4. Nội dung Trang Dự đoán ---
st.title("💰 Công cụ Dự đoán Tổng Chi tiêu")
st.markdown("Sử dụng mô hình **Hồi quy Tuyến tính OLS** để ước tính `Total_Spending` dựa trên các đặc điểm chính của khách hàng.")
st.markdown("*(Lưu ý: Mô hình được xây dựng trên dữ liệu đã loại bỏ các giá trị ngoại vi để tăng độ ổn định)*")
st.markdown("---")

# Chỉ hiển thị nếu model được huấn luyện thành công
if ols_model and scaler:

    # --- Hiển thị Kết quả Đánh giá Mô hình ---
    st.header("Kết quả Mô hình OLS (trên toàn bộ dữ liệu đã làm sạch)")

    # Trích xuất các chỉ số quan trọng từ summary
    r_squared = ols_model.rsquared_adj
    f_prob = ols_model.f_pvalue
    coef_table = ols_model.summary2().tables[1] # Lấy bảng hệ số

    col1, col2 = st.columns(2)
    with col1:
        st.metric("📊 Adj. R-squared", f"{r_squared:.3f}")
    with col2:
        st.metric("📉 Prob (F-statistic)", f"{f_prob:.3f}")

    st.markdown(f"=>  Mô hình giải thích **{r_squared*100:.1f}%** biến thiên của `Total_Spending`.")
    if f_prob < 0.05:
        st.success("Mô hình **có ý nghĩa thống kê** (do F-statistic rất nhỏ, gần bằng 0).")
    else:
        st.warning("Mô hình **chưa có ý nghĩa thống kê**.")

    
    # Hiển thị bảng hệ số
    st.subheader("📋 Yếu tố nào Ảnh hưởng đến Chi tiêu? (Hệ số & P-value)")
    st.dataframe(coef_table.round(4), use_container_width=True)
    
    # Lấy hệ số
    b0 = coef_table.loc['const', 'Coef.']
    b1 = coef_table.loc['Income', 'Coef.']
    b2 = coef_table.loc['Total_Children', 'Coef.']
    b3 = coef_table.loc['Customer_Tenure', 'Coef.']


    st.markdown(f"""
    Chúng ta xem xét cột **`coef`** (hệ số) để biết chiều hướng ảnh hưởng và cột **`P>|t|`** (p-value) để biết yếu tố đó có **quan trọng về mặt thống kê** hay không (nếu p < 0.05 là quan trọng).
                
    -   **`Income` (Thu nhập):**
    * Hệ số: **Dương (+{b1:.2f})**. Nghĩa là thu nhập **tăng** thì chi tiêu **tăng**.
    * P-value: **Rất nhỏ (0.000)**. Khẳng định Thu nhập là yếu tố **cực kỳ quan trọng**. 💪

    - **`Total_Children` (Số con):**
    * Hệ số: **Âm ({b2:.2f})**. Nghĩa là có thêm con thì chi tiêu **giảm**.
    * P-value: **Rất nhỏ (0.000)**. Khẳng định Số con cũng là yếu tố **rất quan trọng** (ảnh hưởng tiêu cực). 📉
    
    - **`Customer_Tenure` (Thâm niên):**
    * Hệ số: **Dương (+{b3:.2f})**. Nghĩa là số năm thâm niên lâu hơn thì chi tiêu có **tăng nhẹ**.
    * **P-value:** Rất nhỏ (0.000), cho thấy ảnh hưởng này (dù nhỏ) là **có ý nghĩa thống kê**, không phải do ngẫu nhiên. => Việc khách hàng gắn bó lâu hơn vẫn đóng góp một phần nhỏ nhưng đáng kể vào việc tăng chi tiêu.
    """)

    st.info(f"""
    => **Công thức (trên dữ liệu chuẩn hóa):** \n
    **Total_Spending** ≈ {b0:.2f} + ({b1:.2f})x(Scaled_Income) + ({b2:.2f})x(Scaled_Total_Children) + ({b3:.2f})x(Scaled_Customer_Tenure)
    """)
    
    st.markdown("---")
    
    # --- Công cụ Ước tính Tương tác ---
    st.header("⚙️ Thử nghiệm Dự đoán Chi tiêu")
    st.write("Nhập thông tin của một khách hàng giả định:")

    with st.form("estimation_form"):
        col_input1, col_input2, col_input3 = st.columns(3)
        with col_input1:
            input_income = st.number_input("Thu nhập (USD)", min_value=0, value=50000, step=1000)
        with col_input2:
            input_children = st.number_input("Tổng số con", min_value=0, value=1, step=1)
        with col_input3:
            input_tenure = st.number_input("Thâm niên (Ngày)", min_value=0, value=365, step=10)

        submitted = st.form_submit_button("Dự đoán Chi tiêu")

    if submitted:
        # Chuẩn bị dữ liệu đầu vào cho dự đoán
        input_data = pd.DataFrame({
            'Income': [input_income],
            'Total_Children': [input_children],
            'Customer_Tenure': [input_tenure]
        })
        # Chọn đúng cột theo feature_cols_ols
        input_data = input_data[features] # Đảm bảo đúng thứ tự cột

        # Chuẩn hóa đầu vào
        try:
            input_scaled = scaler.transform(input_data)
            input_scaled_df = pd.DataFrame(input_scaled, columns=features)
            input_final = sm.add_constant(input_scaled_df, has_constant='add') # Thêm hằng số
        except Exception as e:
            st.error(f"Lỗi khi chuẩn hóa dữ liệu đầu vào: {e}.")
            st.stop()

        # Ước tính bằng mô hình OLS
        prediction = ols_model.predict(input_final)
        predicted_spending = prediction[0]

        # Hiển thị kết quả
        st.subheader("--- Kết quả Dự đoán ---")
        st.success(f"📈 Dự đoán Tổng Chi tiêu (Total Spending): **{predicted_spending:,.2f} USD**")

else:
    st.warning("Không thể huấn luyện hoặc tải mô hình OLS.")