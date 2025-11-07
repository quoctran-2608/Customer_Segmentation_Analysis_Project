# utils.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from config import DATA_PATH
import streamlit as st

@st.cache_data
def load_data(data_path):
    try:
        df = pd.read_csv(data_path)
        
        # --- Thực hiện các bước Tiền xử lý & Feature Engineering ---
        df_clean = df.copy()
    
        # XỬ LÝ ĐỊNH DẠNG DỮ LIỆU
        ## làm sạch `Income`: bằng cách Loại bỏ "$", ",",  khoảng trắng thừa và Chuyển đổi
        df_clean['Income'] = df_clean['Income'].astype("string").str.replace('$', '').str.replace(',', '').str.strip()
        df_clean['Income'] = pd.to_numeric(df_clean['Income'], errors='coerce')
        
        ## Tái phân loại cột Marital_Status
        df_clean['Marital_Status'] = df_clean['Marital_Status'].map({
            'Married': 'Partnered', 'Together': 'Partnered',
            'Single': 'Single', 'Divorced': 'Single', 'Widow': 'Single', 'Alone': 'Single',
            'YOLO': 'Other', 'Absurd': 'Other'
        })
        
        ## Chuyển đổi cột Dt_Customer sang Datetime
        df_clean['Dt_Customer'] = pd.to_datetime(df_clean['Dt_Customer'], format='mixed')
        
        # XỬ LÝ GIÁ TRỊ THIẾU (NULL)
        df_clean['Education'] = df_clean['Education'].fillna(df_clean['Education'].mode()[0])
        df_clean['Income'] = df_clean['Income'].fillna(df_clean['Income'].median())
        df_clean['Dt_Customer'] = df_clean['Dt_Customer'].fillna(df_clean['Dt_Customer'].mode()[0])
        df_clean['NumWebVisitsMonth'] = df_clean['NumWebVisitsMonth'].fillna(df_clean['NumWebVisitsMonth'].mode()[0])
        
        # XỬ LÝ NGOẠI VI (OUTLIERS)
        current_year = datetime.now().year
        df_clean = df_clean[df_clean['Year_Birth'] > 1920]
        
        # FEATURE ENGINEERING (LÀM GIÀU DỮ LIỆU)
        ## Tạo cột Age
        df_clean['Age'] = current_year - df_clean['Year_Birth']
        ## Tạo cột Age_Group
        age_bins = [0, 30, 40, 50, 60, 70, 120]
        age_labels = ['Dưới 30', '30-39', '40-49', '50-59', '60-69', 'Trên 70']
        df_clean['Age_Group'] = pd.cut(df_clean['Age'], bins=age_bins, labels=age_labels, right=False)
        
        ## Tạo cột Total_Children
        df_clean['Total_Children'] = df_clean['Kidhome'] + df_clean['Teenhome']
        
        ## Tạo cột Income_Group
        income_bins = [0, 30000, 50000, 70000, 90000, np.inf]
        income_labels = ['Dưới 30k', '30k-50k', '50k-70k', '70k-90k', 'Trên 90k']
        df_clean['Income_Group'] = pd.cut(df_clean['Income'], bins=income_bins, labels=income_labels, right=False)
        
        ## Tạo cột Total_Spending
        spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        df_clean['Total_Spending'] = df_clean[spending_cols].sum(axis=1)
        
        ## Tạo cột Total_NumberOfPurchases
        purchase_cols = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
        df_clean['Total_NumberOfPurchases'] = df_clean[purchase_cols].sum(axis=1)
        
        ## Tạo cột Customer_Tenure
        latest_date = df_clean['Dt_Customer'].max()
        df_clean['Customer_Tenure'] = (latest_date - df_clean['Dt_Customer']).dt.days
        
        ## Tạo cột Tenure_Group
        tenure_bins = [0, 365, 730, np.inf]
        tenure_labels = ['Dưới 1 năm (Mới)', 'Từ 1-2 năm (Thân thiết)', 'Trên 2 năm (Trung thành)']
        df_clean['Tenure_Group'] = pd.cut(df_clean['Customer_Tenure'], bins=tenure_bins, labels=tenure_labels, right=False)
        
        # XỬ LÝ LỖI LOGIC
        ## Điều kiện 1: Không chi tiêu nhưng có lượt mua hàng
        logic_error_1 = df_clean[(df_clean['Total_Spending'] == 0) & (df_clean['Total_NumberOfPurchases'] > 0)]

        ## Điều kiện 2: Có chi tiêu nhưng không có lượt mua hàng
        logic_error_2 = df_clean[(df_clean['Total_Spending'] > 0) & (df_clean['Total_NumberOfPurchases'] == 0)]
        
        ## Kết hợp cả hai điều kiện
        illogical_data = pd.concat([logic_error_1, logic_error_2])
        
        ## Loại bỏ các hàng lỗi logic
        if not illogical_data.empty:
            df_clean = df_clean.drop(illogical_data.index)
                
        # ÉP KIỂU CHO CÁC CỘT PHÂN LOẠI
        for col in ['Education', 'Marital_Status', 'Country', 'Age_Group', 'Income_Group', 'Tenure_Group']:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype('category')
                
        return df_clean
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy file dữ liệu tại '{data_path}'. Vui lòng kiểm tra lại đường dẫn.")
        return None
    except Exception as e:
        st.error(f"Lỗi khi tải hoặc xử lý dữ liệu: {e}")
        return None
    
    
# --- HÀM LOAD VÀ HUẤN LUYỆN MÔ HÌNH OLS (CACHE) ---
@st.cache_resource
def load_ols_model():
    try:
        # Load data từ URL
        df_clean = load_data(DATA_PATH)

        # --- Chuẩn bị dữ liệu CHO MÔ HÌNH OLS ---
        cols_for_ols = ['Income', 'Total_Children', 'Customer_Tenure', 'Total_Spending']
        df_model_ols = df_clean[cols_for_ols].dropna()

        # --- Xử lý Outliers ---
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

        st.info("Mô hình OLS đã được huấn luyện thành công.")
        return ols_results, scaler_ols, feature_cols_ols

    except Exception as e:
        st.error(f"Lỗi trong quá trình huấn luyện mô hình OLS: {e}")
        return None, None, None
    
def predict_spending(income, total_children, customer_tenure):
    # Lấy từ session_state (đã load trước đó)
    if not st.session_state.get('ols_loaded', False):
        return None
    ols_results = st.session_state.ols_results
    scaler_ols = st.session_state.scaler_ols
    feature_cols_ols = st.session_state.feature_cols_ols
    try:
        # Chuẩn bị dữ liệu đầu vào cho dự đoán
        input_data = pd.DataFrame({
            'Income': [income],
            'Total_Children': [total_children],
            'Customer_Tenure': [customer_tenure]
        })
        # Chọn đúng cột theo feature_cols_ols
        input_data = input_data[feature_cols_ols] # Đảm bảo đúng thứ tự cột

        # Chuẩn hóa đầu vào
        input_scaled = scaler_ols.transform(input_data)
        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_cols_ols)
        input_final = sm.add_constant(input_scaled_df, has_constant='add')  # Thêm hằng số

        # Dự đoán
        prediction = ols_results.predict(input_final)
        return round(prediction[0], 2)
    except Exception as e:
        # st.error(f"Lỗi dự đoán: {e}")
        return None