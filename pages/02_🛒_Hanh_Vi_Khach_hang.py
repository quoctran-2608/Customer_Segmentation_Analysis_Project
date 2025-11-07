# --- 1. Import các thư viện cần thiết ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data
from config import DATA_PATH

# --- 2. Cấu hình trang ---
st.set_page_config(
    page_title="Hành vi Khách hàng",
    page_icon="🛒",
    layout="wide"
)

# --- 3. Tải dữ liệu ---
df = load_data(DATA_PATH)

# --- 4. Nội dung Trang Hành vi ---
st.title("🛒 Hành vi Mua sắm & Tương tác")
st.markdown("Trang này trực quan hóa cách khách hàng chi tiêu, mua sắm qua các kênh, và tương tác trực tuyến.")
st.markdown("---")

# Chỉ hiển thị nội dung nếu dữ liệu được tải thành công
if df is not None:

    # --- 5. Thêm Filter vào Sidebar ---
    st.sidebar.header("Bộ lọc Khám phá:")
   
    # --- Filter Country ---
    country_options = sorted(df['Country'].unique().astype(str))
    selected_country = st.sidebar.multiselect('Chọn Quốc gia:', options=country_options, default=country_options)
    
    # --- Filter Education ---
    education_options = df['Education'].unique().astype(str)
    selected_education = st.sidebar.multiselect('Chọn Trình độ Học vấn:', options=education_options, default=education_options)
    
    # --- Filter Tình trạng hôn nhân ---
    marital_options = sorted(df['Marital_Status'].unique().astype(str))
    selected_marital = st.sidebar.multiselect('Chọn Tình trạng Hôn nhân:', options=marital_options, default=marital_options)
    
    # --- Filter Nhóm Tuổi ---
    # Lấy các nhóm tuổi duy nhất từ cột Age_Group đã tạo
    age_group_options = ['Dưới 30', '30-39', '40-49', '50-59', '60-69', 'Trên 70']
    # Lọc ra các nhóm tuổi thực sự có trong dữ liệu (sau khi lọc ban đầu nếu có)
    available_age_groups = [label for label in age_group_options if label in df['Age_Group'].unique().astype(str)]
    selected_age_group = st.sidebar.multiselect(
        'Chọn Nhóm Tuổi:',
        options=available_age_groups,
        default=available_age_groups
    )
    
    income_group_options = ['Dưới 30k', '30k-50k', '50k-70k', '70k-90k', 'Trên 90k']
    available_income_groups = [label for label in income_group_options if label in df['Income_Group'].unique().astype(str)]
    selected_income_group = st.sidebar.multiselect(
        'Chọn Nhóm Thu nhập (USD):',
        options=available_income_groups,
        default=available_income_groups
    )      

    # --- 6. Áp dụng Filter vào DataFrame ---
    df_filtered = df.copy()
    if selected_country:
        df_filtered = df_filtered[df_filtered['Country'].isin(selected_country)]
    if selected_education:
        df_filtered = df_filtered[df_filtered['Education'].isin(selected_education)]
    if selected_marital:
        df_filtered = df_filtered[df_filtered['Marital_Status'].isin(selected_marital)]
    if selected_age_group:
        df_filtered = df_filtered[df_filtered['Age_Group'].isin(selected_age_group)]
    if selected_income_group:
        df_filtered = df_filtered[df_filtered['Income_Group'].isin(selected_income_group)]

    st.write(f"Đang hiển thị dữ liệu cho **{len(df_filtered):,}** khách hàng.")
    st.markdown("---")

    if df_filtered.empty:
        st.warning("Không có khách hàng nào phù hợp với bộ lọc đã chọn.")
    else:
        # --- Sử dụng Tabs để tổ chức nội dung ---
        tab1, tab2, tab3 = st.tabs(["📊 Chi tiêu (Spending)", "🛍️ Mua sắm (Purchasing)", "🛒 Tương tác"])

        # --- Tab 1: Chi tiêu ---
        with tab1:
            st.header("Phân tích Chi tiêu")

            col1a, col1b = st.columns(2)

            with col1a:
                # Biểu đồ Histogram: Phân phối Tổng Chi tiêu
                st.subheader("Phân phối Tổng Chi tiêu (Histogram)")
                fig_spend_hist, ax_spend_hist = plt.subplots(figsize=(8, 5))
                sns.histplot(data=df_filtered, x='Total_Spending', kde=True, bins=30, color='darkgreen', ax=ax_spend_hist)
                ax_spend_hist.set_xlabel('Tổng Chi tiêu (USD)')
                ax_spend_hist.set_ylabel('Số lượng khách hàng')
                st.pyplot(fig_spend_hist)

                # --- Biểu đồ Box Plot: Phân phối Tổng Chi tiêu ---
                st.subheader("Phân phối Tổng Chi tiêu (Box Plot)")
                fig_spend_box, ax_spend_box = plt.subplots(figsize=(8, 3))
                sns.boxplot(data=df_filtered, x='Total_Spending', color='skyblue', ax=ax_spend_box)
                ax_spend_box.set_xlabel('Tổng Chi tiêu (USD)')
                st.pyplot(fig_spend_box)


            with col1b:
                # Biểu đồ Cột: Tổng Doanh thu theo Danh mục Sản phẩm
                st.subheader("Tổng Doanh thu theo Sản phẩm")
                spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
                # Tính tổng trên dữ liệu đã lọc
                total_spending_by_category_filt = df_filtered[spending_cols].sum().sort_values(ascending=False)
                total_spending_by_category_filt.index = [col.replace('Mnt', '') for col in total_spending_by_category_filt.index]

                fig_spend_cat, ax_spend_cat = plt.subplots(figsize=(8, 5))
                sns.barplot(x=total_spending_by_category_filt.index, y=total_spending_by_category_filt.values, 
                hue=total_spending_by_category_filt.index, palette='rocket', ax=ax_spend_cat)
                ax_spend_cat.set_xlabel('Danh mục Sản phẩm')
                ax_spend_cat.set_ylabel('Tổng Doanh thu (USD)')
                plt.xticks(rotation=15)
                st.pyplot(fig_spend_cat)

        # --- Tab 2: Mua sắm ---
        with tab2:
            st.header("Phân tích Mua sắm")

            col2a, col2b = st.columns(2)

            with col2a:
                # Biểu đồ Histogram: Phân phối Tổng số lần Mua hàng
                st.subheader("Phân phối Tần suất Mua hàng")
                fig_freq_hist, ax_freq_hist = plt.subplots(figsize=(8, 5))
                sns.histplot(data=df_filtered, x='Total_NumberOfPurchases', kde=True, bins=25, color='purple', ax=ax_freq_hist)
                ax_freq_hist.set_xlabel('Tổng số lần Mua hàng')
                ax_freq_hist.set_ylabel('Số lượng khách hàng')
                st.pyplot(fig_freq_hist)

                # Biểu đồ Histogram: Phân phối Recency
                st.subheader("Phân phối Lần mua cuối (Recency)")
                fig_recency_hist, ax_recency_hist = plt.subplots(figsize=(8, 5))
                sns.histplot(data=df_filtered, x='Recency', kde=True, bins=20, color='skyblue', ax=ax_recency_hist)
                ax_recency_hist.set_xlabel('Số ngày kể từ lần mua cuối')
                ax_recency_hist.set_ylabel('Số lượng khách hàng')
                st.pyplot(fig_recency_hist)


            with col2b:
                # Biểu đồ Cột: Tổng Giao dịch theo Kênh Mua sắm
                st.subheader("Tổng Giao dịch theo Kênh")
                purchase_channels = ['NumStorePurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumDealsPurchases']
                ## Tính tổng trên dữ liệu đã lọc
                total_purchases_by_channel_filt = df_filtered[purchase_channels].sum().sort_values(ascending=False)
                # Đổi tên cột cho đẹp hơn (bỏ "Num" và "Purchases")
                total_purchases_by_channel_filt.index = ['Store', 'Web', 'Catalog', 'Deals']

                fig_channel_bar, ax_channel_bar = plt.subplots(figsize=(8, 5))
                sns.barplot(x=total_purchases_by_channel_filt.index, y=total_purchases_by_channel_filt.values,
                hue=total_purchases_by_channel_filt.index, palette='magma', ax=ax_channel_bar)
                ax_channel_bar.set_xlabel('Kênh Mua sắm')
                ax_channel_bar.set_ylabel('Tổng số Giao dịch')
                st.pyplot(fig_channel_bar)

        # --- Tab 3: Tương tác Web ---
        with tab3:
            st.header("Phân tích Tương tác")
            
            # Biểu đồ Cột: Số lượt Truy cập Web/Tháng
            st.subheader("Số lượt Truy cập Web/Tháng")
            web_visits_counts = df_filtered['NumWebVisitsMonth'].value_counts().sort_index()
            fig_web_visits, ax_web_visits = plt.subplots(figsize=(8, 5))
            sns.barplot(
                x=web_visits_counts.index,
                y=web_visits_counts.values,
                hue=web_visits_counts.index,
                palette='GnBu_r', ax=ax_web_visits,
                legend=False
            )
            ax_web_visits.set_xlabel('Số lượt Truy cập Web/tháng')
            ax_web_visits.set_ylabel('Số lượng khách hàng')
            st.pyplot(fig_web_visits)

            st.markdown("---")
            
            # --- Biểu đồ Tròn: Tỷ lệ Phản hồi Chiến dịch ---
            st.subheader("Tỷ lệ Phản hồi Chiến dịch (Tổng thể)")
            response_counts = df_filtered['Response'].value_counts()
            
            # Kiểm tra xem có dữ liệu hay không
            if not response_counts.empty:
                response_labels = ['Không Phản hồi (0)', 'Phản hồi (1)']
                # Lấy số lượng đếm của 0 và 1, đảm bảo thứ tự và xử lý trường hợp thiếu giá trị.
                counts_ordered = [response_counts.get(0, 0), response_counts.get(1, 0)]
                colors = ['#B0C4DE', '#FF6347']
                
                fig_response_pie, ax_response_pie = plt.subplots(figsize=(8, 5))
                ax_response_pie.pie(
                    counts_ordered, 
                    labels=response_labels, 
                    autopct='%1.1f%%', 
                    startangle=90, 
                    colors=colors,
                    wedgeprops={'edgecolor': 'white'}
                )
                ax_response_pie.axis('equal')
                st.pyplot(fig_response_pie)
            else:
                st.warning("Không có dữ liệu Phản hồi (Response) để hiển thị.")

else:
    st.warning("Không thể tải dữ liệu để hiển thị trang này.")