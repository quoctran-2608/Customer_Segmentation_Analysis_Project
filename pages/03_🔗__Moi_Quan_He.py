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
    page_title="Mối Quan hệ & Insights",
    page_icon="🔗",
    layout="wide"
)

# --- 3. Tải dữ liệu ---
df = load_data(DATA_PATH)

# --- 4. Nội dung Trang Mối quan hệ ---
st.title("🔗 Mối Quan hệ & Insights Chính")
st.markdown("Trang này khám phá các mối liên hệ giữa các đặc điểm và hành vi của khách hàng (Phần C trong EDA).")
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
        # --- 7. Hiển thị các Biểu đồ Phân tích Mối quan hệ ---

        # 7.1. Biểu đồ nhiệt Tương quan
        st.subheader("1. Biểu đồ nhiệt Tương quan Tổng thể")
        correlation_cols_full = [
            'Income', 'Age', 'Total_Children', 'Customer_Tenure', 'Recency',
            'Total_Spending', 'Total_NumberOfPurchases', 'NumWebVisitsMonth',
            'NumWebPurchases', 'NumCatalogPurchases', 'NumDealsPurchases', 'NumStorePurchases'
        ]
        df_corr_display = df_filtered if len(df_filtered) > 1 else df
        # Bỏ qua các cột không phải số hoàn toàn trước khi tính corr()
        df_corr_numeric = df_corr_display[correlation_cols_full].select_dtypes(include=np.number)
        correlation_matrix_display = df_corr_numeric.corr()

        fig_heatmap, ax_heatmap = plt.subplots(figsize=(14, 10))
        sns.heatmap(correlation_matrix_display, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, ax=ax_heatmap)
        st.pyplot(fig_heatmap)
        st.markdown("*Biểu đồ này cho thấy mối quan hệ tuyến tính giữa các biến số. Màu đỏ = tương quan thuận, Xanh = tương quan nghịch.*")
        st.markdown("---")

        # 7.2. Các phân tích chuyên sâu (chia 2 cột)
        st.header("2. Phân tích Mối quan hệ Chuyên sâu")
        col_left, col_right = st.columns(2)

        with col_left:
            #  7.2.1. Thu nhập vs. Chi tiêu
            st.subheader("Thu nhập vs. Tổng Chi tiêu")
            fig_inc_spend, ax_inc_spend = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=df_filtered, x='Income', y='Total_Spending', alpha=0.5, ax=ax_inc_spend)
            ax_inc_spend.set_title('Mối quan hệ giữa Thu nhập và Tổng Chi tiêu')
            ax_inc_spend.set_xlabel('Thu nhập (USD)')
            ax_inc_spend.set_ylabel('Tổng Chi tiêu (USD)')
            st.pyplot(fig_inc_spend)

            # 7.2.2. Con cái vs. Chi tiêu
            st.subheader("Số con vs. Tổng Chi tiêu Trung bình")
            fig_child_spend, ax_child_spend = plt.subplots(figsize=(8, 6))
            avg_spend_child_filtered = df_filtered.groupby('Total_Children', observed=True)['Total_Spending'].mean().reset_index()
            sns.barplot(data=avg_spend_child_filtered, 
                        x='Total_Children', y='Total_Spending', 
                        hue='Total_Children',
                        palette='flare', ax=ax_child_spend)
            ax_child_spend.set_title('Ảnh hưởng của Số con đến Chi tiêu Trung bình')
            ax_child_spend.set_xlabel('Tổng số con cái')
            ax_child_spend.set_ylabel('Tổng Chi tiêu Trung bình (USD)')
            st.pyplot(fig_child_spend)

            # 7.2.3. Tỷ lệ Phản hồi theo Thu nhập
            st.subheader("Tỷ lệ Phản hồi theo Thu nhập")
            income_resp_rate_filtered = df_filtered.groupby('Income_Group', observed=True)['Response'].mean() * 100
            fig_resp_inc, ax_resp_inc = plt.subplots(figsize=(8, 6))
            sns.barplot(x=income_resp_rate_filtered.index, 
                        y=income_resp_rate_filtered.values,
                        hue=income_resp_rate_filtered.index, 
                        palette='YlGn', ax=ax_resp_inc)
            # Thêm nhãn phần trăm lên trên mỗi cột
            for p in ax_resp_inc.patches:
                 if p.get_height() > 0: ax_resp_inc.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
            ax_resp_inc.set_title('Tỷ lệ Phản hồi Chiến dịch theo Nhóm Thu nhập')
            ax_resp_inc.set_xlabel('Nhóm Thu nhập (USD)')
            ax_resp_inc.set_ylabel('Tỷ lệ Phản hồi (%)')
            st.pyplot(fig_resp_inc)

        with col_right:
            # 7.2.4. Học vấn vs. Chi tiêu
            st.subheader("Học vấn vs. Tổng Chi tiêu")
            fig_edu_spend, ax_edu_spend = plt.subplots(figsize=(8, 6))
            education_order = ['Basic', '2n Cycle', 'Graduation', 'Master', 'PhD']
            # Lọc order chỉ lấy các giá trị có trong dữ liệu đã lọc
            filtered_edu_order = [e for e in education_order if e in df_filtered['Education'].unique().astype(str)]
            if filtered_edu_order: # Chỉ vẽ nếu có dữ liệu
                sns.boxplot(data=df_filtered, x='Education', y='Total_Spending', 
                            palette='Spectral', hue='Education',
                            order=filtered_edu_order, ax=ax_edu_spend)
                ax_edu_spend.set_title('So sánh Tổng Chi tiêu theo Trình độ Học vấn')
                ax_edu_spend.set_xlabel('Trình độ Học vấn')
                ax_edu_spend.set_ylabel('Tổng Chi tiêu (USD)')                
                st.pyplot(fig_edu_spend)
            else:
                 st.write("Không đủ dữ liệu Học vấn để vẽ biểu đồ.")

            #  7.2.5. Kênh theo Thu nhập (% Stacked Bar)
            st.subheader("Tỷ lệ Kênh theo Nhóm Thu nhập")
            channels = ['NumStorePurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumDealsPurchases']
            ## Đổi tên kênh cho ngắn gọn
            channel_rename = {'NumStorePurchases':'Store', 'NumWebPurchases':'Web', 'NumCatalogPurchases':'Catalog', 'NumDealsPurchases':'Deals'}
            df_channel_income_filt = df_filtered.groupby('Income_Group', observed=True)[channels].sum()
            if not df_channel_income_filt.empty:
                df_channel_income_filt = df_channel_income_filt.rename(columns=channel_rename) # Đổi tên cột
                ## Tính tỷ lệ phần trăm cho mỗi kênh trong từng nhóm thu nhập
                df_channel_income_pct_filt = df_channel_income_filt.apply(lambda x: 100 * x / x.sum(), axis=1)
                ## Vẽ biểu đồ cột chồng
                fig_ch_inc, ax_ch_inc = plt.subplots(figsize=(8, 6))                
                df_channel_income_pct_filt.plot(kind='bar', stacked=True, colormap='cividis', width=0.8, ax=ax_ch_inc)
                ax_ch_inc.set_title('Tỷ lệ Kênh Mua sắm theo Nhóm Thu nhập')
                ax_ch_inc.set_xlabel('Nhóm Thu nhập (USD)')
                ax_ch_inc.set_ylabel('Tỷ lệ Giao dịch (%)')
                ## Thêm chú giải bên ngoài biểu đồ
                ax_ch_inc.legend(title='Kênh', bbox_to_anchor=(1.02, 1), loc='upper left')
                plt.xticks(rotation=0)
                st.pyplot(fig_ch_inc)
            else:
                 st.write("Không đủ dữ liệu Thu nhập để vẽ biểu đồ Tỷ lệ kênh.")

            #  7.2.6. Kênh theo Tuổi (% Stacked Bar) ---
            st.subheader("Tỷ lệ Kênh theo Nhóm Tuổi")
            df_channel_age_filt = df_filtered.groupby('Age_Group', observed=True)[channels].sum()
            if not df_channel_age_filt.empty:
                df_channel_age_filt = df_channel_age_filt.rename(columns=channel_rename) # Đổi tên cột
                df_channel_age_pct_filt = df_channel_age_filt.apply(lambda x: 100 * x / x.sum(), axis=1)
                fig_ch_age, ax_ch_age = plt.subplots(figsize=(8, 6))
                df_channel_age_pct_filt.plot(kind='bar', stacked=True, colormap='viridis', width=0.8, ax=ax_ch_age)
                ax_ch_age.set_title('Tỷ lệ Kênh Mua sắm theo Nhóm Tuổi')
                ax_ch_age.set_xlabel('Nhóm Tuổi')
                ax_ch_age.set_ylabel('Tỷ lệ Giao dịch (%)')
                ## Thêm chú giải bên ngoài biểu đồ
                ax_ch_age.legend(title='Kênh', bbox_to_anchor=(1.02, 1), loc='upper left')
                plt.xticks(rotation=0)
                st.pyplot(fig_ch_age)
            else:
                 st.write("Không đủ dữ liệu Tuổi để vẽ biểu đồ Tỷ lệ kênh.")

            #  7.2.7. Tỷ lệ Phản hồi theo Tuổi
            st.subheader("Tỷ lệ Phản hồi theo Tuổi")
            age_resp_rate_filtered = df_filtered.groupby('Age_Group', observed=True)['Response'].mean() * 100
            fig_resp_age, ax_resp_age = plt.subplots(figsize=(8, 6))
            sns.barplot(x=age_resp_rate_filtered.index, 
                        y=age_resp_rate_filtered.values, 
                        hue=age_resp_rate_filtered.index,
                        palette='YlOrBr', ax=ax_resp_age)
            # Thêm nhãn phần trăm lên trên mỗi cột
            for p in ax_resp_age.patches:
                 if p.get_height() > 0: ax_resp_age.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
            ax_resp_age.set_title('Tỷ lệ Phản hồi Chiến dịch theo Nhóm Tuổi')
            ax_resp_age.set_xlabel('Nhóm Tuổi')
            ax_resp_age.set_ylabel('Tỷ lệ Phản hồi (%)')
            st.pyplot(fig_resp_age)

else:
    st.warning("Không thể tải dữ liệu để hiển thị trang này.")