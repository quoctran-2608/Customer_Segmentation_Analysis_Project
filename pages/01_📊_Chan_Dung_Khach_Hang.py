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
    page_title="Chân dung Khách hàng",
    page_icon="🧑‍🤝‍🧑",
    layout="wide" 
)

# --- 3. Tải dữ liệu ---
df = load_data(DATA_PATH)


# --- 4. Nội dung Trang Chân dung ---
st.title("🧑‍🤝‍🧑 Chân dung Nhân khẩu học Khách hàng")
st.markdown("Trang này trực quan hóa các đặc điểm nhân khẩu học chính. Sử dụng bộ lọc bên trái để khám phá sâu hơn.")
st.markdown("---")

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
        # --- 7. Hiển thị Thông tin & Biểu đồ (Dùng df_filtered) ---
        col1, col2 = st.columns(2)

        with col1:
            # --- Biểu đồ 1 - Histogram: Phân phối Tuổi (Age) ---
            st.subheader("1. Phân phối Chi tiết theo Tuổi")
            fig_age, ax_age = plt.subplots(figsize=(8,5))
            sns.histplot(data=df_filtered, x='Age', bins=20, kde=True, color='teal', ax=ax_age)
            ax_age.set_xlabel('Độ tuổi'); ax_age.set_ylabel('Số lượng khách hàng')
            st.pyplot(fig_age)
         
            # --- Biểu đồ 3: Phân khúc Tuổi (Age_Group) ---
            st.subheader("3. Phân khúc theo Nhóm Tuổi")
            customers_by_age_group = df_filtered['Age_Group'].value_counts()
            fig_age_bar, ax_age_bar = plt.subplots(figsize=(8, 5))
            sns.barplot(x=customers_by_age_group.index,
                        y=customers_by_age_group.values, 
                        palette='viridis', 
                        hue=customers_by_age_group.index,
                        ax=ax_age_bar
                        )            
            ax_age_bar.set_xlabel('Phân khúc Tuổi (Age Group)')
            ax_age_bar.set_ylabel('Số lượng khách hàng')
            st.pyplot(fig_age_bar)

            # --- Biểu đồ 5: Trình độ Học vấn (Education) ---            
            st.subheader("5. Phân bổ theo Trình độ Học vấn")
            edu_counts = df_filtered['Education'].value_counts()
            fig_edu, ax_edu = plt.subplots(figsize=(8, 5))
            sns.barplot(
                x=edu_counts.index, y=edu_counts.values,
                palette='plasma', hue=edu_counts.index, ax=ax_edu
            )
            ax_edu.set_xlabel('Trình độ Học vấn')
            ax_edu.set_ylabel('Số lượng khách hàng')            
            st.pyplot(fig_edu)

            # --- Biểu đồ 7: Tổng số Con cái (Total_Children) ---            
            st.subheader("7. Phân bổ theo Tổng số Con cái")
            child_counts = df_filtered['Total_Children'].value_counts().sort_index()
            fig_child, ax_child = plt.subplots(figsize=(8, 5))
            sns.barplot(
                x=child_counts.index, y=child_counts.values,
                palette='magma', hue=child_counts.index, ax=ax_child
            )
            ax_child.set_xlabel('Tổng số Con cái')
            ax_child.set_ylabel('Số lượng khách hàng')
            st.pyplot(fig_child)           

        with col2:
            # --- Biểu đồ 2- Histogram: Phân phối Thu nhập (Income) ---
            st.subheader("2. Phân phối Chi tiết theo Thu nhập")
            fig_income_hist, ax_income_hist = plt.subplots(figsize=(8, 5))
            sns.histplot(data=df_filtered, x='Income', bins=30, kde=True, color='salmon', ax=ax_income_hist)
            ax_income_hist.set_xlabel('Thu nhập (USD)'); ax_income_hist.set_ylabel('Số lượng khách hàng')
            st.pyplot(fig_income_hist)

            # --- Biểu đồ 4: Phân khúc Thu nhập (Income_Group) ---
            st.subheader("4. Phân khúc theo Nhóm Thu nhập")            
            inc_counts = df_filtered['Income_Group'].value_counts()
            order_inc = ['Dưới 30k','30k-50k','50k-70k','70k-90k','Trên 90k']
            inc_counts = inc_counts.reindex(order_inc, fill_value=0)
            fig_income_bar, ax_income_bar = plt.subplots(figsize=(8, 5))
            sns.barplot(
                x=inc_counts.index, y=inc_counts.values,
                palette='mako', hue=inc_counts.index, ax=ax_income_bar,
            )
            ax_income_bar.set_xlabel('Phân khúc Thu nhập (USD)')
            ax_income_bar.set_ylabel('Số lượng khách hàng')
            st.pyplot(fig_income_bar)

            # --- Biểu đồ 6: Quốc gia (Country) ---
            st.subheader("6. Phân bổ theo Quốc gia")
            country_counts = df_filtered['Country'].value_counts()
            country_order = country_counts.index
            fig_country, ax_country = plt.subplots(figsize=(8, 5))
            sns.barplot(
                x=country_counts.index, y=country_counts.values,
                palette='crest', hue=country_counts.index, ax=ax_country,
                order=country_order
            )
            ax_country.set_xlabel('Quốc gia')
            ax_country.set_ylabel('Số lượng khách hàng')
            st.pyplot(fig_country)

            # --- Biểu đồ 8: Tình trạng Hôn nhân (Marital_Status) ---            
            st.subheader("8. Phân bổ theo Tình trạng Hôn nhân")
            marital_counts = df_filtered['Marital_Status'].value_counts()
            marital_order = marital_counts.index
            fig_marital, ax_marital = plt.subplots(figsize=(8, 5))
            sns.barplot(
                x=marital_counts.index, y=marital_counts.values,
                palette='Set2', hue=marital_counts.index, ax=ax_marital,
                order=marital_order
            )
            ax_marital.set_xlabel('Tình trạng Hôn nhân')
            ax_marital.set_ylabel('Số lượng khách hàng')            
            st.pyplot(fig_marital)

else:
    st.warning("Không thể tải dữ liệu để hiển thị trang này.")