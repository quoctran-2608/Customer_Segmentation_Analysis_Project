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
    page_title="Phân khúc RFM",
    page_icon="✨",
    layout="wide"
)

# --- 3. Tải dữ liệu ---
df = load_data(DATA_PATH)

# --- 4. Tải và xử lý dữ liệu RFM ---
@st.cache_data
def load_and_process_rfm(df):
    try:
        # Sử dụng dataframe đã được tiền xử lý từ hàm load_data
          
        # --- TÍNH TOÁN CÁC CHỈ SỐ RFM ---
        # --- TẠO CÁC CỘT RFM  ---
        df_rfm = df[['ID', 'Recency', 'Total_NumberOfPurchases', 'Total_Spending']].copy()
        df_rfm.rename(columns={
            'Total_NumberOfPurchases': 'Frequency',
            'Total_Spending': 'Monetary'
        }, inplace=True)

        # Chấm điểm RFM (Sử dụng qcut + rank)
        df_rfm['R_Score'] = pd.qcut(df_rfm['Recency'].rank(method='first'), q=4, labels=[4, 3, 2, 1]).astype(int)
        df_rfm['F_Score'] = pd.qcut(df_rfm['Frequency'].rank(method='first'), q=4, labels=[1, 2, 3, 4]).astype(int)
        df_rfm['M_Score'] = pd.qcut(df_rfm['Monetary'].rank(method='first'), q=4, labels=[1, 2, 3, 4]).astype(int)

        # --- Hàm Phân khúc --
        def assign_segment_refined(row):
            R = row['R_Score']
            F = row['F_Score']
            M = row['M_Score']
            if R == 4 and F == 4 and M == 4: return 'Champions (VIP)'
            elif (R <= 2 and F >= 3 and M >= 3): return 'At Risk (Có nguy cơ rời bỏ)'
            elif (R >= 3 and F >= 3): return 'Loyal Customers (Trung thành)'
            elif (R >= 3 and F <= 2 and M >= 3): return 'Potential Loyalists (Tiềm năng)'
            elif R == 4 and F == 1: return 'New Customers (Khách hàng Mới)'
            elif R == 3 and F == 1: return 'Promising (Triển vọng)'
            elif (R >= 2 and R <= 3) and (F >= 2 and F <= 3): return 'Need Attention (Cần Chú ý)'
            elif R <= 2 and F == 1: return 'Hibernating (Đang ngủ đông)'
            elif R == 1: return 'Lost (Đã mất)'
            else: return 'Others'

        df_rfm['Segment'] = df_rfm.apply(assign_segment_refined, axis=1)

        # --- Gộp lại để có dữ liệu cuối cùng ---
        df_final = pd.merge(df, df_rfm[['ID', 'Segment']], on='ID')
        # Tạo cột TotalCampaignsAccepted
        campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
        df_final['TotalCampaignsAccepted'] = df_final[campaign_cols].sum(axis=1)

        # Đảm bảo các cột category có kiểu dữ liệu category
        cat_cols = ['Segment']
        for col in cat_cols:
             if col in df_final.columns:
                 # Chuyển đổi sang string trước để xử lý lỗi "Cannot setitem on a Categorical with a new category"
                 df_final[col] = df_final[col].astype(str)
                 df_final[col] = df_final[col].astype('category')

        return df_final
    except Exception as e:
        st.error(f"Lỗi khi tải hoặc xử lý dữ liệu RFM: {e}")
        return None

# Sử dụng hàm để tải và xử lý dữ liệu RFM
df_final_analysis = load_and_process_rfm(df)

# --- 4. Nội dung Trang Phân khúc RFM ---
st.title("✨ Phân khúc Khách hàng Chiến lược (RFM)")
st.markdown("""
Trang này trình bày kết quả phân khúc khách hàng bằng mô hình **RFM (Recency, Frequency, Monetary)**.
Sử dụng bộ lọc bên trái để xem xét các phân khúc trong các nhóm nhân khẩu học cụ thể (nếu muốn).
""")
st.markdown("---")

# Chỉ hiển thị nội dung nếu dữ liệu được tải thành công
if df_final_analysis is not None:

    # --- 5. Thêm Filter vào Sidebar ---
    st.sidebar.header("Bộ lọc Khám phá RFM:")

	# --- Filter Country ---
    country_options = sorted(df_final_analysis['Country'].unique().astype(str))
    selected_country = st.sidebar.multiselect('Chọn Quốc gia:', options=country_options, default=country_options)
    
    # --- Filter Education ---
    education_options = df_final_analysis['Education'].unique().astype(str)
    selected_education = st.sidebar.multiselect('Chọn Trình độ Học vấn:', options=education_options, default=education_options)
    
    # --- Filter Tình trạng hôn nhân ---
    marital_options = sorted(df_final_analysis['Marital_Status'].unique().astype(str))
    selected_marital = st.sidebar.multiselect('Chọn Tình trạng Hôn nhân:', options=marital_options, default=marital_options)
    
    # --- Filter Nhóm Tuổi ---
    # Lấy các nhóm tuổi duy nhất từ cột Age_Group đã tạo
    age_group_options = ['Dưới 30', '30-39', '40-49', '50-59', '60-69', 'Trên 70']
    # Lọc ra các nhóm tuổi thực sự có trong dữ liệu (sau khi lọc ban đầu nếu có)
    available_age_groups = [label for label in age_group_options if label in df_final_analysis['Age_Group'].unique().astype(str)]
    selected_age_group = st.sidebar.multiselect(
        'Chọn Nhóm Tuổi:',
        options=available_age_groups,
        default=available_age_groups
    )
    
    income_group_options = ['Dưới 30k', '30k-50k', '50k-70k', '70k-90k', 'Trên 90k']
    available_income_groups = [label for label in income_group_options if label in df_final_analysis['Income_Group'].unique().astype(str)]
    selected_income_group = st.sidebar.multiselect(
        'Chọn Nhóm Thu nhập (USD):',
        options=available_income_groups,
        default=available_income_groups
    )  

    # --- 6. Áp dụng Filter vào DataFrame ---
    df_filtered_rfm = df_final_analysis.copy()
    if selected_country:
        df_filtered_rfm = df_filtered_rfm[df_filtered_rfm['Country'].isin(selected_country)]
    if selected_education:
        df_filtered_rfm = df_filtered_rfm[df_filtered_rfm['Education'].isin(selected_education)]
    if selected_marital:
        df_filtered_rfm = df_filtered_rfm[df_filtered_rfm['Marital_Status'].isin(selected_marital)]
    if selected_age_group:
        df_filtered_rfm = df_filtered_rfm[df_filtered_rfm['Age_Group'].isin(selected_age_group)]
    if selected_income_group:
        df_filtered_rfm = df_filtered_rfm[df_filtered_rfm['Income_Group'].isin(selected_income_group)]


    st.write(f"Đang hiển thị dữ liệu cho **{len(df_filtered_rfm):,}** khách hàng.")
    st.markdown("---")

    if df_filtered_rfm.empty:
        st.warning("Không có khách hàng nào phù hợp với bộ lọc đã chọn.")
    else:
        # --- 7. Trực quan hóa Quy mô Phân khúc ---
        st.header("Bản đồ Phân khúc Khách hàng RFM (Quy mô)")
        ## Tính toán số lượng khách hàng trong mỗi phân khúc
        segment_counts_filtered = df_filtered_rfm['Segment'].value_counts()
        # Vẽ biểu đồ
        fig_segment_size, ax_segment_size = plt.subplots(figsize=(12, 8))
        sns.barplot(x=segment_counts_filtered.values, 
                    y=segment_counts_filtered.index, 
                    hue=segment_counts_filtered.index,
                    palette='Spectral', ax=ax_segment_size)        
        ax_segment_size.set_xlabel('Số lượng khách hàng')
        ax_segment_size.set_ylabel('Phân khúc')
        st.pyplot(fig_segment_size)
        st.markdown("---")

        # --- 8. So sánh Đặc điểm Chính Giữa các Phân khúc ---
        st.header("So sánh Đặc điểm Chính Giữa các Phân khúc")

        # Tính toán bảng profile DỰA TRÊN DỮ LIỆU ĐÃ LỌC
        if not df_filtered_rfm.empty:
            profile_comprehensive_filtered = df_filtered_rfm.groupby('Segment', observed=True).agg(
                Avg_Income=('Income', 'mean'), Avg_Age=('Age', 'mean'), Avg_Total_Children=('Total_Children', 'mean'),
                Avg_Tenure=('Customer_Tenure', 'mean'),
                Avg_CatalogPurchases=('NumCatalogPurchases', 'mean'), Avg_DealsPurchases=('NumDealsPurchases', 'mean'),
                Avg_WebVisits=('NumWebVisitsMonth', 'mean'), Avg_Campaigns_Accepted=('TotalCampaignsAccepted', 'mean'),
                Customer_Count=('ID', 'count')
            ).sort_values(by='Avg_Income', ascending=False)
            profile_to_plot_filtered = profile_comprehensive_filtered.reset_index().sort_values(by='Avg_Income', ascending=False)

            # Chia layout để hiển thị nhiều biểu đồ
            col1, col2 = st.columns(2)

            with col1:
                # Biểu đồ Tenure
                st.subheader("So sánh Thâm niên Trung bình")
                fig_tenure_comp, ax_tenure_comp = plt.subplots(figsize=(7, 5))
                sns.barplot(data=profile_to_plot_filtered, 
                            x='Avg_Tenure', y='Segment', 
                            hue='Segment',
                            palette='coolwarm', ax=ax_tenure_comp)
                ax_tenure_comp.set_xlabel('Thâm niên Trung bình (Ngày)')
                ax_tenure_comp.set_ylabel('Phân khúc')
                st.pyplot(fig_tenure_comp)

                # Biểu đồ Age
                st.subheader("So sánh Tuổi Trung bình")
                fig_age_comp, ax_age_comp = plt.subplots(figsize=(7, 5))
                sns.barplot(data=profile_to_plot_filtered, 
                            x='Avg_Age', y='Segment', 
                            hue='Segment',
                            palette='Blues', ax=ax_age_comp)
                ax_age_comp.set_xlabel('Tuổi Trung bình')
                ax_age_comp.set_ylabel('Phân khúc')
                st.pyplot(fig_age_comp)

                # Biểu đồ Catalog
                st.subheader("So sánh Mua qua Catalog (TB)")
                fig_cat_comp, ax_cat_comp = plt.subplots(figsize=(7, 5))
                sns.barplot(data=profile_to_plot_filtered, 
                            x='Avg_CatalogPurchases', y='Segment', 
                            hue='Segment',
                            palette='Blues_r', ax=ax_cat_comp)
                ax_cat_comp.set_xlabel('Số lần mua Catalog (TB)')
                ax_cat_comp.set_ylabel('Phân khúc')
                st.pyplot(fig_cat_comp)

                # Biểu đồ Marketing
                st.subheader("So sánh Tương tác Marketing (TB)")
                fig_mkt_comp, ax_mkt_comp = plt.subplots(figsize=(7, 5))
                sns.barplot(data=profile_to_plot_filtered, 
                            x='Avg_Campaigns_Accepted', y='Segment', 
                            hue='Segment',
                            palette='Greens_r', ax=ax_mkt_comp)
                ax_mkt_comp.set_xlabel('Số Chiến dịch Marketing đã chấp nhận (TB)')
                ax_mkt_comp.set_ylabel('Phân khúc')
                st.pyplot(fig_mkt_comp)

            with col2:
                # Biểu đồ Income
                st.subheader("So sánh Thu nhập Trung bình")
                fig_inc_comp, ax_inc_comp = plt.subplots(figsize=(7, 5))
                sns.barplot(data=profile_to_plot_filtered, 
                            x='Avg_Income', y='Segment',
                            hue='Segment', 
                            palette='RdYlGn', ax=ax_inc_comp)
                ax_inc_comp.set_xlabel('Thu nhập Trung bình (USD)')
                ax_inc_comp.set_ylabel('Phân khúc')
                st.pyplot(fig_inc_comp)

                
                # Biểu đồ Children
                st.subheader("So sánh Số con Trung bình")
                fig_child_comp, ax_child_comp = plt.subplots(figsize=(7, 5))
                sns.barplot(data=profile_to_plot_filtered, 
                            x='Avg_Total_Children', y='Segment', 
                            hue='Segment',
                            palette='Reds', ax=ax_child_comp)
                ax_child_comp.set_xlabel('Số con Trung bình')
                ax_child_comp.set_ylabel('Phân khúc')
                st.pyplot(fig_child_comp)

                 # Biểu đồ Deals
                st.subheader("So sánh Mua qua Deals (TB)")
                fig_deals_comp, ax_deals_comp = plt.subplots(figsize=(7, 5))
                sns.barplot(data=profile_to_plot_filtered, 
                            x='Avg_DealsPurchases', y='Segment', 
                            hue='Segment',
                            palette='Purples_r', ax=ax_deals_comp)
                ax_deals_comp.set_xlabel('Số lần mua Deals (TB)')
                ax_deals_comp.set_ylabel('Phân khúc')
                st.pyplot(fig_deals_comp)

                # Biểu đồ Web Visits
                st.subheader("So sánh Lướt Web/tháng (TB)")
                fig_web_comp, ax_web_comp = plt.subplots(figsize=(7, 5))
                sns.barplot(data=profile_to_plot_filtered, 
                            x='Avg_WebVisits', y='Segment', 
                            hue='Segment',
                            palette='Oranges_r', ax=ax_web_comp)
                ax_web_comp.set_xlabel('Số lượt Lướt Web/tháng (TB)')
                ax_web_comp.set_ylabel('Phân khúc')
                st.pyplot(fig_web_comp)

        else:
            st.write("Không đủ dữ liệu để tính toán và vẽ biểu đồ so sánh với bộ lọc hiện tại.")
        st.markdown("---")


        # --- 9: Khám phá Chân dung Chi tiết Từng Phân khúc ---
        st.header("Khám phá Chân dung Chi tiết Từng Phân khúc")
        ## Tạo dropdown để chọn phân khúc
        available_segments = sorted(df_filtered_rfm['Segment'].unique().astype(str))
        if available_segments:
            selected_segment = st.selectbox(
                "Chọn một Phân khúc để xem chi tiết:",
                options=available_segments
            )
            
            ## Hiển thị chân dung toàn diện và biểu đồ phân phối cho phân khúc đã chọn
            if selected_segment in profile_comprehensive_filtered.index:
                st.subheader(f"Chân dung Toàn diện: {selected_segment}")
                st.dataframe(profile_comprehensive_filtered.loc[[selected_segment]].T.rename(columns={selected_segment: 'Giá trị Trung bình / Phổ biến'}), width=500)

                df_segment_selected = df_filtered_rfm[df_filtered_rfm['Segment'] == selected_segment]
                if not df_segment_selected.empty:
                    st.subheader("Trực quan hóa Phân phối Đặc điểm Chính:")
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        # Biểu đồ Thâm niên
                        fig_tenure, ax_tenure = plt.subplots(figsize=(6, 4))
                        sns.histplot(df_segment_selected['Customer_Tenure'], kde=True, ax=ax_tenure, color='skyblue')
                        ax_tenure.set_title('Phân phối Thâm niên')
                        st.pyplot(fig_tenure)

                        # Biểu đồ Số lần mua Catalog
                        fig_catalog, ax_catalog = plt.subplots(figsize=(6, 4))
                        bins_cat = max(1, df_segment_selected['NumCatalogPurchases'].nunique())
                        sns.histplot(df_segment_selected['NumCatalogPurchases'], kde=False, ax=ax_catalog, color='blue', bins=bins_cat)
                        ax_catalog.set_title('Phân phối Số lần mua Catalog')
                        st.pyplot(fig_catalog)
                        
                        # Biểu đồ Số chiến dịch Marketing đã chấp nhận
                        fig_mkt, ax_mkt = plt.subplots(figsize=(6, 4))
                        bins_mkt = max(1, df_segment_selected['TotalCampaignsAccepted'].nunique())
                        sns.histplot(df_segment_selected['TotalCampaignsAccepted'], kde=False, ax=ax_mkt, color='green', bins=bins_mkt)
                        ax_mkt.set_title('Phân phối số chiến dịch Marketing đã chấp nhận')
                        st.pyplot(fig_mkt)

                    with col4:
                        # Biểu đồ Thu nhập
                        fig_income_seg, ax_income_seg = plt.subplots(figsize=(6, 4))
                        sns.histplot(df_segment_selected['Income'], kde=True, ax=ax_income_seg, color='lightcoral')
                        ax_income_seg.set_title('Phân phối Thu nhập')
                        st.pyplot(fig_income_seg)
                        
                        # Biểu đồ Số lần mua Deals
                        fig_deals, ax_deals = plt.subplots(figsize=(6, 4))
                        bins_deals = max(1, df_segment_selected['NumDealsPurchases'].nunique())
                        sns.histplot(df_segment_selected['NumDealsPurchases'], kde=False, ax=ax_deals, color='purple', bins=bins_deals)
                        ax_deals.set_title('Phân phối Số lần mua Deals')
                        st.pyplot(fig_deals)
                        
                        # Biểu đồ Số lượt Lướt Web/tháng
                        fig_web, ax_web = plt.subplots(figsize=(6, 4))
                        bins_web = max(1, df_segment_selected['NumWebVisitsMonth'].nunique())
                        sns.histplot(df_segment_selected['NumWebVisitsMonth'], kde=False, ax=ax_web, color='orange', bins=bins_web)
                        ax_web.set_title('Phân phối Số lượt Lướt Web/tháng')
                        st.pyplot(fig_web)

                else: st.write("Không có dữ liệu để vẽ biểu đồ cho phân khúc này.")
            else: st.write("Phân khúc đã chọn không có dữ liệu với bộ lọc hiện tại.")
        else: st.write("Không có phân khúc nào để hiển thị với bộ lọc hiện tại.")

else:
    st.warning("Không thể tải dữ liệu để hiển thị trang này.")