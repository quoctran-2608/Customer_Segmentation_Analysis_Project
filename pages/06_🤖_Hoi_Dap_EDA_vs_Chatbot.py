### --- 0. IMPORT CÁC THƯ VIỆN CẦN THIẾT ---
import os 
import streamlit as st 

# Thư viện hỗ trợ load biến môi trường từ file .env
from dotenv import load_dotenv

# Thư viện hỗ trợ load dữ liệu từ file Notebook .ipynb
from langchain_community.document_loaders import NotebookLoader

# Thư viện chia văn bản lớn thành các phân đoạn nhỏ hơn để xử lý hiệu quả hơn.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Thư viện hỗ trợ tạo embeddings sử dụng mô hình AITeamVN/Vietnamese_Embedding trên HuggingFace.
from langchain_huggingface import HuggingFaceEmbeddings

# Thư viện hỗ trợ lưu trữ và truy xuất vector database sử dụng FAISS.
from langchain_community.vectorstores import FAISS 

# Thư viện hỗ trợ tạo mẫu prompt cho mô hình ngôn ngữ
from langchain.prompts import PromptTemplate 

# Thư viện hỗ trợ tạo chuỗi hỏi đáp (question answering chain)
from langchain.chains.question_answering import load_qa_chain

# Thư viện Google Generative AI để gọi API
import google.generativeai as genai 
# Thư viện langchain để tích hợp với Google Generative AI
from langchain_google_genai import ChatGoogleGenerativeAI

# Thư viện hỗ trợ truy xuất đa truy vấn (Multi-Query Retriever)
from langchain.retrievers.multi_query import MultiQueryRetriever

# Thư viện hỗ trợ lưu trữ lịch sử hội thoại (conversation memory)
from langchain.memory import ConversationBufferMemory

# Thư viện hỗ trợ tạo chuỗi hỏi đáp với khả năng truy xuất thông tin và sử dụng memory
from langchain.chains import ConversationalRetrievalChain

# Sử dụng các hàm dự đoán trong utils.py
from utils import load_ols_model, predict_spending

# Cấu hình đường dẫn lưu trữ vector database
from config import VECTOR_STORE_PATH

# --- 1. CÀI ĐẶT vs KHỞI TẠO MÔI TRƯỜNG LÀM VIỆC VỚI GOOGLE API KEY ---
# load biến môi trừờng từ file .env
load_dotenv()

# Lấy GOOGLE_API_KEY từ biến môi trường
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Không tìm thấy key")
    st.stop()
    # Nếu không có API key thì dừng chương trình lại

# Cài đặt API key cho Google Genai, khởi tạo môi trường
genai.configure(api_key = api_key)


# --- 2. HÀM XÂY DỰNG KHO DỮ LIỆU VECTOR KẾT QUẢ EDA CỦA DATASET  ---
@st.cache_resource
def build_EDA_results_store():
    try:
        # Bước 1: Load từ .ipynb, bao gồm cả outputs     
        loader = NotebookLoader(
            path='./Customer_Segmentation_EDA.ipynb',
            include_outputs=True,  # Load outputs từ code cells (text, HTML sẽ thành string)
            max_output_length=3000,  # Giới hạn độ dài output để tránh quá lớn
            remove_newline=True  # Xóa newline thừa cho clean text
        )
        # Load toàn bộ notebook thành Documents (source + outputs nếu có)
        documents = loader.load() 

        # Bước 2: Split văn bản thành các đoạn nhỏ hơn
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Kích thước mỗi chunk
            chunk_overlap=300,  # Overlap để giữ ngữ cảnh
        )
        split_docs = text_splitter.split_documents(documents)
        
        # Bước 3: Vector hóa và lưu vào FAISS (dùng AITeamVN/Vietnamese_Embedding trên HuggingFace)
        ## Tạo embeddings sử dụng mô hình đa ngôn ngữ
        embed_model = HuggingFaceEmbeddings(
            model_name="AITeamVN/Vietnamese_Embedding",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )  
        
        ## Tạo vector store từ các document đã chia nhỏ
        vector_store = FAISS.from_documents(split_docs, embed_model)
        ## Hiển thị thông tin số đoạn đã load và vector hóa
        st.info(f"Đã load và vector hóa {len(split_docs)} đoạn insight (dùng AITeamVN/Vietnamese_Embedding Embeddings).")
        
        ## Lưu vector store vào local
        vector_store.save_local("faiss_index")  # Lưu vào thư mục faiss_index 
  
        st.success("Dữ liệu Kinh doanh đã được phân tích xong, sẵn sàng để trả lời câu hỏi")     
    except Exception as e:
        st.error(f"Lỗi lưu vector database: {str(e)}")
 
# # Hàm kiểm tra VectorDB đã tồn tại chưa
def check_EDA_results_store(): 
    return True if os.path.exists(VECTOR_STORE_PATH) else False

# --- 3. HÀM TẠO CHUỖI HỎI ĐÁP CONVERSATIONAL VỚI MEMORY ---
def get_conversational_chain(retriever, memory):    
    prompt_template = """
    Bạn là một Trợ lý AI chuyên phân tích dữ liệu. 
    Nhiệm vụ của bạn là trả lời chi tiết câu hỏi của người dùng DỰA HOÀN TOÀN VÀO NỘI DUNG (Context) được cung cấp, kết hợp với lịch sử chat nếu cần để infer thêm.
    Nội dung này được trích xuất từ các kết luận (insight) trong một file Jupyter Notebook "Customer_Segmentation_EDA.ipynb".
    
    Hãy đọc kỹ Context và lịch sử chat, trả lời rõ ràng, súc tích. Nếu câu hỏi follow-up (như 'liệt kê đầy đủ'), hãy liên kết với query trước.
    
    QUAN TRỌNG: 
    1- Nếu câu trả lời không có trong Context, hãy thử paraphrase hoặc dùng history để tìm thêm. Nếu vẫn không, nói: "Xin lỗi, tôi không tìm thấy thông tin chính xác, nhưng dựa trên insight gần nhất: [giải thích ngắn]". Không được bịa thông tin.
    2- Câu trả lời đưa ra phải bằng tiếng Việt.
    3- Câu trả lời cần đúng ngữ cảnh là đang trong cuộc trò chuyện, chứ không được đưa ra từ ngữ dùng trong ngữ cảnh đang đọc tài liệu. Ví dụ, không nói kiểu "xem lại các phần trước của phân tích để biết thêm chi tiết", "được trích từ phần tổng kết C",...
    Context:
    {context}
    
    Question:
    {question}
    
    Answer (trả lời bằng tiếng Việt):
    """    
    try:
        # Khởi tạo mô hình LLM của Google Gemini
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
        
        # Định nghĩa Prompt Template
        qa_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])        
        
        # Tạo chuỗi conversational retrieval với memory
        chain = ConversationalRetrievalChain.from_llm(
            llm=model,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=False  # Không cần trả về source docs
        )
                
        return chain            
    except Exception as e:
        st.error(f"Lỗi tạo chuỗi hỏi đáp: {str(e)}")
    return None

# --- HÀM DETECT USER INTENT SỬ DỤNG GEMINI ---
def detect_intent(user_question):
    try:
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
        prompt = f"""
        Phân loại intent của câu hỏi sau: '{user_question}'
        Nếu intent là 'dự đoán chi tiêu' hoặc liên quan đến dự đoán chi tiêu khách hàng, trả về 'prediction'.
        Nếu không, trả về 'general'.
        Chỉ trả về 'prediction' hoặc 'general', không giải thích.
        """
        response = model.invoke(prompt)
        intent = response.content.strip().lower()
        return intent == 'prediction'
    except Exception as e:
        st.error(f"Lỗi detect intent: {str(e)}")
        return False

# --- 4. HÀM NHẬN VÔ CÂU HỎI CỦA USER VÀ XỬ LÝ, TRẢ VỀ CÂU TRẢ LỜI
def user_input(user_question):
    try:        
        #*** KIẾM TRA XEM CHẾ ĐỘ DỰ ĐOÁN ĐÃ ĐƯỢC KÍCH HOẠT CHƯA ***
        if st.session_state.get('prediction_mode', False):
            current_step = st.session_state.get('prediction_step', 'income')
            
            if current_step == 'income':
                try:
                    income = float(user_question)
                    if income < 0:
                        raise ValueError
                    st.session_state.prediction_data['income'] = income
                    st.session_state.prediction_step = 'total_children'
                    return "Vui lòng cho biết tổng số con (số nguyên, ví dụ 0,1,2,...):"
                except ValueError:
                    return "Thu nhập phải là số thực không âm. Vui lòng nhập lại."
            
            elif current_step == 'total_children':
                try:
                    total_children = int(user_question)
                    if total_children < 0:
                        raise ValueError
                    st.session_state.prediction_data['total_children'] = total_children
                    st.session_state.prediction_step = 'customer_tenure'
                    return "Vui lòng cho biết thâm niên (số ngày, ví dụ 365):"
                except ValueError:
                    return "Tổng số con phải là số nguyên không âm. Vui lòng nhập lại."
            
            elif current_step == 'customer_tenure':
                try:
                    customer_tenure = int(user_question)
                    if customer_tenure < 0:
                        raise ValueError
                    st.session_state.prediction_data['customer_tenure'] = customer_tenure
                    
                    # Dự đoán (gọi từ utils.py)
                    data = st.session_state.prediction_data
                    result = predict_spending(data['income'], data['total_children'], data['customer_tenure'])
                    
                    # Reset trạng thái
                    st.session_state.prediction_mode = False
                    del st.session_state.prediction_step
                    del st.session_state.prediction_data
                    
                    if result is not None:
                        return f"Kết quả dự đoán chi tiêu: {result} USD."
                    else:
                        return "Lỗi trong quá trình dự đoán. Vui lòng thử lại."
                except ValueError:
                    return "Thâm niên phải là số nguyên không âm. Vui lòng nhập lại."
        
        #*** GỌI HÀM DETECT INTENT ĐỂ XEM CÓ PHẢI DỰ ĐOÁN CHI TIÊU KHÔNG ***
        # (Lẽ ra phần này sẽ đặt trước "KIẾM TRA XEM CHẾ ĐỘ DỰ ĐOÁN ĐÃ ĐƯỢC KÍCH HOẠT CHƯA"
        # Tuy nhiên để tránh việc lặp lại detect intent nhiều lần khi user đang nhập dữ liệu (income, children, tenure))
        if detect_intent(user_question): 
        # Nếu đúng là intent dự đoán chi tiêu
            ## Load mô hình OLS chỉ khi cần (lần đầu tiên) và cache qua session_state
            if not st.session_state.get('ols_loaded', False):
                with st.spinner("Đang tải mô hình dự đoán (chỉ lần đầu)..."):
                    ols_results, scaler_ols, feature_cols_ols = load_ols_model()
                    if ols_results is None:
                        return "Lỗi tải mô hình dự đoán. Vui lòng thử lại sau."
                    st.session_state.ols_results = ols_results
                    st.session_state.scaler_ols = scaler_ols
                    st.session_state.feature_cols_ols = feature_cols_ols
                    st.session_state.ols_loaded = True
            
            ## Khởi động chế độ dự đoán
            st.session_state.prediction_mode = True
            st.session_state.prediction_step = 'income'
            st.session_state.prediction_data = {}
            return "Để dự đoán chi tiêu người dùng, vui lòng cung cấp thu nhập (số thực không âm, USD):"       
        
        
        #*** CHẾ ĐỘ HỎI ĐÁP BÌNH THƯỜNG NẾU KHÔNG PHẢI PREDICTION***
        ## Load embed_model và vector_store chỉ khi cần, cache trong session_state
        if 'embed_model' not in st.session_state:
            with st.spinner("Đang tải embedding model lần đầu (sẽ nhanh hơn lần sau)..."):
                st.session_state.embed_model = HuggingFaceEmbeddings(
                    model_name="AITeamVN/Vietnamese_Embedding",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True}
                )        
        embed_model = st.session_state.embed_model
        
        ## Load vector_store từ session_state nếu đã có, nếu chưa thì load từ local
        if not check_EDA_results_store():
            st.error("EDA Results Vector store không tồn tại. Hãy tạo trước!")        
       
        if 'vector_store' not in st.session_state:
            with st.spinner("Đang tải vector store lần đầu..."):
                st.session_state.vector_store = FAISS.load_local(
                    VECTOR_STORE_PATH, embed_model, allow_dangerous_deserialization=True
                )            
        vector_store = st.session_state.vector_store

        # Lấy retriever từ vector store (tìm 7 kết quả liên quan nhất)
        base_retriever = vector_store.as_retriever(search_kwargs={"k": 7})
        
        # Tạo MultiQueryRetriever để LLM tự generate 3-5 variant queries
        # Khởi tạo LLM cho MultiQuery (dùng cùng Gemini model)
        llm_for_multi = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
        
        # Custom prompt để generate 3-5 variants bằng tiếng Việt
        multi_prompt = PromptTemplate.from_template(
            "You are an AI assistant. Suggest 3 to 5 alternative questions in Vietnamese that capture different perspectives on the user question: {question}. Output each on a new line."
        )
        
        multi_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,  # Wrap retriever hiện tại
            llm=llm_for_multi,  # LLM để generate variants
            prompt=multi_prompt  # Sử dụng custom prompt
        )
        
        # Sử dụng multi_retriever thay vì base_retriever
        retriever = multi_retriever
        
        # Lấy memory từ session_state (sẽ được khởi tạo ở phần chính)
        memory = st.session_state.memory
        
        # Lấy chuỗi hỏi đáp conversational
        qa_chain = get_conversational_chain(retriever, memory)
        
        ## nếu không có chain thì trả về rỗng
        if not qa_chain:
            return
        
        # Tạo câu trả lời dựa trên câu hỏi và lịch sử chat (memory sẽ tự xử lý)
        response = qa_chain({"question": user_question})
       
        return response['answer']
    
    except Exception as e:
        st.error(f"Lỗi xử lý câu hỏi: {str(e)}")  
    return None
    
# --- 5. GIAO DIỆN STREAMLIT CHO ỨNG DỤNG CHATBOT HỎI ĐÁP VỀ KẾT QUẢ EDA ---

st.set_page_config(page_title="Chatbot Giải thích Insight EDA", layout="wide")

st.title("🤖 Chatbot Giải thích Insight EDA về Chân dung khách hàng")
try: 
    # Kiểm tra kho dữ liệu "EDA Results Vector Database" đã tồn tại chưa   
    is_has_EDA_results_store = check_EDA_results_store()    
    if is_has_EDA_results_store:
        st.success("✅ EDA Results Vector Database đã tạo thành công. Bạn có thể hỏi đáp ngay bây giờ!")
    else:        
        # Nút để xây dựng kho dữ liệu vector từ kết quả EDA
        st.error("⚠️ Chưa tìm thấy EDA Results Vector Database. Vui lòng bấm nút bên dưới để tạo kho dữ liệu 'EDA RESULTS VECTOR DATABASE' từ file 'Customer_Segmentation_EDA.ipynb' trước khi hỏi đáp.")
        if st.button("👉 BẮT ĐẦU TẠO 'EDA RESULTS VECTOR DATABASE'"):
            build_EDA_results_store()           
    
    # Khởi tạo lịch sử chat
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Tôi là AI Assistant phân tích dữ liệu. Hãy hỏi tôi bất cứ điều gì về tập dữ liệu \"marketing_data_with_missing_values.csv\" và tôi sẽ trả lời dựa trên các kết luận (insight) đã được phân tích. Bạn cũng có thể hỏi về dự đoán chi tiêu người dùng để kích hoạt tính năng dự đoán."}]
    
    # Khởi tạo memory cho conversational chain
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
    # Khởi tạo trạng thái cho prediction vs ols_loaded nếu chưa có
    if "prediction_mode" not in st.session_state:
        st.session_state.prediction_mode = False
    if "prediction_data" not in st.session_state:
        st.session_state.prediction_data = {}
    if "prediction_step" not in st.session_state:
        st.session_state.prediction_step = 'income'
    if "ols_loaded" not in st.session_state:
        st.session_state.ols_loaded = False
        
    # Hiển thị các tin nhắn cũ
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Nhận input từ người dùng
    if prompt := st.chat_input("Nhập câu hỏi của bạn về kết quả EDA ở đây: Ví dụ, 'Có các phân khúc khách hàng nào'? Hoặc hỏi về dự đoán chi tiêu"):
        
        # 1. Thêm câu hỏi người dùng vào lịch sử và hiển thị ngay
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # 2. Xử lý câu hỏi và hiển thị câu trả lời của AI
        assistant_response = ""         
        with st.chat_message("assistant"):
            with st.spinner("Chatbot đang suy nghĩ..."):
                if is_has_EDA_results_store:
                    # Gọi hàm xử lý câu hỏi của user, và nhận câu trả lời của bot
                    assistant_response = user_input(prompt)
                    
                    # Hiển thị câu trả lời của bot
                    st.markdown(assistant_response)                
                else:
                    st.warning("Vui lòng tạo 'EDA Results Vector Database' trước khi hỏi đáp!")
                
        # 3. Thêm câu trả lời của bot vào lịch sử chat
        if assistant_response:
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

except Exception as e:
    st.error(f"Đã xảy ra lỗi: {e}")
    st.error("LỖI: Vui lòng kiểm tra lại GOOGLE_API_KEY của bạn và đảm bảo file 'Customer_Segmentation_EDA.ipynb' ở cùng thư mục.")
