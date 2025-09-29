import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
from pathlib import Path

# Azure OpenAI and AI Search
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SearchableField,
    SearchField,
    VectorSearchAlgorithmKind,
    HnswParameters,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch
)
from azure.core.credentials import AzureKeyCredential

# PDF processing
import PyPDF2
import tiktoken
import hashlib
import json
from typing import List, Dict
import numpy as np

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# Azure AI Search Configuration
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
SEARCH_INDEX_NAME = "security-policies-index"

# Initialize Azure OpenAI client
openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION
)

class DocumentProcessor:
    """PDF 문서를 처리하고 청크로 분할하는 클래스"""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """PDF에서 텍스트 추출"""
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def create_chunks(self, text: str) -> List[Dict]:
        """텍스트를 청크로 분할"""
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # 청크 ID 생성
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()
            
            chunks.append({
                "id": chunk_id,
                "content": chunk_text,
                "chunk_index": len(chunks)
            })
        
        return chunks

class AzureSearchManager:
    """Azure AI Search 관리 클래스"""
    
    def __init__(self):
        self.credential = AzureKeyCredential(SEARCH_KEY)
        self.index_client = SearchIndexClient(
            endpoint=SEARCH_ENDPOINT,
            credential=self.credential
        )
        self.search_client = None
        
    def create_index(self, recreate=False):
        """검색 인덱스 생성"""

        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,
                vector_search_profile_name="myHnswProfile"
            ),
            SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True)
        ]
        
        # Vector search configuration
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="myHnsw",
                    kind=VectorSearchAlgorithmKind.HNSW,
                    parameters=HnswParameters(
                        m=4,
                        ef_construction=400,
                        ef_search=500,
                        metric="cosine"
                    )
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="myHnswProfile",
                    algorithm_configuration_name="myHnsw",
                )
            ]
        )
        
        # Semantic search configuration
        semantic_config = SemanticConfiguration(
            name="my-semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                content_fields=[SemanticField(field_name="content")]
            )
        )
        
        semantic_search = SemanticSearch(configurations=[semantic_config])
        
        index = SearchIndex(
            name=SEARCH_INDEX_NAME,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search
        )
        
        try:
            self.index_client.create_or_update_index(index)
            st.success(f"인덱스 '{SEARCH_INDEX_NAME}' 생성/업데이트 완료")
        except Exception as e:
            st.error(f"인덱스 생성 실패: {str(e)}")
        
        self.search_client = SearchClient(
            endpoint=SEARCH_ENDPOINT,
            index_name=SEARCH_INDEX_NAME,
            credential=self.credential
        )
    
    def get_embedding(self, text: str) -> List[float]:
        """텍스트의 임베딩 벡터 생성"""
        response = openai_client.embeddings.create(
            model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            input=text
        )
        return response.data[0].embedding
    
    def upload_documents(self, chunks: List[Dict]):
        """문서 청크를 인덱스에 업로드"""
        documents = []
        
        with st.spinner("문서를 벡터화하고 있습니다..."):
            for chunk in chunks:
                embedding = self.get_embedding(chunk["content"])
                
                document = {
                    "id": chunk["id"],
                    "content": chunk["content"],
                    "content_vector": embedding,
                    "chunk_index": chunk["chunk_index"]
                }
                documents.append(document)
        
        # 배치로 업로드
        try:
            result = self.search_client.upload_documents(documents=documents)
            st.success(f"{len(documents)}개의 청크를 인덱스에 업로드했습니다.")
            return True
        except Exception as e:
            st.error(f"문서 업로드 실패: {str(e)}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """벡터 검색 수행"""
        query_embedding = self.get_embedding(query)
        
        results = self.search_client.search(
            search_text=query,
            vector_queries=[{
                "vector": query_embedding,
                "k_nearest_neighbors": top_k,
                "fields": "content_vector",
                "kind": "vector"
            }],
            select=["id", "content"],
            top=top_k
        )
        
        return [{"id": r["id"], "content": r["content"]} for r in results]

class SecurityChatbot:
    """보안 규정 챗봇 클래스"""
    
    def __init__(self, search_manager: AzureSearchManager):
        self.search_manager = search_manager
        self.system_prompt = """당신은 회사의 보안 규정 전문가입니다. 
        사용자가 제공한 코드나 상황에 대해 보안 규정 문서를 기반으로 답변하세요.
        
        다음 지침을 따르세요:
        1. 보안 규정 위반 사항을 명확히 지적하세요
        2. 해당 규정의 구체적인 내용을 인용하세요
        3. 개선 방안을 제시하세요
        4. 가능한 경우 수정된 코드 예제를 제공하세요
        5. 추가 보안 권고사항이 있다면 함께 제공하세요"""
    
    def get_security_code_examples(self, security_type: str) -> str:
        """보안 유형별 코드 예제 제공"""
        examples = {
            "encryption": """
# AES 암호화 예제
from cryptography.fernet import Fernet
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2

def generate_key(password: str) -> bytes:
    kdf = PBKDF2(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b'stable_salt',  # 실제로는 랜덤 salt 사용
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key

def encrypt_data(data: str, key: bytes) -> str:
    f = Fernet(key)
    encrypted = f.encrypt(data.encode())
    return encrypted.decode()

def decrypt_data(encrypted_data: str, key: bytes) -> str:
    f = Fernet(key)
    decrypted = f.decrypt(encrypted_data.encode())
    return decrypted.decode()

# 사용 예시
key = generate_key("your-secret-password")
credit_card = "1234-5678-9012-3456"
encrypted_cc = encrypt_data(credit_card, key)
print(f"암호화된 카드번호: {encrypted_cc}")
""",
            "hashing": """
# 비밀번호 해싱 예제
import hashlib
import secrets

def hash_password(password: str) -> tuple:
    salt = secrets.token_hex(16)
    pwdhash = hashlib.pbkdf2_hmac('sha256', 
                                   password.encode('utf-8'), 
                                   salt.encode('utf-8'), 
                                   100000)
    return salt, pwdhash.hex()

def verify_password(stored_password: str, stored_salt: str, provided_password: str) -> bool:
    pwdhash = hashlib.pbkdf2_hmac('sha256',
                                   provided_password.encode('utf-8'),
                                   stored_salt.encode('utf-8'),
                                   100000)
    return pwdhash.hex() == stored_password
""",
            "input_validation": """
# 입력 검증 예제
import re
from typing import Optional

def validate_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def sanitize_sql_input(user_input: str) -> str:
    # SQL 인젝션 방지를 위한 특수문자 제거
    dangerous_chars = ["'", '"', ';', '--', '/*', '*/', 'xp_', 'sp_']
    sanitized = user_input
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '')
    return sanitized

def validate_file_upload(filename: str, allowed_extensions: list) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Prepared Statement 사용 예제 (pymysql)
import pymysql

def safe_query(connection, user_id: int):
    with connection.cursor() as cursor:
        # 안전한 방법 - Prepared Statement
        sql = "SELECT * FROM users WHERE id = %s"
        cursor.execute(sql, (user_id,))
        result = cursor.fetchone()
    return result
"""
        }
        return examples.get(security_type, "")
    
    def generate_response(self, query: str, context: List[Dict]) -> str:
        """RAG 기반 응답 생성"""
        # 컨텍스트 준비
        context_text = "\n\n".join([f"[문서 {i+1}]\n{doc['content']}" 
                                    for i, doc in enumerate(context)])
        
        # 보안 위반 키워드 검색
        security_keywords = {
            "암호화": "encryption",
            "해싱": "hashing",
            "입력검증": "input_validation",
            "신용카드": "encryption",
            "비밀번호": "hashing",
            "SQL": "input_validation"
        }
        
        detected_type = None
        for keyword, sec_type in security_keywords.items():
            if keyword in query:
                detected_type = sec_type
                break
        
        # 프롬프트 구성
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
보안 규정 문서:
{context_text}

사용자 질문: {query}

위 보안 규정을 참고하여 답변해주세요. 
코드 개선이 필요한 경우 구체적인 예제를 제공해주세요."""}
        ]
        
        # Azure OpenAI 호출
        response = openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            temperature=0.3,
            max_tokens=2000
        )
        
        answer = response.choices[0].message.content
        
        # 보안 코드 예제 추가
        if detected_type:
            code_example = self.get_security_code_examples(detected_type)
            if code_example:
                answer += f"\n\n### 📝 추천 보안 코드 예제:\n```python{code_example}```"
        
        return answer
    
def main():
    st.set_page_config(
        page_title="사내 보안 규정 챗봇",
        page_icon="🔐",
        layout="wide"
    )
    
    st.title("🔐 사내 보안 규정 챗봇")
    st.markdown("PDF 형태의 사내 보안 규정 문서를 업로드하고, 코드나 상황에 대한 보안 검토를 받아보세요.")
    
    # 세션 상태 초기화
    if "search_manager" not in st.session_state:
        st.session_state.search_manager = AzureSearchManager()
        st.session_state.search_manager.create_index()
    
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = SecurityChatbot(st.session_state.search_manager)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "document_uploaded" not in st.session_state:
        st.session_state.document_uploaded = False
    
    # 사이드바 - 문서 업로드
    with st.sidebar:
        st.header("📄 보안 규정 문서 업로드")
        
        uploaded_file = st.file_uploader(
            "PDF 파일을 선택하세요",
            type=['pdf'],
            help="회사 보안 규정이 담긴 PDF 문서를 업로드하세요"
        )
        
        if uploaded_file is not None:
            if st.button("문서 처리 시작", type="primary"):
                with st.spinner("문서를 읽고 있습니다..."):
                    # 문서 처리
                    processor = DocumentProcessor()
                    text = processor.extract_text_from_pdf(uploaded_file)
                    chunks = processor.create_chunks(text)
                    
                    # 인덱스에 업로드
                    success = st.session_state.search_manager.upload_documents(chunks)
                    
                    if success:
                        st.session_state.document_uploaded = True
                        st.success("✅ 문서 업로드 완료!")
                        st.balloons()
        
        if st.session_state.document_uploaded:
            st.info("📚 보안 규정 문서가 로드되었습니다.")
        
        st.divider()
        
        # 예제 질문들
        st.header("💡 예제 질문")
        example_questions = [
            "신용카드 번호를 데이터베이스에 저장하려고 하는데 보안 규정은?",
            "사용자 비밀번호를 평문으로 저장하면 안되는 이유와 대안은?",
            "외부 API 호출 시 인증 토큰 관리 방법은?",
            "개인정보를 로그에 남기면 안되는 이유는?",
            "SQL 인젝션을 방지하는 방법은?"
        ]
        
        for q in example_questions:
            if st.button(q, key=f"example_{q[:20]}"):         
                st.session_state.messages.append({"role": "user", "content": q})

            
    # 메인 채팅 인터페이스
    st.header("💬 보안 규정")
    
    if not st.session_state.document_uploaded:
        st.warning("⚠️ 먼저 보안 규정 문서를 업로드해주세요.")
    
    # 채팅 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # 가장 최근 user 메시지에 대해서만 챗봇 응답 처리
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                last_user_message = st.session_state.messages[-1]["content"]
                with st.chat_message("assistant"):
                    with st.spinner("보안 규정을 검토중입니다..."):
                        search_results = st.session_state.search_manager.search(last_user_message, top_k=3)
                        if search_results:
                            response = st.session_state.chatbot.generate_response(last_user_message, search_results)
                            with st.expander("📖 참고한 보안 규정"):
                                for i, doc in enumerate(search_results, 1):
                                    st.markdown(f"**[참고 {i}]**")
                                    st.text(doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'])
                                    st.divider()
                        else:
                            response = "관련 보안 규정을 찾을 수 없습니다. 다른 질문을 해주세요."
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})            
           
    # 사용자 입력
    if prompt := st.chat_input("사내 보안 규정에 관한 질문을 입력하세요 (예: 이 코드의 보안 문제점은?)"):
        if not st.session_state.document_uploaded:
            st.error("문서를 먼저 업로드해주세요!")
        else:
            # 사용자 메시지 추가
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # AI 응답 생성
            with st.chat_message("assistant"):
                with st.spinner("보안 규정을 검토중입니다..."):
                    # 관련 문서 검색
                    search_results = st.session_state.search_manager.search(prompt, top_k=3)
                    
                    if search_results:
                        # 응답 생성
                        response = st.session_state.chatbot.generate_response(prompt, search_results)
                        
                        # 관련 문서 표시 (접을 수 있는 섹션)
                        with st.expander("📖 참고한 보안 규정"):
                            for i, doc in enumerate(search_results, 1):
                                st.markdown(f"**[참고 {i}]**")
                                st.text(doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'])
                                st.divider()
                    else:
                        response = "관련 보안 규정을 찾을 수 없습니다. 다른 질문을 해주세요."
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # 하단 정보
    with st.expander("ℹ️ 사용 방법"):
        st.markdown("""
        1. **문서 업로드**: 사이드바에서 보안 규정 PDF를 업로드하세요
        2. **질문하기**: 코드나 상황을 설명하고 보안 검토를 요청하세요
        3. **답변 확인**: AI가 관련 규정과 개선 방안을 제시합니다
        
        **지원 기능**:
        - 🔐 암호화 관련 규정 및 코드 예제
        - 🔑 인증/인가 보안 가이드
        - 💾 데이터 저장 보안 규정
        - 🛡️ 입력 검증 및 SQL 인젝션 방지
        - 📝 보안 코드 작성 가이드
        """)

if __name__ == "__main__":
    main()

