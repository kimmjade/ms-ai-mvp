# 🔐사내 개발 보안 규정 챗봇
## 🌐접속 링크
https://ktds613-webapp-01-a6d8a3dwaqgua3cf.canadacentral-01.azurewebsites.net/

## 💡프로젝트 개요
기업마다 다를 수 있는 **개발 보안 규정**을 쉽게 확인할 수 있는 **챗봇**을 개발하는 프로젝트입니다. 많은 개발자들이 일일이 보안 규정 문서를 찾아보는 것이 번거롭고 시간 소모적일 수 있기 때문에, 챗봇을 통해 규정을 확인할 수 있도록 도와줍니다.
## 🏘️아키텍처
<img width="578" height="365" alt="image" src="https://github.com/user-attachments/assets/5202a2ac-6053-412f-a999-bf100c16a23b" />

1. 관리자가 사내 보안 규정에 관한 PDF 파일을 업로드
2. PDF파일을 벡터화(Azure openAI text embedding model - text-embedding-ada-002)
3. 벡터화된 문서를 인덱스에 업로드(Azure AI Search)
4. 사용자가 LLM 모델에 질문 (Azure openAI - gpt-4.1-mini)
5. Azure AI Search 가 벡터 검색 수행
6. LLM 이 검색 결과 응답

## 🛠️기술 스택
- streamlit
- python 3.12
- Azure openAI (gpt-4.1-mini, text-embedding-ada-002)
- Azure AI Search Service
- Azure webapp

## ⚙️로컬 실행 방법
1. root directory에 .env 파일 생성
```
AZURE_OPENAI_ENDPOINT=azure_openai_endpoint
AZURE_OPENAI_KEY=azure_openai_api_key
AZURE_OPENAI_DEPLOYMENT=azure_openai_deployment
AZURE_OPENAI_API_VERSION=azure_openai_api_version
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=azure_openai_embedding_deployment
AZURE_SEARCH_ENDPOINT=azure_search_endpoint
AZURE_SEARCH_KEY=azure_search_key
```
2. python 패키지 설치
```bash
pip install -r requirements.txt
```
3. streamlit 실행
```bash
streamlit run app.py
```

## 📝향후 개선
1. 관리자 페이지 개발
   - 기업 보안 규정 문서는 사내 관리자만 업로드를 진행하지만, 지금은 관리자 페이지가 따로 없어 사실 모든 사용자가 파일 업로드가 가능
2. 인덱스에 문서 업로드
   - 파일을 새로 업로드할때 인덱스의 내용을 전체 삭제하고 업로드 할 지, 기존의 것에 합쳐서 업로드 할지 고민
