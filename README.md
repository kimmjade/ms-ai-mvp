# 🔐사내 개발 보안 규정 챗봇

## 💡프로젝트 개요
기업마다 다를 수 있는 **개발 보안 규정**을 쉽게 확인할 수 있는 **챗봇**을 개발하는 프로젝트입니다. 많은 개발자들이 일일이 보안 규정 문서를 찾아보는 것이 번거롭고 시간 소모적일 수 있기 때문에, 챗봇을 통해 규정을 확인할 수 있도록 도와줍니다.
## 🏘️아키텍처
1. 관리자가 사내 보안 규정에 관한 PDF 파일을 업로드
2. PDF파일을 벡터화(Azure openAI text embedding model - text-embedding-ada-002)
3. 벡터화된 문서를 인덱스에 업로드(Azure AI Search)
4. 사용자가 LLM 모델에 질문 (Azure openAI - gpt-4.1-mini)
5. Azure AI Search 가 벡터 검색 수행
6. LLM 이 검색 결과 응답

## 🛠️기술 스택
streamlit

python 3.12

Azure openAI (gpt-4.1-mini, text-embedding-ada-002)

Azure AI Search Service

## 🌐접속 링크
https://ktds613-webapp-01-a6d8a3dwaqgua3cf.canadacentral-01.azurewebsites.net/

## ⚙️로컬 실행 방법
```bash
# 패키지 설치
pip install -r requirements.txt
```
```bash
# streamlit 실행
streamlit run app.py
```
## 기능 설명
