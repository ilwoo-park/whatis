# whatis - 상품 이미지 분석 프로그램

## 프로젝트 개요
Google ADK(Agent Development Kit)와 Gemini 2.5 Flash Lite 모델을 사용하여 상품 이미지를 분석하는 Python 프로그램.

## 기술 스택
- Python 3.10+
- google-adk (Google Agent Development Kit) https://google.github.io/adk-docs/
- 모델: gemini-2.5-flash-lite
- Vector DB: USearch (usearch) — 경량 벡터 검색 엔진
- Embedding: Gemini gemini-embedding-001 (768차원)
- 메타데이터: JSON 파일로 별도 관리 (datasets/vectordb/products_meta.json)

## Python Virtual enviroments
- C:\Users\ilwoo\Envs\gemini
