# whatis

상품 이미지를 분석하여 상품명, 브랜드, 카테고리 등의 정보를 자동으로 추출하는 CLI 프로그램입니다.

Google ADK(Agent Development Kit)를 기반으로 2개의 AI 에이전트가 순차적으로 동작합니다.

## 아키텍처

```
SequentialAgent (product_analyzer)
├── 1. image_analyzer  (gemini-2.5-flash-lite)  — 이미지 분석, 도구 없음
└── 2. rag_agent       (gemini-2.5-flash)       — RAG 보완 + 최종 JSON 출력
```

| Agent | 모델 | 역할 |
|-------|------|------|
| `image_analyzer` | gemini-2.5-flash-lite | 이미지에서 시각 정보 추출 (저비용) |
| `rag_agent` | gemini-2.5-flash | 브랜드/상품명 미식별 시 로컬 DB 검색 또는 Google Search로 보완 |

### 동작 흐름

1. `image_analyzer`가 이미지를 분석하고 결과를 session state에 저장
2. `rag_agent`가 분석 결과를 읽고:
   - 브랜드/상품명이 식별된 경우 → 그대로 출력 (`source: image`)
  - 미식별인 경우 → 로컬 DB 검색 (`source: local_db`) → 없으면 Google Search (`source: google_search`)
  - RAG를 사용한 경우 (`source: local_db` 또는 `source: google_search`) 확률 정보 `rag_confidence` 포함
3. Google Search로 찾은 상품 정보는 로컬 DB에 자동 저장 (다음 분석 시 활용)

## 설치

### 요구 사항

- Python 3.10+
- Google API Key ([Google AI Studio](https://aistudio.google.com/)에서 발급)

### 설정

```bash
pip install -r requirements.txt
```

프로젝트 루트에 `.env` 파일을 생성하고 API 키를 입력합니다:

```
GOOGLE_API_KEY=your_api_key_here
```

## 사용법

```
python main.py <이미지_파일_또는_디렉토리> [샘플 수] [옵션]
```

### 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--country CODE` | 국가 코드 | `KR` |
| `--lang CODE` | 언어 코드 | `ko` |
| `--random` | 랜덤 샘플 선택 | - |

### 예시

```bash
# 단일 이미지 분석
python main.py product.jpg

# 디렉토리 내 전체 이미지 분석
python main.py datasets/images

# 디렉토리에서 5개 샘플 분석
python main.py datasets/images 5

# 랜덤으로 3개 선택하여 분석
python main.py datasets/images 3 --random

# 영어로 분석 결과 출력
python main.py datasets/images 3 --country US --lang en

# 일본어로 분석 결과 출력
python main.py datasets/images 3 --country JP --lang ja
```

### 실행 출력 예시

```
설정: country=KR, lang=ko

총 3개 이미지 분석 (랜덤 샘플)

[1/3] 분석 중: image27.jpg ...
  >> [image_analyzer] 호출됨
  >> [rag_agent] 호출됨
[1/3] 완료: image27.jpg (4.52s)

[2/3] 분석 중: image85.jpg ...
  >> [image_analyzer] 호출됨
  >> [rag_agent] 호출됨
     -> tool: search_local_db()
  -> tool: google_search_agent()
     -> tool: save_to_local_db()
[2/3] 완료: image85.jpg (8.31s)
```

## 출력 형식

분석 결과는 JSON으로 출력되며, `outputs/` 디렉토리에 타임스탬프 파일로 자동 저장됩니다.

```json
{
  "product_name": "Burrstkuchl 소시지",
  "category": "Food",
  "brand": "Burrstkuchl",
  "image_features": "두 개의 링 모양으로 연결된 갈색 소시지가 투명 포장에 담겨 있음",
  "key_features": ["소시지", "고리 모양", "갈색", "투명 포장", "Durstkuch!"],
  "expiration_date": "",
  "source": "local_db",
  "rag_confidence": {
    "probability": 0.57,
    "method": "local_db_score",
    "evidence": "로컬 DB의 키워드/상품 유형 일치"
  },
  "inference_time": "8.66s"
}
```

| 필드 | 설명 |
|------|------|
| `product_name` | 상품명 |
| `category` | 카테고리 (영문) |
| `brand` | 브랜드명 (미식별 시 빈 문자열) |
| `image_features` | 시각적 특징 요약 |
| `key_features` | 주요 특징 리스트 (이미지에서 인식된 텍스트 포함) |
| `expiration_date` | 유통기한 (이미지에 표시된 경우) |
| `source` | 정보 출처 — `image`, `local_db`, `google_search` |
| `rag_confidence` | RAG 사용 시 신뢰도 정보 (`source`가 `local_db`/`google_search`일 때만 포함) |
| `inference_time` | 분석 소요 시간 |

## 프로젝트 구조

```
whatis/
├── main.py                  # CLI 진입점
├── analyzer/
│   ├── agent.py             # SequentialAgent 정의 (image_analyzer + rag_agent)
│   └── tools.py             # 로컬 DB 검색/저장 도구
├── datasets/
│   ├── images/              # 분석할 상품 이미지
│   └── products_db.json     # 로컬 상품 DB (자동 생성)
├── outputs/                  # 분석 결과 JSON 저장 (자동 생성)
├── requirements.txt
└── .env                     # Google API Key (직접 생성)
```
