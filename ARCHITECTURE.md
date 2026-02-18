# whatis 아키텍처 문서

## 1. 전체 구조 개요

```mermaid
flowchart TD
    CLI(["사용자 CLI 입력"]) --> MAIN

    subgraph MAIN["main.py — CLI 진입점 + 배치 처리"]
        M1["이미지 파일 로딩 (bytes + MIME)"]
        M2["InMemoryRunner 생성 + 세션 관리"]
        M3["재시도 로직 (최대 3회 시도)"]
        M4["결과 JSON 파싱 + outputs/ 저장"]
        M1 --> M2 --> M3 --> M4
    end

    MAIN -->|"ADK runner.run_async()"| SEQ

    subgraph SEQ["SequentialAgent: product_analyzer — analyzer/agent.py"]
        IA["Sub-Agent 1: image_analyzer\n모델: gemini-2.5-flash-lite  /  도구: 없음\n이미지 시각 분석 → output_key: image_analysis"]
        RA["Sub-Agent 2: rag_agent\n모델: gemini-2.5-flash  /  도구: 3개\nRAG 보완 + 최종 JSON 출력"]
        IA -->|"session state 전달"| RA
    end

    RA --> TOOLS

    subgraph TOOLS["analyzer/tools.py — Vector DB 도구"]
        SL["search_local_db()\nkey_features 기반 코사인 검색"]
        SV["save_to_local_db()\n신규 상품 임베딩 후 저장"]
        EMB["Gemini gemini-embedding-001\n768차원 임베딩"]
        US[("USearch Index\ncosine / 768d")]
        META[("products_meta.json\nkey → product info")]
        SL & SV --> EMB
        EMB --> US
        EMB --> META
    end
```

---

## 2. 컴포넌트별 상세 설명

### 2.1 `main.py` — CLI 진입점

| 역할 | 상세 |
|------|------|
| 인수 파싱 | `image/directory`, `sample_count`, `--country`, `--lang`, `--random` |
| 이미지 로딩 | `load_image_as_part()`: 파일을 `bytes` 로 읽어 `types.Part(inline_data=Blob)` 생성 |
| 해상도 출력 | Pillow `Image.open()`으로 WxH 확인 |
| ADK 세션 | `InMemoryRunner` + `session_service.create_session()`, state에 `country/lang` 전달 |
| 이벤트 스트림 | `runner.run_async()` 이벤트를 순회, `author` 변경 시 에이전트 호출 출력, `function_call` 도구 이름 출력 |
| 토큰 집계 | 이벤트의 `usage_metadata`를 누적 합산 |
| 재시도 | `MAX_RETRIES=2`, 3회 시도, `RETRY_DELAY=3`초 대기 |
| 결과 저장 | `outputs/result_YYYYMMDD_HHMMSS.json` 자동 저장 |

**배치 처리 흐름:**

```mermaid
flowchart TD
    START(["images 순회 시작"]) --> AS["analyze_single()"]
    AS --> AI["analyze_image()\n재시도 포함 (최대 3회)"]
    AI --> ADK["ADK InMemoryRunner\n(SequentialAgent 실행)"]
    ADK --> PARSE["JSON 파싱\n+ inference_time + token_usage 추가"]
    PARSE --> OK{성공?}
    OK -->|Yes| RECORD["결과 results 에 추가"]
    OK -->|No| FAIL["{'error': 'analysis_failed'} 기록"]
    FAIL --> RECORD
    RECORD --> MORE{"다음 이미지?"}
    MORE -->|Yes| AS
    MORE -->|No| SAVE["outputs/result_*.json 저장"]
```

---

### 2.2 `analyzer/agent.py` — 에이전트 정의

#### Sub-Agent 1: `image_analyzer`

```
모델  : gemini-2.5-flash-lite   (저비용, 이미지 분석 특화)
도구  : 없음
입력  : 사용자 메시지 + 이미지 (inline bytes)
출력  : JSON → session state["image_analysis"] 에 저장 (output_key)
```

**출력 스키마:**
```json
{
  "product_name": "상품명",
  "product_name_confidence": 0.0 ~ 1.0,
  "category": "영문 카테고리",
  "brand": "브랜드명",
  "brand_confidence": 0.0 ~ 1.0,
  "image_features": "시각적 특징 요약",
  "key_features": ["인식된 텍스트/특징 리스트"],
  "expiration_date": "유통기한 또는 빈 문자열"
}
```

**신뢰도(confidence) 기준:**
| 값 | 의미 |
|----|------|
| 1.0 | 텍스트/로고가 이미지에서 명확히 판독됨 |
| 0.7–0.9 | 일부 가려지거나 흐릿하지만 높은 확신 |
| 0.4–0.6 | 포장 스타일/색상 등 시각적 맥락으로 추론 |
| 0.1–0.3 | 일반적 외형 기반 낮은 확신 |
| 0.0 | 식별 불가 |

---

#### Sub-Agent 2: `rag_agent`

```
모델  : gemini-2.5-flash   (function calling 지원)
도구  : search_local_db, save_to_local_db, GoogleSearchTool
입력  : {image_analysis} — session state 템플릿 변수
출력  : 최종 JSON (source + rag_confidence 포함)
```

**RAG 판단 로직:**

```mermaid
flowchart TD
    START(["image_analysis 읽기"]) --> CHECK{"product_name_confidence > 0.7\nAND brand_confidence > 0.7?"}
    CHECK -->|YES| SRC_IMAGE["source = image\nrag_confidence 미포함"]
    CHECK -->|NO| SDB["search_local_db(key_features)"]
    SDB --> FOUND{"found = true?"}
    FOUND -->|YES| SRC_LOCAL["source = local_db\nrag_confidence.probability = score\nmethod = local_db_score"]
    FOUND -->|NO| GSEARCH["GoogleSearchTool 호출"]
    GSEARCH --> SAVE["save_to_local_db()"]
    SAVE --> SRC_GOOGLE["source = google_search\nrag_confidence.probability = 추정값\nmethod = google_search_estimate"]
```

**`source` 값별 출력 규칙:**

| source | rag_confidence | 설명 |
|--------|---------------|------|
| `image` | 미포함 | 이미지에서 직접 식별 (high confidence) |
| `local_db` | 포함 | Vector DB 코사인 유사도 score 직접 사용 |
| `google_search` | 포함 | 근거 일치도 기반 0~1 추정값 |

---

#### `root_agent` — SequentialAgent

```python
root_agent = SequentialAgent(
    name="product_analyzer",
    sub_agents=[image_analyzer, rag_agent],
)
```

ADK `SequentialAgent`는 sub_agents를 순서대로 실행하며, 이전 에이전트의 `output_key` 결과가 session state를 통해 다음 에이전트에 전달됩니다.

---

### 2.3 `analyzer/tools.py` — Vector DB 도구

#### 스토리지 구조

```
datasets/vectordb/
├── products.usearch       # USearch 바이너리 벡터 인덱스
│                          # ndim=768, metric="cos"
└── products_meta.json     # 메타데이터
    {
      "next_key": 42,
      "products": {
        "0": { "product_name": ..., "brand": ..., "key_features": [...], ... },
        "1": { ... },
        ...
      }
    }
```

- **키(key)**: 자동 증가 정수 (`next_key`), USearch index와 meta JSON이 동일한 키로 연동
- **인덱스**: 메모리에 싱글턴으로 유지 (`_index`, `_meta` 전역 변수), 최초 조회 시 디스크에서 로드

---

#### `_get_embedding(texts)` — 임베딩 생성

```
Gemini API: gemini-embedding-001
출력 차원: 768
입력: 텍스트 리스트 (batch)
재시도: 최대 2회 (RETRY_DELAY=3초)
```

---

#### `search_local_db(key_features)` → str (JSON)

```
1. key_features 리스트를 공백으로 join → 쿼리 문자열
2. Gemini로 768차원 임베딩 생성
3. USearch index.search(query_vec, n=3) — Top-K 코사인 검색
4. score = 1.0 - cosine_distance  (코사인 유사도)
5. score < 0.3 인 결과 필터링
6. 매칭 있으면 {"found": true, "results": [...]} 반환
   매칭 없으면 {"found": false, "message": "..."} 반환
```

---

#### `save_to_local_db(product_name, brand, category, key_features, source, country, lang)` → str (JSON)

```
1. 중복 체크: 동일 product_name + brand 존재 시 저장 생략
2. key_features 임베딩 생성
3. index.add(next_key, vector)
4. meta["products"][key] = { 상품 정보 + uuid + created_at }
5. next_key 증가
6. index.save() + meta JSON 저장 (영속화)
```

---

#### 마이그레이션 (`_migrate_json_db`)

최초 실행 시 `datasets/products_db.json`이 존재하면 자동으로 Vector DB로 이전합니다. 이후 실행에서는 `products_meta.json`에 데이터가 있으면 건너뜁니다.

---

## 3. 데이터 흐름 (End-to-End)

```mermaid
sequenceDiagram
    participant User as 사용자 CLI
    participant Main as main.py
    participant ADK as ADK InMemoryRunner
    participant IA as image_analyzer
    participant RA as rag_agent
    participant Tools as tools.py
    participant VDB as USearch VectorDB
    participant GS as Google Search

    User->>Main: 이미지 파일 경로
    Main->>Main: 이미지 bytes 로딩 (types.Part)
    Main->>ADK: session 생성 (state: country, lang)
    Main->>ADK: run_async(text + image_part)
    ADK->>IA: 이미지 + 텍스트 입력
    IA->>ADK: JSON → session state["image_analysis"]
    ADK->>RA: {image_analysis} 전달

    alt confidence ≤ 0.7
        RA->>Tools: search_local_db(key_features)
        Tools->>VDB: 임베딩 생성 + 코사인 검색
        VDB-->>Tools: Top-K 결과
        Tools-->>RA: {found, results}

        alt found = false
            RA->>GS: Google Search (key_features)
            GS-->>RA: 검색 결과
            RA->>Tools: save_to_local_db(...)
            Tools->>VDB: 임베딩 생성 + 저장 (영속화)
        end
    end

    RA->>ADK: 최종 JSON (source, rag_confidence)
    ADK->>Main: 이벤트 스트림
    Main->>Main: JSON 파싱 + inference_time + token_usage
    Main->>User: outputs/result_YYYYMMDD_HHMMSS.json 저장
```

---

## 4. 출력 JSON 스키마

```json
{
  "product_name": "상품명",
  "product_name_confidence": 0.85,
  "category": "Food",
  "brand": "브랜드명",
  "brand_confidence": 0.90,
  "image_features": "시각적 특징 요약",
  "key_features": ["키워드1", "키워드2"],
  "expiration_date": "2025.12.31",
  "source": "local_db",
  "rag_confidence": {
    "probability": 0.72,
    "method": "local_db_score",
    "evidence": "key_features 일치도 기반"
  },
  "inference_time": "6.42s",
  "token_usage": {
    "input_tokens": 1200,
    "output_tokens": 350,
    "total_tokens": 1550
  }
}
```

| 필드 | source=image | source=local_db | source=google_search |
|------|:---:|:---:|:---:|
| `rag_confidence` | ✗ | ✓ | ✓ |
| `rag_confidence.method` | — | `local_db_score` | `google_search_estimate` |

---

## 5. 모듈 의존 관계

```mermaid
graph TD
    MAIN["main.py"]
    AGENT["analyzer.agent"]
    TOOLS["analyzer.tools"]
    ADK_AGENTS["google.adk.agents\nAgent, SequentialAgent"]
    ADK_TOOLSET["google.adk.tools\nGoogleSearchTool"]
    ADK_RUNNERS["google.adk.runners\nInMemoryRunner"]
    GENAI_TYPES["google.genai.types\nContent, Part, Blob"]
    GENAI["google.genai\nembed_content"]
    USEARCH["usearch\nIndex"]
    NUMPY["numpy"]
    PIL["PIL.Image\n해상도 확인"]

    MAIN --> AGENT
    MAIN --> ADK_RUNNERS
    MAIN --> GENAI_TYPES
    MAIN --> PIL
    AGENT --> ADK_AGENTS
    AGENT --> ADK_TOOLSET
    AGENT --> TOOLS
    TOOLS --> USEARCH
    TOOLS --> NUMPY
    TOOLS --> GENAI
```

---

## 6. 설정 및 환경 변수

| 항목 | 값 | 위치 |
|------|----|----|
| `GOOGLE_API_KEY` | Google AI Studio API 키 | `.env` |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | `tools.py` 상수 |
| `EMBEDDING_DIM` | `768` | `tools.py` 상수 |
| `MAX_RETRIES` | `2` (3회 시도) | `tools.py`, `main.py` |
| `RETRY_DELAY` | `3`초 | `tools.py`, `main.py` |
| Vector DB 경로 | `datasets/vectordb/` | `tools.py` 상수 |

---

## 7. 주요 설계 결정 사항

### 7.1 2-Agent 분리 (flash-lite + flash)
- `gemini-2.5-flash-lite`는 function calling을 지원하지 않지만 이미지 분석에는 도구가 불필요
- 비용이 높은 `gemini-2.5-flash`는 RAG/검색 단계에서만 사용하여 비용 절감

### 7.2 session state를 통한 에이전트 간 데이터 전달
- `image_analyzer`의 `output_key="image_analysis"` → session state에 저장
- `rag_agent` 프롬프트의 `{image_analysis}` 템플릿 변수로 참조

### 7.3 Vector DB (USearch + Gemini Embedding)
- 기존 키워드 집합 교집합 방식 → 의미 기반 검색(semantic search)으로 전환
- 코사인 유사도 기반이므로 "초콜릿" → "초코파이" 같은 의미 유사 매칭 가능
- score < 0.3 필터링으로 무관한 결과 차단

### 7.4 Google Search → 자동 DB 저장 (RAG 누적)
- Google Search로 찾은 상품 정보를 `save_to_local_db()`로 즉시 저장
- 이후 동일 상품 분석 시 Google Search 없이 local_db에서 처리 (비용 절감, 속도 향상)

### 7.5 배치 단위 예외 격리
- 이미지 하나 실패 시 `{"error": "analysis_failed"}` 기록 후 다음 이미지 계속 처리
- 전체 배치 실행이 단일 실패로 중단되지 않음
