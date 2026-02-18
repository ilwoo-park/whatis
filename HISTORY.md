# Project History

## 2026-02-17: Vector DB 전환 (JSON → USearch + Gemini Embedding)

### 배경
- 기존: `products_db.json`에 상품 정보를 저장하고, 단어 집합 교집합(set intersection)으로 검색
- 문제: 키워드 완전 일치만 가능하여 "초콜릿"→"초코파이" 매칭 불가, 다국어 검색 불가, 오타/변형 대응 불가
- 목표: 의미 기반 검색(semantic search)으로 검색 정확도 향상

### 변경 내용
- **파일:** `analyzer/tools.py` — 전체 재작성
  - JSON DB(`_load_db`/`_save_db`) → USearch 벡터 인덱스 + 별도 메타데이터 JSON 방식으로 전환
  - Gemini `gemini-embedding-001` (768차원) 임베딩으로 key_features를 벡터화
  - USearch `Index(metric="cos")` 사용, cosine similarity 기반 검색
  - 메타데이터는 `datasets/vectordb/products_meta.json`에 key-value 형태로 관리
  - 최초 실행 시 기존 `products_db.json`에서 자동 마이그레이션
  - `search_local_db` / `save_to_local_db` 함수 시그니처 및 반환 형식 동일 유지
- **파일:** `requirements.txt` — `chromadb` → `usearch`, `numpy` 추가
- **파일:** `AGENT.md`, `CLAUDE.md` — 기술 스택에 USearch, Embedding 정보 추가

### 데이터 구조
```
datasets/vectordb/
├── products.usearch       # USearch 벡터 인덱스 파일
└── products_meta.json     # 상품 메타데이터 (integer key → product info)
```

### 검증 방법
```bash
& "C:\Users\ilwoo\Envs\gemini\Scripts\python.exe" -m py_compile main.py analyzer\agent.py analyzer\tools.py
& "C:\Users\ilwoo\Envs\gemini\Scripts\python.exe" main.py datasets/images 3 --random
```
- 최초 실행 시 `[마이그레이션] JSON DB → Vector DB: N개 상품 이전 완료` 메시지 확인
- `source=local_db` 시 cosine similarity 기반 score 확인

## 2026-02-15: 2-Agent 구조로 리팩터링 (flash-lite + flash)

### 배경
- 기존: Single Agent(`gemini-2.5-flash`)가 이미지 분석 + RAG 도구 호출을 모두 담당
- 문제: `gemini-2.5-flash-lite`는 function calling 미지원이지만, 이미지 분석에는 도구가 불필요
- 목표: 이미지 분석은 flash-lite로, RAG는 flash로 분리하여 비용 절감

### 변경 내용
- **파일:** `analyzer/agent.py`
- **방식:** ADK `SequentialAgent`로 2개 서브 에이전트 순차 실행

| Agent | 모델 | 역할 | 도구 |
|-------|------|------|------|
| `image_analyzer` | gemini-2.5-flash-lite | 이미지 분석, `output_key="image_analysis"`로 state 저장 | 없음 |
| `rag_agent` | gemini-2.5-flash | state에서 `{image_analysis}` 읽고, brand/product_name 미식별 시 RAG 수행 후 최종 JSON 출력 | search_local_db, save_to_local_db, google_search_agent |

- `tools.py`, `main.py`는 변경 없음 (`root_agent` export명 유지)

### 검증 방법
```bash
python main.py datasets/images 3 --random
```
- brand 식별 가능 이미지 → source: `image` (flash-lite가 분석)
- brand 미식별 이미지 → source: `google_search` 또는 `local_db`
- inference_time 비교로 flash-lite 단독 분석 속도 확인

## 2026-02-16: 안정성 개선 + RAG 확률 정보 추가

### 배경
- 간헐적으로 ADK 템플릿 변수 치환 단계에서 `KeyError: Context variable not found: country`가 발생하여 배치 실행이 중단됨
- 배치 분석 중 단일 이미지 실패 시 전체 실행이 종료되는 문제가 있어 대량 처리 안정성 개선 필요
- RAG(`local_db`, `google_search`) 사용 시 결과 신뢰도를 함께 확인할 수 있도록 확률 정보 노출 요구

### 변경 내용
- **파일:** `analyzer/agent.py`
  - 프롬프트에서 취약한 템플릿 변수(`{country}`, `{lang}`) 의존 제거
  - `rag_agent` 출력 스키마에 `rag_confidence` 추가
    - `source=local_db`일 때: `search_local_db`의 최고 `score`를 `rag_confidence.probability`로 사용
    - `source=google_search`일 때: 근거 일치도 기반 `0~1` 추정값 사용
    - `source=image`일 때: `rag_confidence` 미포함
- **파일:** `main.py`
  - 사용자 메시지에 `country/lang` 힌트(`국가코드`, `출력언어`)를 명시하여 컨텍스트 전달 강화
  - 배치 루프에서 이미지 단위 예외 격리 추가(실패 항목만 `analysis_failed`로 기록하고 다음 항목 계속 진행)
  - 단일 파일 분석 실패 시에도 JSON 에러 형태로 반환
- **파일:** `README.md`
  - 동작 흐름에 RAG 확률 정보(`rag_confidence`) 포함 규칙 반영
  - 출력 예시 JSON 및 필드 설명에 `rag_confidence` 항목 추가

### 검증 방법
```bash
& "C:\Users\ilwoo\Envs\gemini\Scripts\python.exe" -m py_compile main.py analyzer\agent.py analyzer\tools.py
& "C:\Users\ilwoo\Envs\gemini\Scripts\python.exe" main.py datasets/images 5 --random
& "C:\Users\ilwoo\Envs\gemini\Scripts\python.exe" main.py datasets/images 10 --random
```

### 검증 결과
- 지정 가상환경(`C:\Users\ilwoo\Envs\gemini`)에서 문법 검증 통과
- 5개/10개 랜덤 샘플 실행 정상 종료
- 10개 실행에서 `source=local_db` 케이스 확인 및 `rag_confidence` 출력 확인
  - 예: `probability: 0.57`, `method: local_db_score`
