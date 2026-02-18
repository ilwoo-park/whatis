import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types
from usearch.index import Index

VECTORDB_DIR = Path(__file__).resolve().parent.parent / "datasets" / "vectordb"
INDEX_PATH = VECTORDB_DIR / "products.usearch"
META_PATH = VECTORDB_DIR / "products_meta.json"
JSON_DB_PATH = Path(__file__).resolve().parent.parent / "datasets" / "products_db.json"

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768

MAX_RETRIES = 2
RETRY_DELAY = 3  # seconds


def _get_embedding(texts: list[str]) -> list[list[float]]:
    """Gemini embedding API를 호출하여 텍스트 임베딩을 반환합니다. 실패 시 최대 2회 재시도합니다."""
    import time as _time

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    last_error = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
            )
            return [e.values for e in result.embeddings]
        except Exception as e:
            last_error = e
            if attempt <= MAX_RETRIES:
                print(f"  !! Embedding API 오류 (시도 {attempt}/{MAX_RETRIES + 1}): {e}", flush=True)
                _time.sleep(RETRY_DELAY)
            else:
                raise last_error


# --- 메타데이터 관리 ---

def _load_meta() -> dict:
    """메타데이터 JSON을 로드합니다."""
    if not META_PATH.exists():
        return {"products": {}, "next_key": 0}
    text = META_PATH.read_text(encoding="utf-8")
    return json.loads(text) if text.strip() else {"products": {}, "next_key": 0}


def _save_meta(meta: dict) -> None:
    """메타데이터 JSON을 저장합니다."""
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# --- USearch 인덱스 관리 (싱글턴) ---

_index: Index | None = None
_meta: dict | None = None


def _get_index() -> tuple[Index, dict]:
    """USearch 인덱스와 메타데이터를 반환합니다 (싱글턴)."""
    global _index, _meta
    if _index is not None and _meta is not None:
        return _index, _meta

    VECTORDB_DIR.mkdir(parents=True, exist_ok=True)

    _index = Index(ndim=EMBEDDING_DIM, metric="cos")
    _meta = _load_meta()

    if INDEX_PATH.exists() and _meta["products"]:
        _index.load(str(INDEX_PATH))
    elif not _meta["products"]:
        # 인덱스가 비어 있으면 기존 JSON DB에서 마이그레이션
        _migrate_json_db()

    return _index, _meta


def _persist() -> None:
    """인덱스와 메타데이터를 디스크에 저장합니다."""
    if _index is not None:
        _index.save(str(INDEX_PATH))
    if _meta is not None:
        _save_meta(_meta)


def _migrate_json_db() -> None:
    """기존 JSON DB 데이터를 USearch Vector DB로 마이그레이션합니다."""
    if not JSON_DB_PATH.exists():
        return
    text = JSON_DB_PATH.read_text(encoding="utf-8")
    if not text.strip():
        return

    db = json.loads(text)
    products = db.get("products", [])
    if not products:
        return

    texts = []
    entries = []

    for product in products:
        key_features = product.get("key_features", [])
        doc_text = " ".join(key_features)
        if not doc_text.strip():
            continue

        texts.append(doc_text)
        entries.append({
            "id": product.get("id", str(uuid.uuid4())),
            "product_name": product.get("product_name", ""),
            "brand": product.get("brand", ""),
            "category": product.get("category", ""),
            "key_features": key_features,
            "source": product.get("source", ""),
            "country": product.get("country", "KR"),
            "lang": product.get("lang", "ko"),
            "created_at": product.get("created_at", ""),
        })

    if not texts:
        return

    embeddings = _get_embedding(texts)
    vectors = np.array(embeddings, dtype=np.float32)

    for i, entry in enumerate(entries):
        key = _meta["next_key"]
        _meta["next_key"] = key + 1
        _index.add(key, vectors[i])
        _meta["products"][str(key)] = entry

    _persist()
    print(f"  [마이그레이션] JSON DB → Vector DB: {len(entries)}개 상품 이전 완료")


def search_local_db(key_features: list[str]) -> str:
    """key_features를 사용하여 로컬 Vector DB에서 유사 상품을 검색합니다.

    Args:
        key_features: 검색할 키워드 리스트

    Returns:
        매칭된 상품 정보 JSON 문자열 또는 결과 없음 메시지
    """
    index, meta = _get_index()

    if len(index) == 0 or not key_features:
        return json.dumps({"found": False, "message": "로컬 DB에 상품이 없습니다."}, ensure_ascii=False)

    query_text = " ".join(key_features)
    query_vec = np.array(_get_embedding([query_text])[0], dtype=np.float32)

    n_results = min(3, len(index))
    results = index.search(query_vec, n_results)

    matched = []
    for i in range(len(results.keys)):
        key = str(int(results.keys[i]))
        distance = float(results.distances[i])
        # USearch cosine metric: distance = 1 - similarity
        score = round(1.0 - distance, 2)

        if score < 0.3:
            continue

        entry = meta["products"].get(key)
        if not entry:
            continue

        matched.append({
            "product_name": entry["product_name"],
            "brand": entry["brand"],
            "category": entry["category"],
            "key_features": entry["key_features"],
            "score": score,
        })

    if not matched:
        return json.dumps({"found": False, "message": "매칭되는 상품을 찾지 못했습니다."}, ensure_ascii=False)

    return json.dumps({"found": True, "results": matched}, ensure_ascii=False)


def save_to_local_db(
    product_name: str,
    brand: str,
    category: str,
    key_features: list[str],
    source: str,
    country: str = "KR",
    lang: str = "ko",
) -> str:
    """검색 결과를 로컬 Vector DB에 저장합니다.

    Args:
        product_name: 상품명
        brand: 브랜드명
        category: 카테고리
        key_features: 주요 특징 리스트
        source: 데이터 출처 (google_search 등)
        country: 국가 코드 (e.g. KR, US, JP)
        lang: 언어 코드 (e.g. ko, en, ja)

    Returns:
        저장 결과 메시지
    """
    index, meta = _get_index()

    # 중복 체크: 동일 상품명+브랜드가 있는지 확인
    for entry in meta["products"].values():
        if entry["product_name"] == product_name and entry["brand"] == brand:
            return json.dumps(
                {"saved": False, "message": "이미 동일한 상품이 DB에 존재합니다."},
                ensure_ascii=False,
            )

    doc_text = " ".join(key_features)
    vec = np.array(_get_embedding([doc_text])[0], dtype=np.float32)

    key = meta["next_key"]
    meta["next_key"] = key + 1
    index.add(key, vec)

    meta["products"][str(key)] = {
        "id": str(uuid.uuid4()),
        "product_name": product_name,
        "brand": brand,
        "category": category,
        "key_features": key_features,
        "source": source,
        "country": country,
        "lang": lang,
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }

    _persist()

    return json.dumps(
        {"saved": True, "message": f"'{product_name}' 상품이 DB에 저장되었습니다."},
        ensure_ascii=False,
    )
