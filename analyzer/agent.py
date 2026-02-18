from google.adk.agents import Agent, SequentialAgent
from google.adk.tools.google_search_tool import GoogleSearchTool

from .tools import search_local_db, save_to_local_db

google_search_tool = GoogleSearchTool(bypass_multi_tools_limit=True)

image_analyzer = Agent(
    name="image_analyzer",
    model="gemini-2.5-flash-lite",
    description="상품 이미지를 분석하여 시각 정보를 추출하는 에이전트",
    output_key="image_analysis",
    instruction="""You are a product image analysis expert.
Analyze the product image precisely and return only one JSON object.

Context from user message:
- country code: e.g. 국가코드=KR
- output language: e.g. 출력언어=ko
Use country context to prioritize local language/brands (KR/JP/US/CN etc).

Output rules:
- No text outside JSON, no markdown.
- All string fields except `category` must use user's output language.
- `category` must always be in English.
- `image_features` must be a single string (never object/array).
- `key_features` must be an array of short strings only.

What to extract:
1) All readable package text (brand, product name, variant, certifications, weight/volume, dates)
2) Visual evidence (dominant colors, package type/material, graphics/mascots, layout)
3) Best determination of product_name and brand using text first, visual clues second

Critical dedup rules for `key_features`:
- Include only UNIQUE items.
- Do NOT repeat same text with minor spacing/case/punctuation differences.
- Merge near-duplicates into one normalized item.
- One concept per item; no long paragraphs.
- Keep concise: 8~20 items maximum.

Response JSON schema:
{
  "product_name": "Full product name including variant if visible, else ''",
  "product_name_confidence": 0.0,
  "category": "English category",
  "brand": "Brand name or ''",
  "brand_confidence": 0.0,
  "image_features": "Concise but detailed visual summary in one string",
  "key_features": ["Unique text/design facts"],
  "expiration_date": "YYYY.MM.DD or visible format, else ''"
}

Confidence guide:
- 1.0 clear readable evidence
- 0.7~0.9 mostly clear
- 0.4~0.6 inferred from visuals
- 0.1~0.3 weak guess
- 0.0 unknown

If not a product image:
{"error": "Unable to identify product image", "description": "reason"}
""",
)

rag_agent = Agent(
    name="rag_agent",
    model="gemini-2.5-flash",
    description="이미지 분석 결과를 보완하고 최종 JSON을 출력하는 에이전트",
    instruction="""You receive image analysis from previous step in session state: {image_analysis}

Use country/language from user message:
- country code (국가코드)
- output language (출력언어)
For search, prioritize target-country market sources and query language.

## Decision: skip RAG or use RAG?

SKIP RAG (source = "image") when ALL of these are true:
- product_name is not empty AND product_name_confidence > 0.7
- brand is not empty AND brand_confidence > 0.7
→ In this case, output immediately. Do NOT call any tool. Do NOT include rag_confidence field.

USE RAG when ANY of these is true:
- product_name is empty OR product_name_confidence <= 0.7
- brand is empty OR brand_confidence <= 0.7

## RAG flow (only when needed):
1) Call `search_local_db(key_features)`.
2) If best local score >= 0.5 and matches image evidence → use it:
   - source = "local_db"
   - rag_confidence = {probability: <score>, method: "local_db_score", evidence: "<short match summary>"}
3) Otherwise search web using strongest clues (readable text + category + visuals).
   - Try refined queries if first result is unclear.
   - Verify result matches image before accepting.
   - Save with `save_to_local_db` using correct `country`/`lang`.
   - source = "google_search"
   - rag_confidence = {probability: <0~1>, method: "google_search_estimate", evidence: "<match summary>"}

## Output rules:
- Return JSON only, no markdown.
- All string fields except `category` use user's output language.
- `category` must always be in English.
- `image_features` must be a single string (never object or array).
- `key_features` must be UNIQUE concise items only (no paragraphs), 8~20 max.
- CRITICAL: when source = "image", the `rag_confidence` field MUST NOT appear in the output at all.

Response schema:
{
  "product_name": "final name",
  "product_name_confidence": 0.0,
  "category": "English category",
  "brand": "final brand or ''",
  "brand_confidence": 0.0,
  "image_features": "single string summary",
  "key_features": ["unique facts"],
  "expiration_date": "date or ''",
  "source": "image | local_db | google_search"
}

When source = "local_db" or "google_search", append this field:
  "rag_confidence": {
    "probability": 0.0,
    "method": "local_db_score | google_search_estimate",
    "evidence": "short evidence"
  }
When source = "image", omit rag_confidence entirely — do not output the key.

Confidence rule:
- source=image: carry over confidence values from image_analysis unchanged.
- source=local_db/google_search with corrected brand/name: set confidence to rag_confidence.probability.

If input contains error, pass it through.
""",
    tools=[search_local_db, save_to_local_db, google_search_tool],
)

root_agent = SequentialAgent(
    name="product_analyzer",
    description="상품 이미지를 분석하여 상품 정보를 알려주는 에이전트",
    sub_agents=[image_analyzer, rag_agent],
)
