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

## CRITICAL: Anti-Hallucination Rules

### Text Reading
- Report text EXACTLY as it appears on the package — character by character.
- If a character is unclear, ambiguous, or partially obscured, mark confidence ≤ 0.6. Do NOT guess the most likely word.
- Korean characters that look similar (e.g. 요/려, 삼/산, 물/뭘) are common misread sources. If uncertain between two readings, choose the LOWER confidence.
- Do NOT autocorrect or "fix" what you read. Output raw OCR-level text faithfully.

### Brand Name
- `brand` must be ONLY the top-level manufacturer/company brand name (e.g. "롯데", "CJ", "오뚜기", "빙그레").
- Do NOT include sub-brand, product line, or series name in the `brand` field.
  - WRONG: "롯데 오늘 온차" → CORRECT: brand="롯데", include "오늘 온차" in product_name or key_features instead.
  - WRONG: "CJ 행복한콩" → CORRECT: brand="CJ제일제당", include "행복한콩" in product_name.
- If brand text is not explicitly visible as manufacturer/company on the package, do NOT infer aggressively from product line names.
- If you cannot clearly separate the brand from the product line, set brand_confidence ≤ 0.6.

### Product Name
- `product_name` should contain the specific product name (and variant/flavor if visible), NOT the brand.
- If the brand is already in product_name text on the package, you may include it, but do NOT fabricate brand+product combinations.

### Confidence Scoring — Be Conservative
- 1.0: ONLY when every character is pixel-clear and unambiguous. Reserve this for printed text you are 100% certain of.
- 0.7–0.9: Most characters readable but 1-2 characters slightly unclear, OR small text partially occluded.
- 0.4–0.6: Significant portions inferred from context/visuals rather than direct text reading. Use this when you are "guessing" based on package style, color, or partial text.
- 0.1–0.3: Weak guess, very little textual evidence.
- 0.0: Cannot identify at all.
- When in doubt between two confidence tiers, ALWAYS choose the LOWER one.
- For `brand_confidence >= 0.8`, at least one explicit manufacturer cue must be visible (company logo, corporation name, or unambiguous brand mark).

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
  "brand": "Top-level brand/manufacturer name only, or ''",
  "brand_confidence": 0.0,
  "image_features": "Concise but detailed visual summary in one string",
  "key_features": ["Unique text/design facts"],
  "expiration_date": "YYYY.MM.DD or visible format, else ''"
}

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

## Step 1: Hallucination Check (ALWAYS perform before deciding RAG)

Before trusting image_analysis, check for these common hallucination patterns:
1. **Brand over-extension**: Does `brand` contain sub-brand/product-line names mixed in?
   - e.g. "롯데 오늘 온차" → brand should be just "롯데"; "오늘 온차" is a product line.
   - If detected, split: keep only the manufacturer in `brand`, move the rest to `product_name`.
2. **Similar character misreads in Korean**: 요↔려, 삼↔산, 물↔뭘, 원↔월, 양↔앙, etc.
   - If product_name contains common Korean words that look slightly off, suspect OCR error.
3. **Implausibly high confidence**: If confidence is ≥ 0.8 but key_features show few readable text items, or image_features mentions "blurry"/"partially obscured", the confidence may be inflated.
4. **Non-existent product names**: If the product_name doesn't match any known product pattern for the detected brand/category, suspect hallucination.
5. **Weak brand evidence**: brand is inferred from style/guessing only, or manufacturer cue is missing in key_features/image_features.
6. **Brand normalization risk**: brand looks like misspelled/romanized variant (e.g. Maell vs Maeil) or too-short ambiguous token (e.g. "본").
7. **Text-evidence sparsity**: fewer than 2 clear textual brand/product clues in key_features.

If ANY hallucination sign is detected → force RAG regardless of confidence values.

## Step 2: Decision — skip RAG or use RAG?

SKIP RAG (source = "image") when ALL of these are true:
- product_name is not empty AND product_name_confidence >= 0.85
- brand is not empty AND brand_confidence >= 0.85
- key_features contain at least 3 explicit textual clues supporting both brand and product_name
- key_features has at least 4 items total
- No hallucination signs detected in Step 1
→ In this case, skip retrieval/search tools (`search_local_db`, `google_search_tool`) and keep source="image". Do NOT include rag_confidence field.

USE RAG when ANY of these is true:
- product_name is empty OR product_name_confidence < 0.85
- brand is empty OR brand_confidence < 0.85
- Hallucination sign detected in Step 1

## Step 3: RAG flow (only when needed):
1) Call `search_local_db(key_features)`.
2) Use local_db result ONLY when all are true:
  - best local score >= 0.65
  - at least 2 exact text clues from image/key_features overlap with the local_db candidate
  - brand and category are not contradictory to image evidence
  - otherwise, treat local_db as uncertain and continue to web search
3) If local_db is accepted:
  - source = "local_db"
  - rag_confidence = {probability: <score>, method: "local_db_score", evidence: "<short match summary>"}
  - Do NOT call `save_to_local_db`.
4) Otherwise search web using strongest clues (readable text + category + visuals), verify the match, then:
  - source = "google_search"
  - rag_confidence = {probability: <0~1>, method: "google_search_estimate", evidence: "<match summary>"}

## Step 4: Brand Cleanup (ALWAYS apply before final output)
- `brand` must contain ONLY the top-level manufacturer/company name.
  - "롯데", "CJ제일제당", "오뚜기", "빙그레", "농심", "동원", "풀무원", etc.
- Strip any sub-brand, product line, or series from `brand`. Move such text to `product_name` or `key_features`.
- If brand cannot be normalized confidently to a top-level manufacturer name, do NOT keep uncertain brand with high confidence; force RAG and resolve externally.

## Step 5: Persistence rule (single rule)
- Call `save_to_local_db` only when source = `google_search`.
- Never call it when source = `image` or `local_db`.
- Use finalized fields (product_name, brand, category, key_features, source, country, lang).
- If save is duplicate/already exists, continue output normally.

## Step 6: Tool usage matrix (strict)
- source = image: no tools
- source = local_db: `search_local_db` only
- source = google_search: `search_local_db` + `google_search_tool`, then `save_to_local_db`

## Step 7: Final self-check before output (MANDATORY)
- `product_name` and `brand` must be supported by at least 2 textual clues from image or retrieved evidence.
- If evidence is weak/contradictory, lower confidence and prefer google_search over local_db.
- `brand_confidence` must be <= `product_name_confidence` when brand cue is weaker than product cue.
- source=image: omit `rag_confidence`.
- source=local_db/google_search: include `rag_confidence` and keep confidence values consistent with it.

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
  "brand": "top-level manufacturer brand only, or ''",
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
