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
When the user sends a product image, analyze it and extract all visible information.

Respond ONLY with a JSON object. Do NOT include any text outside the JSON. Do NOT use markdown code blocks (```).

All string values in the JSON must be written in the user's requested language from the conversation context.

Response JSON schema:
{
  "product_name": "Product name or type identified from the image. Use empty string '' if not identifiable",
  "product_name_confidence": "number between 0.0 and 1.0 indicating how confident you are about the product_name. 1.0 = text clearly visible, 0.0 = pure guess",
  "category": "Product category in English (e.g. Electronics, Clothing, Food, Beverage, Furniture, etc.)",
  "brand": "Brand name if identifiable. Use empty string '' if not identifiable",
  "brand_confidence": "number between 0.0 and 1.0 indicating how confident you are about the brand. 1.0 = logo/text clearly visible, 0.0 = pure guess. Use 0.0 when brand is empty",
  "image_features": "Summary of visual characteristics (shape, color, package design, etc.)",
  "key_features": ["List of notable features including color, material, design, AND all recognized text/labels visible in the image"],
  "expiration_date": "Expiration date or best-before date if visible in the image (e.g. '2025.12.31'). Use empty string '' if not visible"
}

Confidence scoring guidelines:
- 1.0: Text/logo is clearly readable in the image
- 0.7-0.9: Partially visible or slightly blurred but highly likely
- 0.4-0.6: Inferred from visual context (packaging style, shape, color) without clear text
- 0.1-0.3: Low confidence guess based on general appearance
- 0.0: Not identifiable or empty string value

If no product is found or the image is not a product:
{"error": "Unable to identify product image", "description": "reason"}
""",
)

rag_agent = Agent(
    name="rag_agent",
    model="gemini-2.5-flash",
    description="이미지 분석 결과를 보완하고 최종 JSON을 출력하는 에이전트",
    instruction="""You receive a product image analysis result from a previous step.
The analysis is available in the session state as: {image_analysis}

## Your task
1. Read the image analysis result above.
2. Determine if RAG search is needed. RAG is needed when ANY of these conditions is true:
   - `product_name` is empty or not identified
   - `brand` is empty or not identified
   - `product_name_confidence` <= 0.7
   - `brand_confidence` <= 0.7
3. If RAG search IS needed:
   a. Call `search_local_db` with the `key_features` from the analysis.
   b. If local DB returns results (`found: true`), use the best matching result to fill in or correct `brand` and/or `product_name`. Set `source` to `local_db`.
    - Also set `rag_confidence.probability` using the best result's `score` value directly.
    - Set `rag_confidence.method` to `local_db_score`.
    - Set `rag_confidence.evidence` to a short explanation of matched key features.
   c. If local DB has no results (`found: false`), use `google_search` to search for the product using key features as the query.
      - After finding information via Google Search, call `save_to_local_db` to save the product info for future use.
    - When calling `save_to_local_db`, pass concrete `country` and `lang` values inferred from the user request. If not provided, use `country="KR"` and `lang="ko"`.
      - Set `source` to `google_search`.
    - Set `rag_confidence.probability` as your confidence estimate between 0 and 1 based on evidence consistency (brand match, product name match, packaging/text consistency).
    - Set `rag_confidence.method` to `google_search_estimate`.
    - Set `rag_confidence.evidence` to a short evidence summary.
4. If RAG search is NOT needed (both product_name and brand are identified with confidence > 0.7), set `source` to `image`.
  - In this case, do not include `rag_confidence`.

## Response format
Respond ONLY with a JSON object. Do NOT include any text outside the JSON. Do NOT use markdown code blocks (```).

All string values in the JSON must be written in the user's requested language from the conversation context.

Response JSON schema:
{
  "product_name": "Product name or type identified",
  "product_name_confidence": "number between 0.0 and 1.0 (carry over from image_analysis, or update if RAG improved it)",
  "category": "Product category in English (e.g. Electronics, Clothing, Food, Beverage, Furniture, etc.)",
  "brand": "Brand name if identifiable. Use empty string '' if not identifiable",
  "brand_confidence": "number between 0.0 and 1.0 (carry over from image_analysis, or update if RAG improved it)",
  "image_features": "Summary of visual characteristics (shape, color, package design, etc.)",
  "key_features": ["List of notable features"],
  "expiration_date": "Expiration date if visible (e.g. '2025.12.31'). Use empty string '' if not visible",
  "source": "image | local_db | google_search",
  "rag_confidence": {
    "probability": "number between 0 and 1 (include only when source is local_db or google_search)",
    "method": "local_db_score | google_search_estimate",
    "evidence": "short reason for the probability"
  }
}

Confidence update rules:
- If source is "image": keep the original confidence values from image_analysis as-is.
- If source is "local_db" or "google_search" and you updated product_name or brand: set the confidence to rag_confidence.probability.

If the analysis contains an error, pass it through as-is.
""",
    tools=[search_local_db, save_to_local_db, google_search_tool],
)

root_agent = SequentialAgent(
    name="product_analyzer",
    description="상품 이미지를 분석하여 상품 정보를 알려주는 에이전트",
    sub_agents=[image_analyzer, rag_agent],
)
