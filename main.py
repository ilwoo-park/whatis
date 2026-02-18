import asyncio
import json
import logging
import random
import re
import sys
import time
import mimetypes
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from google.genai import types
from PIL import Image

from analyzer.agent import root_agent

load_dotenv(".env")

sys.stdout.reconfigure(encoding="utf-8")
logging.getLogger("google.genai.models").setLevel(logging.ERROR)


def load_image_as_part(image_path: str) -> types.Part:
    """이미지 파일을 읽어 types.Part 객체로 변환합니다."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None or not mime_type.startswith("image/"):
        raise ValueError(f"지원하지 않는 파일 형식입니다: {image_path}")

    image_bytes = path.read_bytes()
    return types.Part(
        inline_data=types.Blob(mime_type=mime_type, data=image_bytes)
    )


def get_image_resolution(image_path: str) -> str:
    """이미지 해상도를 'WxH' 문자열로 반환합니다."""
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            return f"{w}x{h}"
    except Exception:
        return "unknown"


async def analyze_image(image_path: str, country: str = "KR", lang: str = "ko") -> tuple[str, dict]:
    """상품 이미지를 분석하여 결과 텍스트와 토큰 사용량을 반환합니다."""
    runner = InMemoryRunner(agent=root_agent, app_name="whatis")
    session = await runner.session_service.create_session(
        app_name="whatis", user_id="user",
        state={"country": country, "lang": lang},
    )

    image_part = load_image_as_part(image_path)
    content = types.Content(
        role="user",
        parts=[
            types.Part(text=f"이 상품 이미지를 분석해주세요. 국가코드={country}, 출력언어={lang}"),
            image_part,
        ],
    )

    result_parts = []
    all_text_parts = []
    current_agent = None
    token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    async for event in runner.run_async(
        user_id="user", session_id=session.id, new_message=content
    ):
        author = getattr(event, "author", None)
        if author and author != current_agent:
            current_agent = author
            result_parts = []
            print(f"  >> [{current_agent}] 호출됨", flush=True)
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.function_call:
                    print(f"     -> tool: {part.function_call.name}()", flush=True)
                    continue
                if part.text:
                    result_parts.append(part.text)
                    all_text_parts.append(part.text)
        # 토큰 사용량 누적
        um = getattr(event, "usage_metadata", None)
        if um:
            token_usage["input_tokens"] += um.prompt_token_count or 0
            token_usage["output_tokens"] += um.candidates_token_count or 0
            token_usage["total_tokens"] += um.total_token_count or 0

    final_text = "\n".join(result_parts).strip()
    if final_text:
        return final_text, token_usage

    fallback_text = "\n".join(all_text_parts).strip()
    if fallback_text:
        return fallback_text, token_usage

    raise ValueError("모델이 텍스트 응답을 반환하지 않았습니다.")


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}


def strip_code_block(text: str) -> str:
    """마크다운 코드 블록(```json ... ```)을 제거합니다."""
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


MAX_RETRIES = 2
RETRY_DELAY = 3  # seconds

ALLOWED_SOURCES = {"image", "local_db", "google_search"}
REQUIRED_RESULT_KEYS = {
    "product_name",
    "product_name_confidence",
    "category",
    "brand",
    "brand_confidence",
    "image_features",
    "key_features",
    "expiration_date",
    "source",
}


def validate_result_payload(payload: dict) -> None:
    """모델 결과 JSON 최소 스키마를 검증합니다."""
    if not isinstance(payload, dict):
        raise ValueError("모델 응답 JSON이 객체 형식이 아닙니다.")

    if "error" in payload:
        return

    missing = [key for key in REQUIRED_RESULT_KEYS if key not in payload]
    if missing:
        raise ValueError(f"필수 필드 누락: {', '.join(missing)}")

    source = payload.get("source")
    if source not in ALLOWED_SOURCES:
        raise ValueError(f"유효하지 않은 source 값: {source}")

    key_features = payload.get("key_features")
    if not isinstance(key_features, list) or len(key_features) == 0:
        raise ValueError("key_features는 비어있지 않은 배열이어야 합니다.")

    if not all(isinstance(item, str) for item in key_features):
        raise ValueError("key_features의 모든 항목은 문자열이어야 합니다.")

    for field in ("product_name", "category", "brand", "image_features"):
        if not isinstance(payload.get(field), str):
            raise ValueError(f"{field} 필드는 문자열이어야 합니다.")

    for confidence_field in ("product_name_confidence", "brand_confidence"):
        value = payload.get(confidence_field)
        if not isinstance(value, (int, float)):
            raise ValueError(f"{confidence_field} 필드는 숫자여야 합니다.")
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{confidence_field} 값은 0.0~1.0 범위여야 합니다.")

    expiration_date = payload.get("expiration_date")
    if expiration_date is None:
        payload["expiration_date"] = ""
    elif not isinstance(expiration_date, str):
        raise ValueError("expiration_date 필드는 문자열 또는 빈 문자열이어야 합니다.")


async def analyze_single(image_path: str, country: str = "KR", lang: str = "ko"):
    """단일 이미지를 분석하고 결과를 반환합니다. 실패 시 최대 2회 재시도합니다."""
    start = time.time()
    last_error = None
    for attempt in range(1, MAX_RETRIES + 2):  # 1 + 2 retries = 3 attempts
        try:
            result, token_usage = await analyze_image(image_path, country, lang)
            if not result.strip():
                raise ValueError("모델 응답이 비어 있습니다.")

            elapsed = round(time.time() - start, 2)
            cleaned = strip_code_block(result)
            if not cleaned:
                raise ValueError("응답 정제 후 비어 있습니다.")

            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON 파싱 실패: {e}") from e

            validate_result_payload(parsed)
            parsed["inference_time"] = f"{elapsed}s"
            parsed["token_usage"] = token_usage
            return parsed
        except Exception as e:
            last_error = e
            if attempt <= MAX_RETRIES:
                print(f"  !! API 오류 (시도 {attempt}/{MAX_RETRIES + 1}): {e}", flush=True)
                print(f"  !! {RETRY_DELAY}초 후 재시도...", flush=True)
                await asyncio.sleep(RETRY_DELAY)
            else:
                raise last_error


async def main():
    argv = sys.argv[1:]
    country = "KR"
    lang = "ko"
    use_random = False
    positional = []

    i = 0
    while i < len(argv):
        if argv[i] == "--country" and i + 1 < len(argv):
            country = argv[i + 1]
            i += 2
        elif argv[i] == "--lang" and i + 1 < len(argv):
            lang = argv[i + 1]
            i += 2
        elif argv[i] == "--random":
            use_random = True
            i += 1
        else:
            positional.append(argv[i])
            i += 1

    if len(positional) < 1:
        print("사용법: python main.py <이미지_파일_경로 또는 디렉토리> [샘플 수] [옵션]")
        print()
        print("옵션:")
        print("  --country CODE  국가 코드 (기본: KR)")
        print("  --lang CODE     언어 코드 (기본: ko)")
        print("  --random        랜덤 샘플 선택")
        print()
        print("예시: python main.py product.jpg")
        print("예시: python main.py datasets/images 5 --random")
        print("예시: python main.py datasets/images 3 --country US --lang en")
        sys.exit(1)

    target = Path(positional[0])
    sample_count = int(positional[1]) if len(positional) >= 2 else None
    print(f"설정: country={country}, lang={lang}\n")

    if target.is_dir():
        images = sorted(
            f for f in target.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            print(f"디렉토리에 이미지 파일이 없습니다: {target}")
            sys.exit(1)

        if sample_count is not None:
            if use_random:
                images = random.sample(images, min(sample_count, len(images)))
            else:
                images = images[:sample_count]

        total = len(images)
        mode = "랜덤 샘플" if use_random and sample_count else "샘플" if sample_count else ""
        label = f"총 {total}개 이미지 분석" + (f" ({mode})" if mode else "")
        print(f"{label}\n")
        results = []
        for idx, img in enumerate(images, 1):
            resolution = get_image_resolution(str(img))
            print(f"[{idx}/{total}] 분석 중: {img.name} ({resolution}) ...", flush=True)
            try:
                parsed = await analyze_single(str(img), country, lang)
            except Exception as e:
                parsed = {
                    "error": "analysis_failed",
                    "message": str(e),
                    "inference_time": "",
                }
                print(f"[{idx}/{total}] 실패: {img.name} ({e})\n", flush=True)
            results.append({"file": img.name, "result": parsed})
            if "error" not in parsed:
                tu = parsed.get("token_usage", {})
                print(
                    f"[{idx}/{total}] 완료: {img.name} ({parsed.get('inference_time', '')})"
                    f" | tokens: in={tu.get('input_tokens', 0)} out={tu.get('output_tokens', 0)} total={tu.get('total_tokens', 0)}\n",
                    flush=True,
                )

        output = results
    else:
        try:
            output = await analyze_single(str(target), country, lang)
        except Exception as e:
            output = {
                "error": "analysis_failed",
                "message": str(e),
            }

    output_json = json.dumps(output, ensure_ascii=False, indent=2)
    print(output_json)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"result_{timestamp}.json"
    output_path.write_text(output_json, encoding="utf-8")
    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
