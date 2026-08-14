import json
import time

from .logging_util import LoggingUtil


class LlmJsonUtil:
    """LLM 응답을 JSON 으로 파싱할 때 쓰는 공용 헬퍼.

    배경: 요약(summarizer) 워크플로에서 LLM 이 JSON 을 문자열 중간에서 끊어 반환해
    `json.loads` 가 "Unterminated string starting at: line 4 column 21 (char 64)" 로
    실패했고, 재시도가 없어 청크 1개 실패가 전체 요약 중단으로 이어졌다.

    구조화 출력(with_structured_output)이나 response_format=json_object 를 쓰는
    워크플로는 API 가 형식을 보장하지만, 자유형 응답을 직접 파싱하는 워크플로는
    이 헬퍼로 (1) 코드펜스 제거 (2) 파싱 실패 시 재시도 를 함께 처리한다.
    """

    @staticmethod
    def extract_json_text(text: str) -> str:
        """LLM 응답에서 JSON 본문만 추출. 코드펜스(```json ... ```)를 벗겨낸다."""
        if not text:
            return ""

        cleaned = text.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1]
            cleaned = cleaned.split("```", 1)[0]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
            if "```" in cleaned:
                cleaned = cleaned.split("```", 1)[0]

        return cleaned.strip()

    @staticmethod
    def invoke_and_parse(llm, messages, label: str, attempts: int = 3, retry_delay: float = 1.0) -> dict:
        """LLM 호출 → JSON 파싱. 파싱 실패 시 재호출한다.

        JSON 강제 모드를 켜도 출력 토큰 상한에 걸리면 잘릴 수 있어 재시도는 여전히 필요하다.
        마지막 시도까지 실패하면 원래 예외를 그대로 올려 호출자(main.process_*_job)가
        isFailed=True 를 기록하게 한다.
        """
        last_error = None

        for attempt in range(1, attempts + 1):
            response = llm.invoke(messages).content
            cleaned = LlmJsonUtil.extract_json_text(response)

            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                last_error = e
                preview = (cleaned[:200] + "...") if len(cleaned) > 200 else cleaned
                LoggingUtil.warning(
                    label,
                    f"JSON 파싱 실패 ({attempt}/{attempts}): {e} | 응답 길이={len(cleaned)} | 앞부분={preview}"
                )
                if attempt < attempts:
                    time.sleep(retry_delay)

        raise last_error
