"""
EventFlowStitcher

청크 모드(RecursiveRequirementsValidationGenerator) 로 추출된 이벤트들의 nextEvents 와 level 을
글로벌 컨텍스트로 보강하는 후처리 워크플로우.

문제 배경:
- RequirementsValidator 는 청크 단위로 호출됨. 각 청크 LLM 은 자기 청크 텍스트만 봄.
- 청크 경계에서 잘린 시나리오는 LLM 이 "다음 이벤트가 뭔지 확신 못해서" nextEvents 를 빈 배열로 둠.
- 청크 0 의 출력에 빈 nextEvents 가 깔리면, 청크 1+ 가 previousChunkSummary 로 그걸 보고 "이 프로젝트는 nextEvents 안 채우는 컨벤션" 으로 cascading 패턴 학습.
- 결과: 누적 events 의 nextEvents 가 모두 비고, level 도 대부분 1 → 프론트 BPMN 시각화에서 모든 이벤트가 "Single Events" 탭으로 떨어져 흐름 없이 그려짐.

해결:
- 모든 청크 처리가 끝난 뒤 한 번 stitcher 호출.
- 누적 events + actors + 원본 요구사항을 한 번에 LLM 에 주고 nextEvents/level 만 채우게 함.
- 이벤트 추가/삭제/이름 변경 금지 (시각 모델 일관성 보존).
"""
from typing import Any, Dict, List
from datetime import datetime
import json
import re

from project_generator.utils.logging_util import LoggingUtil
from project_generator.utils.llm_factory import create_chat_llm


class EventFlowStitcher:
    """청크 단위로 추출된 이벤트 흐름(nextEvents)을 글로벌 컨텍스트로 보강."""

    # 매우 큰 요구사항 텍스트가 들어와도 프롬프트 토큰 폭주 방지.
    # 흐름 추론은 전체 텍스트 정독 없이도 이벤트 목록 + 설명으로 상당 부분 가능.
    MAX_REQUIREMENTS_CHARS = 60000

    def __init__(self):
        self.llm = create_chat_llm(
            temperature=0.2,
            streaming=False,
            timeout=180,
            model_kwargs={
                "response_format": {"type": "json_object"}
            }
        )

    def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            LoggingUtil.info("EventFlowStitcher", "Event flow stitching started")

            events: List[Dict] = input_data.get("events", []) or []
            actors: List[Dict] = input_data.get("actors", []) or []
            requirements_text: str = input_data.get("requirements", "") or ""

            if not events:
                LoggingUtil.info("EventFlowStitcher", "No events to stitch")
                return {
                    "type": "FLOW_STITCH_RESULT",
                    "content": {"events": events},
                    "progress": 100
                }

            # 원본 요구사항이 너무 길면 잘라서 토큰 절약. 흐름 추론엔 이벤트 description 이 더 중요함.
            if requirements_text and len(requirements_text) > self.MAX_REQUIREMENTS_CHARS:
                LoggingUtil.info(
                    "EventFlowStitcher",
                    f"Truncating requirements from {len(requirements_text)} to {self.MAX_REQUIREMENTS_CHARS} chars"
                )
                requirements_text = requirements_text[: self.MAX_REQUIREMENTS_CHARS] + "\n... [TRUNCATED]"

            prompt = self._build_prompt(events, actors, requirements_text)
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, "content") else str(response)

            stitched = self._parse_and_merge(response_text, events)

            LoggingUtil.info(
                "EventFlowStitcher",
                f"Stitch complete: {self._count_with_next(stitched)}/{len(stitched)} events have nextEvents"
            )

            return {
                "type": "FLOW_STITCH_RESULT",
                "content": {"events": stitched},
                "progress": 100
            }

        except Exception as e:
            LoggingUtil.error("EventFlowStitcher", f"Failed: {str(e)}")
            # 실패해도 원본 events 그대로 돌려줘서 fallback 동작 (그림은 안 그려져도 데이터 보존).
            return {
                "type": "FLOW_STITCH_RESULT",
                "content": {"events": input_data.get("events", [])},
                "progress": 100,
                "error": str(e)
            }

    @staticmethod
    def _count_with_next(events: List[Dict]) -> int:
        return sum(1 for e in events if e.get("nextEvents"))

    def _build_prompt(self, events: List[Dict], actors: List[Dict], requirements_text: str) -> List[Dict]:
        # 입력 이벤트 요약 (LLM 에 보낼 핵심 필드만)
        events_for_llm = [
            {
                "name": e.get("name"),
                "actor": e.get("actor"),
                "displayName": e.get("displayName"),
                "description": e.get("description"),
                "currentLevel": e.get("level", 1),
                "currentNextEvents": e.get("nextEvents", []) or []
            }
            for e in events
            if e.get("name")
        ]

        actors_for_llm = [
            {
                "name": a.get("name"),
                "events": a.get("events", []) or []
            }
            for a in actors
            if a.get("name")
        ]

        system_prompt = """You are an Expert Business Analyst & Domain-Driven Design Specialist.

You are given a list of business events that were extracted from a requirements document in chunks. Because each chunk was processed in isolation, the inter-event flow information (nextEvents) and the sequence levels were not populated reliably. Your job is to fix that.

**Your Task**
For each event, populate:
- `nextEvents`: array of event names that LOGICALLY follow this event in the business process.
- `level`: integer >= 1, indicating sequence priority in its containing process. Earlier events get lower level numbers; events at the same level are concurrent or independent.

**Hard Rules**
- You MUST NOT add new events.
- You MUST NOT remove any events.
- You MUST NOT change event names.
- You MUST NOT modify any field other than `nextEvents` and `level`.
- Every `nextEvents` entry MUST exactly match the `name` of one of the provided events.
- An event may have zero or more nextEvents. Zero means it's a terminal/standalone event.

**How to Infer Flow**
1. Read each event's description and actor. The narrative in the original requirements describes business processes — find sequential cause-and-effect relations.
2. Events handled by the same actor are often sequential when they refer to consecutive steps of a workflow.
3. Cross-actor handoffs (e.g., Customer places order → System validates → Warehouse ships) are common and SHOULD be captured as nextEvents.
4. Branches: if an event leads to multiple possible next events (parallel or alternative), list all of them in nextEvents.
5. Terminal events: events that complete a workflow (e.g., OrderDelivered, PaymentSettled) usually have empty nextEvents.
6. Assign levels so that for each chain A -> B -> C, level(A) < level(B) < level(C). Concurrent events get the same level.

**Output Format (JSON only, no markdown, no extra text)**
{
    "events": [
        {"name": "EventA", "level": 1, "nextEvents": ["EventB", "EventC"]},
        {"name": "EventB", "level": 2, "nextEvents": ["EventD"]},
        ...
    ]
}

Return ALL input events in the output. Order does not matter, but every input event's name MUST appear in the output exactly once."""

        user_prompt = f"""Input events ({len(events_for_llm)} total):
{json.dumps(events_for_llm, ensure_ascii=False, indent=2)}

Input actors:
{json.dumps(actors_for_llm, ensure_ascii=False, indent=2)}

Original requirements (for full business context — may be truncated):
{requirements_text or '(not provided)'}

Now produce the JSON output with nextEvents and level populated for ALL events."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def _parse_and_merge(self, response_text: str, original_events: List[Dict]) -> List[Dict]:
        """LLM 응답을 파싱해서 원본 events 에 nextEvents/level 만 머지."""
        cleaned = self._extract_json(response_text)
        data = json.loads(cleaned)
        stitched = data.get("events", [])

        # 인덱싱: 이벤트 이름 → 업데이트 정보
        valid_names = {e.get("name") for e in original_events if e.get("name")}
        stitch_index: Dict[str, Dict] = {}
        for item in stitched:
            name = item.get("name")
            if not name:
                continue
            # LLM 이 nextEvents 에 입력에 없는 이름을 적었으면 필터링 (할루시네이션 방지)
            next_events = [n for n in (item.get("nextEvents") or []) if n in valid_names and n != name]
            level = item.get("level")
            try:
                level = int(level) if level is not None else None
            except (ValueError, TypeError):
                level = None
            stitch_index[name] = {"nextEvents": next_events, "level": level}

        # 머지: 원본 이벤트 객체에 nextEvents/level 만 덮어쓰기. 다른 필드는 절대 안 건드림.
        merged: List[Dict] = []
        for e in original_events:
            name = e.get("name")
            new_e = dict(e)  # 새 dict (참조 공유 방지)
            if name and name in stitch_index:
                upd = stitch_index[name]
                new_e["nextEvents"] = upd["nextEvents"]
                if upd["level"] is not None:
                    new_e["level"] = upd["level"]
            else:
                # LLM 응답에 빠진 이벤트는 원본 그대로 유지
                new_e["nextEvents"] = e.get("nextEvents", []) or []
            merged.append(new_e)

        return merged

    @staticmethod
    def _extract_json(text: str) -> str:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
