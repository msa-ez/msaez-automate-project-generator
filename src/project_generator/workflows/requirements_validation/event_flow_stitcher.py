"""
EventFlowStitcher

청크 모드(RecursiveRequirementsValidationGenerator) 로 추출된 이벤트들의 nextEvents 와 level 을
글로벌 컨텍스트로 보강하는 후처리 워크플로우.

문제 배경:
- RequirementsValidator 는 청크 단위로 호출됨. 각 청크 LLM 은 자기 청크 텍스트만 봄.
- 청크 경계에서 잘린 시나리오는 LLM 이 "다음 이벤트가 뭔지 확신 못해서" nextEvents 를 빈 배열로 둠.
- 청크 0 의 출력에 빈 nextEvents 가 깔리면, 청크 1+ 가 previousChunkSummary 로 그걸 보고 "이 프로젝트는
  nextEvents 안 채우는 컨벤션" 으로 cascading 패턴 학습.
- 결과: 누적 events 의 nextEvents 가 모두 비고, level 도 대부분 1 → 프론트 BPMN 시각화에서 모든 이벤트가
  "Single Events" 탭으로 떨어져 흐름 없이 그려짐.

해결:
- 모든 청크 처리가 끝난 뒤 stitcher 호출. 누적 events 를 BATCH_SIZE 단위로 쪼개 LLM 콜을 여러 번 함.
- 각 콜은 자기 배치 이벤트에만 nextEvents/level 을 채우되, **모든 이벤트 이름 목록을 namespace 로 제공**
  하므로 cross-batch 참조도 가능.
- 한 번에 100+ 이벤트를 던지면 LLM 이 작업 메모리/출력 토큰 한계로 보수적으로 빈 배열만 채워서
  품질이 망가짐 (14/117 같은 사례). 배치 단위로 쪼개면 콜 당 작업량이 줄어 quality 가 회복됨.
- 배치 간 병렬 처리 (ThreadPoolExecutor, P-GPT 게이트웨이 부하 고려해 BATCH_CONCURRENCY=3).
- 이벤트 추가/삭제/이름 변경 금지 (시각 모델 일관성 보존).
"""
from typing import Any, Dict, List, Set
from concurrent.futures import ThreadPoolExecutor
import json

from project_generator.utils.logging_util import LoggingUtil
from project_generator.utils.llm_factory import create_chat_llm


class EventFlowStitcher:
    """청크 단위로 추출된 이벤트 흐름(nextEvents)을 글로벌 컨텍스트로 보강."""

    # 매우 큰 요구사항 텍스트가 들어와도 프롬프트 토큰 폭주 방지.
    # 흐름 추론은 전체 텍스트 정독 없이도 이벤트 목록 + 설명으로 상당 부분 가능.
    MAX_REQUIREMENTS_CHARS = 60000

    # 한 LLM 콜이 다룰 이벤트 수. 25 가 sweet spot:
    # - 너무 작으면 (예: 10) 콜 수가 너무 많아 시간/비용 증가
    # - 너무 크면 (예: 50+) LLM 이 보수적으로 빈 nextEvents 만 채움
    BATCH_SIZE = 25

    # 배치 병렬 처리 동시성. P-GPT 게이트웨이 부하 고려해 3 으로 시작.
    # summarizer 와 동일 — 너무 키우면 429/연결거부, 너무 낮으면 직렬과 다를 게 없음.
    BATCH_CONCURRENCY = 3

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

            # 전체 이벤트 이름 namespace — 매 배치 콜에 같이 제공해서 cross-batch 참조 허용
            all_event_brief = [
                {
                    "name": e.get("name"),
                    "actor": e.get("actor"),
                    "displayName": e.get("displayName")
                }
                for e in events
                if e.get("name")
            ]
            valid_names: Set[str] = {e["name"] for e in all_event_brief}

            # 배치 분할
            batches: List[List[Dict]] = [
                events[i:i + self.BATCH_SIZE]
                for i in range(0, len(events), self.BATCH_SIZE)
            ]
            LoggingUtil.info(
                "EventFlowStitcher",
                f"{len(events)} events → {len(batches)} batches of up to {self.BATCH_SIZE} (concurrency={self.BATCH_CONCURRENCY})"
            )

            # 배치별 LLM 콜 병렬 실행
            stitch_index: Dict[str, Dict] = {}
            with ThreadPoolExecutor(max_workers=self.BATCH_CONCURRENCY) as executor:
                futures = [
                    executor.submit(
                        self._process_batch,
                        batch, batch_idx, len(batches),
                        all_event_brief, actors, requirements_text, valid_names
                    )
                    for batch_idx, batch in enumerate(batches)
                ]
                for fut in futures:
                    try:
                        partial = fut.result()
                        # 후행 배치가 같은 이름을 또 채울 일은 없지만 안전상 update 로 머지
                        stitch_index.update(partial)
                    except Exception as e:
                        LoggingUtil.error("EventFlowStitcher", f"Batch failed: {str(e)}")
                        # 한 배치 실패해도 나머지 결과는 사용

            # 머지: 원본 events 에 nextEvents/level 만 덮어쓰기
            stitched = self._merge(events, stitch_index)

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

    def _process_batch(
        self,
        batch_events: List[Dict],
        batch_idx: int,
        total_batches: int,
        all_event_brief: List[Dict],
        actors: List[Dict],
        requirements_text: str,
        valid_names: Set[str],
    ) -> Dict[str, Dict]:
        """배치 하나 LLM 콜 → {name: {nextEvents, level}} 반환."""
        try:
            LoggingUtil.info(
                "EventFlowStitcher",
                f"Batch {batch_idx + 1}/{total_batches}: processing {len(batch_events)} events"
            )

            prompt = self._build_batch_prompt(batch_events, all_event_brief, actors, requirements_text)
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, "content") else str(response)

            partial = self._parse_batch_response(response_text, batch_events, valid_names)
            filled = sum(1 for v in partial.values() if v.get("nextEvents"))
            LoggingUtil.info(
                "EventFlowStitcher",
                f"Batch {batch_idx + 1}/{total_batches}: filled nextEvents for {filled}/{len(batch_events)}"
            )
            return partial
        except Exception as e:
            LoggingUtil.error("EventFlowStitcher", f"Batch {batch_idx + 1} error: {str(e)}")
            return {}

    def _build_batch_prompt(
        self,
        batch_events: List[Dict],
        all_event_brief: List[Dict],
        actors: List[Dict],
        requirements_text: str,
    ) -> List[Dict]:
        # 이번 배치에서 LLM 이 채울 대상 (풀 디테일)
        events_for_llm = [
            {
                "name": e.get("name"),
                "actor": e.get("actor"),
                "displayName": e.get("displayName"),
                "description": e.get("description"),
                "currentLevel": e.get("level", 1),
                "currentNextEvents": e.get("nextEvents", []) or []
            }
            for e in batch_events
            if e.get("name")
        ]

        # nextEvents 의 valid target — 모든 이벤트 이름 + actor + displayName (cross-batch 참조 허용)
        all_names_for_llm = all_event_brief

        actors_for_llm = [
            {
                "name": a.get("name"),
                "events": a.get("events", []) or []
            }
            for a in actors
            if a.get("name")
        ]

        system_prompt = """You are an Expert Business Analyst & Domain-Driven Design Specialist.

You are given a SUBSET of business events that were extracted from a large requirements document. The full event list (names only) is also provided as the valid namespace for the `nextEvents` field. Your job is to fill in the inter-event flow (nextEvents) and sequence (level) for the events in the SUBSET.

**Your Task**
For each event in the SUBSET, populate:
- `nextEvents`: array of event names that LOGICALLY follow this event in the business process. The target names MUST come from the FULL event name list (so cross-subset references are allowed and encouraged when the business flow demands it).
- `level`: integer >= 1, indicating sequence priority in its containing process. Earlier events get lower level numbers; events at the same level are concurrent or independent.

**Hard Rules**
- Output exactly one entry per SUBSET event (same names, same count).
- You MUST NOT add events outside the subset to the output.
- You MUST NOT change event names.
- Every `nextEvents` entry MUST exactly match a `name` from the FULL event name list (case- and spacing-sensitive).
- An event may have zero or more nextEvents. Zero means it's a terminal/standalone event.

**How to Infer Flow**
1. Read each event's description and actor. The narrative in the original requirements describes business processes — find sequential cause-and-effect relations.
2. Events handled by the same actor are often sequential when they refer to consecutive steps of a workflow.
3. Cross-actor handoffs (e.g., Customer places order → System validates → Warehouse ships) are common and SHOULD be captured as nextEvents.
4. Branches: if an event leads to multiple possible next events (parallel or alternative), list all of them in nextEvents.
5. Terminal events: events that complete a workflow (e.g., OrderDelivered, PaymentSettled) usually have empty nextEvents.
6. **Default assumption**: most events ARE part of a chain. If you can find any plausible follow-up event in the full name list, include it. Only leave nextEvents empty when the event is clearly a workflow terminator.
7. Assign levels so that for each chain A -> B -> C, level(A) < level(B) < level(C). Concurrent events get the same level.

**Output Format (JSON only, no markdown, no extra text)**
{
    "events": [
        {"name": "EventA", "level": 1, "nextEvents": ["EventB", "EventC"]},
        {"name": "EventB", "level": 2, "nextEvents": ["EventD"]},
        ...
    ]
}

Return ALL events from the SUBSET (no more, no less). Order does not matter."""

        user_prompt = f"""**SUBSET to process** ({len(events_for_llm)} events — populate nextEvents + level for these):
{json.dumps(events_for_llm, ensure_ascii=False, indent=2)}

**FULL event name list** ({len(all_names_for_llm)} events — these are the only valid nextEvents targets):
{json.dumps(all_names_for_llm, ensure_ascii=False, indent=2)}

**Actors**:
{json.dumps(actors_for_llm, ensure_ascii=False, indent=2)}

**Original requirements** (may be truncated):
{requirements_text or '(not provided)'}

Now produce the JSON output. Remember: output entries ONLY for the SUBSET, but nextEvents may target ANY name from the FULL list."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def _parse_batch_response(
        self,
        response_text: str,
        batch_events: List[Dict],
        valid_names: Set[str],
    ) -> Dict[str, Dict]:
        """LLM 응답 → {name: {nextEvents, level}}. 배치에 속한 이벤트만 살림."""
        cleaned = self._extract_json(response_text)
        data = json.loads(cleaned)
        stitched = data.get("events", [])

        batch_names = {e.get("name") for e in batch_events if e.get("name")}
        result: Dict[str, Dict] = {}
        for item in stitched:
            name = item.get("name")
            if not name or name not in batch_names:
                # 배치 외 이벤트가 출력에 끼어들면 무시
                continue
            # nextEvents 가 입력에 없는 이름이면 필터링 (할루시네이션 차단)
            next_events = [
                n for n in (item.get("nextEvents") or [])
                if n in valid_names and n != name
            ]
            level = item.get("level")
            try:
                level = int(level) if level is not None else None
            except (ValueError, TypeError):
                level = None
            result[name] = {"nextEvents": next_events, "level": level}
        return result

    def _merge(self, original_events: List[Dict], stitch_index: Dict[str, Dict]) -> List[Dict]:
        """원본 events 에 nextEvents/level 만 덮어쓰기. 다른 필드는 그대로 보존."""
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
