from typing import Any, Dict, List, TypedDict
from datetime import datetime
import json
import re

from project_generator.utils.logging_util import LoggingUtil
from project_generator.utils.llm_factory import create_chat_llm


class RequirementsValidatorState(TypedDict):
    inputs: Dict[str, Any]
    inference: str
    progress: int
    result: Dict[str, Any]
    error: str
    timestamp: str


class RequirementsValidator:
    """
    LangGraph workflow for validating and analyzing requirements into Event Storming model.
    Extracts events, actors, and recommends bounded context count.
    """

    def __init__(self):
        self.llm = create_chat_llm(
            model="gpt-4.1-2025-04-14",
            temperature=0.3,
            streaming=False,
            model_kwargs={
                "response_format": {"type": "json_object"}
            }
        )

    def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            LoggingUtil.info("RequirementsValidator", "Requirements validation started")

            # 1. 입력 추출
            requirements_text = input_data.get('requirements', {}).get('userStory', '')
            previous_chunk_summary = input_data.get('previousChunkSummary', {})
            current_chunk_start_line = input_data.get('currentChunkStartLine', 1)

            if not requirements_text:
                LoggingUtil.info("RequirementsValidator", "No requirements provided")
                return self._create_empty_result()

            # 2. 라인 번호 추가
            line_numbered_requirements = self._add_line_numbers(requirements_text, current_chunk_start_line)

            # 3. 프롬프트 생성 및 LLM 호출
            prompt = self._build_prompt(line_numbered_requirements, previous_chunk_summary)
            response = self.llm.invoke(prompt)

            # 4. JSON 파싱
            response_text = response.content if hasattr(response, 'content') else str(response)
            result_data = json.loads(response_text)

            # 5. ANALYSIS_RESULT 처리 (type은 항상 ANALYSIS_RESULT)
            content = result_data.get('content', {})
            
            # 5-1. Refs 변환 (phrase → indexes)
            events = self._convert_event_refs(
                content.get('events', []),
                requirements_text,
                line_numbered_requirements,
                current_chunk_start_line
            )

            # 5-2. 결과 구성
            LoggingUtil.info(
                "RequirementsValidator",
                f"Validation completed: {len(events)} events, {len(content.get('actors', []))} actors"
            )

            return {
                'type': 'ANALYSIS_RESULT',
                'content': {
                    'recommendedBoundedContextsNumber': content.get('recommendedBoundedContextsNumber', 3),
                    'reasonOfRecommendedBoundedContextsNumber': content.get('reasonOfRecommendedBoundedContextsNumber', ''),
                    'events': events,
                    'actors': content.get('actors', [])
                },
                'progress': 100
            }

        except Exception as e:
            LoggingUtil.error("RequirementsValidator", f"Failed: {str(e)}")
            return {
                'type': 'ANALYSIS_RESULT',
                'content': {
                    'events': [],
                    'actors': [],
                    'recommendedBoundedContextsNumber': 3,
                    'reasonOfRecommendedBoundedContextsNumber': f'Error occurred: {str(e)}'
                },
                'progress': 100,
                'error': str(e)
            }

    def _add_line_numbers(self, text: str, start_line: int = 1, use_xml: bool = True) -> str:
        """텍스트에 라인 번호 추가"""
        lines = text.split('\n')
        numbered_lines = []
        for i, line in enumerate(lines):
            line_num = start_line + i
            if use_xml:
                numbered_lines.append(f'<{line_num}>{line}</{line_num}>')
            else:
                numbered_lines.append(f'{line_num}|{line}')
        return '\n'.join(numbered_lines)

    def _build_prompt(self, line_numbered_requirements: str, previous_chunk_summary: Dict) -> List[Dict]:
        """프론트엔드와 동일한 프롬프트 생성"""

        system_prompt = """You are an Expert Business Analyst & Domain-Driven Design Specialist.

Your goal is to transform business requirements into a comprehensive Big Picture Event Storming model that accurately represents the domain's business processes, actors, and event flows.

With extensive experience in business process analysis and domain-driven design, you specialize in decomposing complex requirements into event-driven models. You excel at identifying domain events, mapping actor responsibilities, and orchestrating event flows that capture the complete business narrative.

**Task:** Analyze business requirements and transform them into a comprehensive Big Picture Event Storming model.

**Requirements Analysis Process:**
1. Thorough Examination: Examine requirements in detail before generating events and actors
2. Business Goals Focus: Prioritize business objectives and user value delivery
3. Scenario Extraction: Identify and document all user scenarios and workflows
4. Business Rules Capture: Extract both implicit and explicit business rules and constraints
5. Process Dependencies: Map relationships and dependencies between business processes

**Event Discovery Methodology:**
1. Comprehensive Coverage: Convert EVERY significant business moment into domain events
2. Complete State Capture: Ensure ALL business-significant state changes are represented as events
3. Flow Completeness: Include both happy path scenarios AND exception flows
4. State Change Focus: Generate events ONLY for business-significant state changes. Do NOT create events for read-only operations
5. No Omissions: Do not skip or summarize any business processes
6. Naming Convention: Use PascalCase and past participle form (e.g., OrderPlaced, PaymentProcessed)
7. Primary Business Actions: Focus on the primary business action rather than secondary consequences

**Actor Identification Strategy:**
1. Event Ownership: Group events by their responsible actors (human or system)
2. Process Ownership: Establish clear accountability for each business process
3. Interaction Mapping: Define clear interaction points between different actors
4. Naming Consistency: Ensure actor names are consistent with exact spacing and case matching
5. Responsibility Separation: Keep actor responsibilities distinct and non-overlapping

**Event Flow and Relationships:**
1. Story Representation: Ensure every user story is reflected through events
2. Chain Completeness: Validate that event chains are logically complete
3. Event Connections: Consider connections to existing events when defining nextEvents
4. Sequence Logic: Maintain clear and logical event sequences using level numbers

**Bounded Context Recommendation:**
1. Context Analysis: Analyze actor interactions, domain boundaries, and business capabilities
2. Recommended Number: Suggest between 3 to 15 bounded contexts based on complexity
3. Clear Justification: Provide detailed rationale explaining which specific bounded contexts are recommended and why
4. Domain Alignment: Ensure bounded contexts align with business capabilities and organizational structure

**Source Traceability Requirements:**
1. Mandatory Refs: Every event MUST include refs linking back to specific requirement lines
2. Refs Format: Use format [[[startLineNumber, "minimal_start_phrase"], [endLineNumber, "minimal_end_phrase"]]]
3. Minimal Phrases: Use 1-2 word phrases that uniquely identify the position in the line
4. Valid Line Numbers: Refs must reference valid line numbers from the requirements section
5. Multiple References: Include multiple ranges if an event references multiple parts of requirements

**Consistency with Previous Analysis:**
1. Actor Name Matching: When using existing actor names from previous chunks, ensure exact spacing and case matching
2. Event Continuity: Consider potential connections to existing events when defining nextEvents
3. Context Alignment: Align event groupings with the overall bounded context structure

**CRITICAL OUTPUT REQUIREMENTS:**
- You MUST output ONLY valid JSON, no explanations, no markdown code fences
- Do NOT include ```json or ``` markers
- Do NOT include any text before or after the JSON object
- You MUST always generate events and actors, even if requirements are brief
- Type MUST ALWAYS be "ANALYSIS_RESULT"

**Output Format:**
{
    "type": "ANALYSIS_RESULT",
    "content": {
        "recommendedBoundedContextsNumber": (number: 3-15),
        "reasonOfRecommendedBoundedContextsNumber": "(Detailed analysis explaining which bounded contexts and why)",
        "events": [
            {
                "name": "(EventName in PascalCase & Past Participle)",
                "displayName": "(Natural language display name)",
                "actor": "(ActorName - must match an actor from actors array)",
                "level": (number: event sequence priority starting from 1),
                "description": "(Detailed description of what happened and why)",
                "inputs": ["(Required data or conditions)"],
                "outputs": ["(Resulting data or state changes)"],
                "nextEvents": ["(SubsequentEventName1)", "(SubsequentEventName2)"],
                "refs": [[[(startLineNumber), "(minimal_start_phrase)"], [(endLineNumber), "(minimal_end_phrase)"]]]
            }
        ],
        "actors": [
            {
                "name": "(ActorName - exact match with event.actor values)",
                "events": ["(AssociatedEventName1)", "(AssociatedEventName2)"],
                "lane": (number: 0-based vertical position for swimlane)
            }
        ]
    }
}"""

        # Previous chunk context 생성
        context_info = self._build_previous_chunk_context(previous_chunk_summary)

        user_prompt = f"""{context_info}

**Requirements Document:**
{line_numbered_requirements}

Please analyze the requirements and generate the Event Storming model."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def _build_previous_chunk_context(self, previous_chunk_summary: Dict) -> str:
        """이전 청크 컨텍스트 생성"""
        if not previous_chunk_summary or not previous_chunk_summary.get('events'):
            return """<previous_chunk_context>
    <status>first_chunk</status>
    <description>This is the first chunk. No previous context available.</description>
</previous_chunk_context>"""

        events = previous_chunk_summary.get('events', [])
        actors = previous_chunk_summary.get('actors', [])

        events_xml = '\n'.join([
            f"""        <event>
            <name>{e.get('name', '')}</name>
            <actor>{e.get('actor', '')}</actor>
            {''.join([f'<event>{ne}</event>' for ne in e.get('nextEvents', [])])}
        </event>"""
            for e in events
        ])

        actors_xml = '\n'.join([
            f"""        <actor>
            <name>{a.get('name', '')}</name>
            <handled_events>{''.join([f'<event>{evt}</event>' for evt in a.get('events', [])])}</handled_events>
        </actor>"""
            for a in actors
        ])

        return f"""<previous_chunk_context>
    <status>continuation</status>
    <description>This is a continuation of a previous analysis. Maintain consistency with the following existing elements.</description>
    
    <previously_identified_events>
{events_xml}
    </previously_identified_events>
    
    <previously_identified_actors>
{actors_xml}
    </previously_identified_actors>
    
    <consistency_requirements>
        <requirement>Ensure new events and actors are consistent with these existing elements</requirement>
        <requirement>When using existing actor names, match spacing and case exactly</requirement>
        <requirement>Consider potential connections to existing events when defining nextEvents</requirement>
    </consistency_requirements>
</previous_chunk_context>"""

    def _convert_event_refs(
        self,
        events: List[Dict],
        raw_requirements: str,
        line_numbered_requirements: str,
        start_line_offset: int
    ) -> List[Dict]:
        """Event refs를 phrase → indexes → absolute refs로 변환"""

        converted_events = []

        for event in events:
            refs = event.get('refs', [])
            if not refs:
                converted_events.append(event)
                continue

            # 1. sanitizeAndConvertRefs: phrase → [[[line, col], [line, col]]]
            sanitized_refs = self._sanitize_and_convert_refs(refs, line_numbered_requirements)

            # 2. absolute line numbers로 변환 (start_line_offset 적용)
            absolute_refs = []
            for ref in sanitized_refs:
                if isinstance(ref, list) and len(ref) >= 2:
                    start_line = ref[0][0] if isinstance(ref[0], list) and len(ref[0]) > 0 else 1
                    start_col = ref[0][1] if isinstance(ref[0], list) and len(ref[0]) > 1 else 0
                    end_line = ref[1][0] if isinstance(ref[1], list) and len(ref[1]) > 0 else 1
                    end_col = ref[1][1] if isinstance(ref[1], list) and len(ref[1]) > 1 else 0
                    
                    # offset 적용 (청크의 실제 시작 라인)
                    abs_start_line = start_line + start_line_offset - 1
                    abs_end_line = end_line + start_line_offset - 1
                    
                    absolute_refs.append([[abs_start_line, start_col], [abs_end_line, end_col]])

            event['refs'] = absolute_refs
            converted_events.append(event)

        return converted_events

    def _sanitize_and_convert_refs(self, refs: List, line_numbered_text: str) -> List:
        """Frontend RefsTraceUtil.sanitizeAndConvertRefs와 동일"""
        lines = line_numbered_text.split('\n')

        # 라인 번호 맵 생성
        line_number_map = {}
        for idx, line in enumerate(lines):
            match = re.match(r'^<(\d+)>.*</\1>$', line)
            if match:
                line_num = int(match.group(1))
                line_number_map[line_num] = idx

        sanitized = []
        for ref in refs:
            if not isinstance(ref, list) or len(ref) < 2:
                continue

            start_item = ref[0]
            end_item = ref[1]

            if not isinstance(start_item, list) or len(start_item) < 2:
                continue
            if not isinstance(end_item, list) or len(end_item) < 2:
                continue

            start_line = start_item[0]
            start_phrase = start_item[1] if isinstance(start_item[1], str) else ''
            end_line = end_item[0]
            end_phrase = end_item[1] if isinstance(end_item[1], str) else ''

            try:
                start_line_num = int(start_line)
                end_line_num = int(end_line)
            except (ValueError, TypeError):
                continue

            if start_line_num not in line_number_map or end_line_num not in line_number_map:
                continue

            start_idx = line_number_map[start_line_num]
            end_idx = line_number_map[end_line_num]

            if 0 <= start_idx < len(lines) and 0 <= end_idx < len(lines):
                start_line_text = lines[start_idx]
                end_line_text = lines[end_idx]

                # XML 태그 제거
                start_match = re.match(rf'^<{start_line_num}>(.*)</{start_line_num}>$', start_line_text)
                if start_match:
                    start_line_text = start_match.group(1)

                end_match = re.match(rf'^<{end_line_num}>(.*)</{end_line_num}>$', end_line_text)
                if end_match:
                    end_line_text = end_match.group(1)

                # phrase 위치 찾기
                start_col = 0
                if start_phrase and start_phrase in start_line_text:
                    start_col = start_line_text.find(start_phrase)

                end_col = len(end_line_text) - 1
                if end_phrase and end_phrase in end_line_text:
                    end_col = end_line_text.find(end_phrase) + len(end_phrase) - 1

                sanitized.append([[start_line_num, start_col], [end_line_num, end_col]])

        return sanitized

    def _create_empty_result(self) -> Dict[str, Any]:
        """빈 결과 생성"""
        return {
            'type': 'ANALYSIS_RESULT',
            'content': {
                'events': [],
                'actors': [],
                'recommendedBoundedContextsNumber': 3,
                'reasonOfRecommendedBoundedContextsNumber': 'No requirements provided'
            },
            'progress': 100
        }

