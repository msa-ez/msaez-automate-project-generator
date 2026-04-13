from typing import Any, Dict, List, TypedDict, Union
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
import json
import re

from project_generator.utils.logging_util import LoggingUtil
from project_generator.utils.refs_trace_util import RefsTraceUtil
from project_generator.utils.llm_factory import create_chat_llm


class TraceabilityGeneratorState(TypedDict):
    inputs: Dict[str, Any]
    inference: str
    progress: int
    result: Dict[str, Any]
    error: str
    timestamp: str


class TraceabilityGenerator:
    """
    LangGraph workflow for adding traceability (refs) to domain objects.
    Maps domain objects (aggregates, enumerations, valueObjects) to functional requirements.
    """

    def __init__(self):
        self.llm = create_chat_llm(
            model="gpt-4.1-2025-04-14",
            temperature=0,
            streaming=False
        )
        # Structured output으로 refs 필수 및 비어있지 않도록 강제
        # preview_fields_generator와 동일하게 anyOf 사용 + json_schema method
        self.llm_structured = self.llm.with_structured_output(
            self._get_response_schema(),
            method="json_schema",
            strict=True
        )

    def _restore_trace_map(self, trace_map):
        """Firebase가 배열로 변환한 traceMap을 객체로 복원"""
        if not trace_map:
            return {}
        
        # 이미 객체 형태인 경우 그대로 반환
        if isinstance(trace_map, dict) and not isinstance(trace_map, list):
            # 문자열 키 그대로 유지하되, 숫자 키로도 접근 가능하도록 양쪽 모두 저장
            # (프론트엔드 RefsTraceUtil.convertToOriginalRefsUsingTraceMap에서 traceMap[i] || traceMap[String(i)]로 접근)
            normalized_trace_map = {}
            for key, value in trace_map.items():
                normalized_trace_map[key] = value
                # 숫자로 변환 가능하면 숫자 키로도 저장 (양쪽 모두 접근 가능)
                try:
                    num_key = int(key)
                    if str(num_key) == str(key):
                        normalized_trace_map[num_key] = value
                except (ValueError, TypeError):
                    pass
            return normalized_trace_map
        
        # 배열 형태인 경우 변환
        restored_trace_map = {}
        if isinstance(trace_map, list):
            processed_keys = []  # 디버깅용: 처리된 키 추적
            for index, item in enumerate(trace_map):
                if not item:
                    continue
                # {"key":"4","value":{...}} 형태인 경우
                if isinstance(item, dict) and 'key' in item and 'value' in item:
                    key = item['key']
                    value = item['value']
                    # 문자열 키로 저장
                    restored_trace_map[str(key)] = value
                    # 숫자로 변환 가능하면 숫자 키로도 저장 (양쪽 모두 접근 가능)
                    try:
                        num_key = int(key)
                        # 숫자 키로도 저장 (문자열 키와 숫자 키 모두 접근 가능하도록)
                        restored_trace_map[num_key] = value
                        processed_keys.append(num_key)
                    except (ValueError, TypeError):
                        pass
                else:
                    # Firebase가 비연속 숫자 키 객체를 배열로 변환한 경우
                    # 배열 인덱스가 원래 키와 일치함 (예: 인덱스 4 = 원래 키 "4")
                    # None이 아닌 항목의 인덱스를 키로 사용
                    if isinstance(item, dict):
                        # 인덱스를 키로 사용 (문자열과 숫자 모두 저장)
                        restored_trace_map[index] = item
                        restored_trace_map[str(index)] = item
                        processed_keys.append(index)
            
            # 디버깅: 홀수/짝수 키 확인
            if processed_keys:
                odd_keys = sorted([k for k in processed_keys if k % 2 == 1])[:10]
                even_keys = sorted([k for k in processed_keys if k % 2 == 0])[:10]
                from project_generator.utils.logging_util import LoggingUtil
                LoggingUtil.debug("TraceabilityGenerator", 
                    f"_restore_trace_map: 처리된 키 샘플 - 짝수: {even_keys}, 홀수: {odd_keys}, 총 키 수: {len(processed_keys)}")
        
        return restored_trace_map
    
    def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            LoggingUtil.info("TraceabilityGenerator", "Traceability generation started")

            # 1. 입력 추출
            generated_draft_options = input_data.get('generatedDraftOptions', [])
            bounded_context_name = input_data.get('boundedContextName', '')
            functional_requirements = input_data.get('functionalRequirements', '')
            trace_map = input_data.get('traceMap', {})
            
            # traceMap 복원 (Firebase가 배열로 변환한 경우 처리)
            trace_map = self._restore_trace_map(trace_map)
            
            # 입력 데이터 검증
            if not functional_requirements or not functional_requirements.strip():
                LoggingUtil.warning("TraceabilityGenerator", 
                    f"functionalRequirements is empty or None! boundedContextName: {bounded_context_name}")
                return {
                    'draftTraceMap': {'aggregates': [], 'enumerations': [], 'valueObjects': []},
                    'inference': f'Traceability skipped: functionalRequirements is empty for {bounded_context_name}',
                    'progress': 100
                }

            # 2. 모든 도메인 객체 추출 (중복 제거)
            all_domain_objects = self._extract_all_domain_objects(generated_draft_options)

            # 3. 필터링된 옵션 생성 (name, alias만)
            filtered_options = self._filter_generated_draft_options(generated_draft_options)

            # 4. 라인 번호 추가
            line_numbered_requirements = self._add_line_numbers(functional_requirements)

            # 5. 프롬프트 생성 및 LLM 호출 (structured output 사용)
            prompt = self._build_prompt(
                filtered_options,
                bounded_context_name,
                line_numbered_requirements,
                all_domain_objects
            )

            # Structured output으로 호출하여 refs 필수 및 비어있지 않도록 강제
            result_data = self.llm_structured.invoke(prompt)

            # 6. refs 변환 (phrase → indexes)
            draft_trace_map = self._convert_refs_to_indexes(
                result_data,
                functional_requirements,
                line_numbered_requirements,
                trace_map
            )

            LoggingUtil.info(
                "TraceabilityGenerator",
                f"Traceability completed: {len(draft_trace_map.get('aggregates', []))} aggregates, "
                f"{len(draft_trace_map.get('enumerations', []))} enumerations, "
                f"{len(draft_trace_map.get('valueObjects', []))} valueObjects"
            )

            return {
                'draftTraceMap': draft_trace_map,
                'inference': f'Traceability added for {bounded_context_name}',
                'progress': 100
            }

        except Exception as e:
            LoggingUtil.error("TraceabilityGenerator", f"Failed: {str(e)}")
            return {
                'draftTraceMap': {'aggregates': [], 'enumerations': [], 'valueObjects': []},
                'inference': f'Error: {str(e)}',
                'progress': 100,
                'error': str(e)
            }

    def _extract_all_domain_objects(self, generated_draft_options: List[Dict]) -> Dict[str, List[Dict]]:
        """모든 옵션에서 도메인 객체 추출 (중복 제거)"""
        all_objects = {
            'aggregates': [],
            'enumerations': [],
            'valueObjects': []
        }

        seen_aggregates = set()
        seen_enumerations = set()
        seen_value_objects = set()

        for option in generated_draft_options:
            structure_array = option.get('structure', option)
            if not isinstance(structure_array, list):
                structure_array = [structure_array]

            for aggregate_info in structure_array:
                # Aggregate
                if aggregate_info.get('aggregate') and aggregate_info['aggregate'].get('name'):
                    name = aggregate_info['aggregate']['name']
                    if name not in seen_aggregates:
                        all_objects['aggregates'].append({
                            'name': name,
                            'alias': aggregate_info['aggregate'].get('alias', '')
                        })
                        seen_aggregates.add(name)

                # Enumerations
                if aggregate_info.get('enumerations'):
                    for enumeration in aggregate_info['enumerations']:
                        name = enumeration.get('name')
                        if name and name not in seen_enumerations:
                            all_objects['enumerations'].append({
                                'name': name,
                                'alias': enumeration.get('alias', '')
                            })
                            seen_enumerations.add(name)

                # Value Objects
                if aggregate_info.get('valueObjects'):
                    for value_object in aggregate_info['valueObjects']:
                        name = value_object.get('name')
                        if name and name not in seen_value_objects:
                            all_objects['valueObjects'].append({
                                'name': name,
                                'alias': value_object.get('alias', '')
                            })
                            seen_value_objects.add(name)

        return all_objects

    def _filter_generated_draft_options(self, generated_draft_options: List[Dict]) -> List[List[Dict]]:
        """name, alias만 남기고 필터링"""
        filtered = []
        for option in generated_draft_options:
            filtered_structure = []
            structure = option.get('structure', [])
            for struct in structure:
                filtered_agg_info = {
                    'aggregate': {
                        'name': struct.get('aggregate', {}).get('name', ''),
                        'alias': struct.get('aggregate', {}).get('alias', '')
                    },
                    'enumerations': [
                        {'name': e.get('name', ''), 'alias': e.get('alias', '')}
                        for e in struct.get('enumerations', [])
                    ],
                    'valueObjects': [
                        {'name': v.get('name', ''), 'alias': v.get('alias', '')}
                        for v in struct.get('valueObjects', [])
                    ]
                }
                filtered_structure.append(filtered_agg_info)
            if filtered_structure:
                filtered.append(filtered_structure)
        return filtered

    def _add_line_numbers(self, text: str, start_line: int = 1, use_xml: bool = True) -> str:
        """텍스트에 라인 번호 추가 (프론트엔드 TextTraceUtil.addLineNumbers와 동일)"""
        lines = text.split('\n')
        numbered_lines = []
        for i, line in enumerate(lines):
            line_num = start_line + i
            if use_xml:
                numbered_lines.append(f'<{line_num}>{line}</{line_num}>')
            else:
                numbered_lines.append(f'{line_num}|{line}')
        return '\n'.join(numbered_lines)

    def _build_prompt(
        self,
        filtered_options: List[List[Dict]],
        bounded_context_name: str,
        line_numbered_requirements: str,
        all_domain_objects: Dict[str, List[Dict]]
    ) -> List[Dict]:
        """프론트엔드와 동일한 프롬프트 생성"""
        
        system_prompt = """You are a Domain-Driven Design (DDD) Traceability Expert.

Your goal is to establish precise traceability mappings between pre-generated domain objects and their source functional requirements, ensuring every domain element has clear justification in the business requirements.

<instruction>
    <core_instructions>
        <title>Domain Object Traceability Mapping Task</title>
        <task_description>Your task is to add traceability information (refs) to pre-generated domain objects by mapping them to specific parts of the functional requirements. You must establish clear traceability links that justify the existence of each domain object.</task_description>
        
        <input_description>
            <title>You will be given:</title>
            <item id="1">**Generated Draft Options:** Pre-generated domain objects (aggregates, enumerations, value objects) that need traceability</item>
            <item id="2">**Functional Requirements:** Line-numbered business requirements document</item>
            <item id="3">**Bounded Context Name:** The target bounded context for these domain objects</item>
            <item id="4">**Domain Objects to Trace:** Complete list of all domain objects requiring traceability</item>
        </input_description>

        <guidelines>
            <title>Traceability Mapping Guidelines</title>
            
            <section id="requirements_analysis">
                <title>Requirements Analysis Process</title>
                <rule id="1">**Business Context Understanding:** Carefully analyze the functional requirements to understand the business domain and context</rule>
                <rule id="2">**Concept Identification:** Identify text segments that describe business concepts, entities, processes, and data elements</rule>
                <rule id="3">**Explicit Mentions:** Look for explicit mentions of entities, statuses, enumerations, processes, and business rules</rule>
                <rule id="4">**Conceptual Relationships:** Consider both direct mentions and implied conceptual relationships in the requirements</rule>
            </section>

            <section id="domain_object_mapping">
                <title>Domain Object Mapping Strategy</title>
                <rule id="1">**Justification Matching:** For each domain object, find the specific requirement text that justifies its creation</rule>
                <rule id="2">**Name Alignment:** Match object names and aliases to corresponding business concepts in the requirements</rule>
                <rule id="3">**Type-Specific Mapping:** Apply appropriate mapping strategies for different object types:
                    - **Aggregates:** Map to entity descriptions, business object definitions, or core domain concepts
                    - **Enumerations:** Map to status values, type classifications, or categorical data mentions
                    - **Value Objects:** Map to attribute groups, specification descriptions, or composite data elements</rule>
                <rule id="4">**Complete Coverage:** Every provided domain object must be mapped with at least one traceability reference</rule>
            </section>

            <section id="traceability_reference_format">
                <title>Traceability Reference (refs) Format</title>
                <rule id="1">**Mandatory Refs:** Each domain object MUST include a 'refs' array containing precise references to requirement text</rule>
                <rule id="2">**Format Structure:** Use format [[[startLineNumber, "minimal_start_phrase"], [endLineNumber, "minimal_end_phrase"]]]</rule>
                <rule id="3">**Minimal Phrases:** Use MINIMAL phrases (1-2 words) that uniquely identify the position in the requirement line</rule>
                <rule id="4">**Shortest Identification:** Use the shortest possible phrase that can accurately locate the specific part of requirements</rule>
                <rule id="5">**Valid Line Numbers:** Only reference line numbers that exist in the provided functional requirements section</rule>
                <rule id="6">**Multiple References:** Include multiple ranges if a domain object is derived from multiple requirement sections</rule>
                <rule id="7">**Precision:** Ensure referenced text actually supports and justifies the existence of the domain object</rule>
                <rule id="8">**CRITICAL:** startLineNumber and endLineNumber MUST be NUMBERS (not strings). Phrases MUST be NON-EMPTY strings (at least 1 character). Empty strings "" in phrases are STRICTLY FORBIDDEN.</rule>
            </section>

            <section id="accuracy_requirements">
                <title>Precision and Accuracy Standards</title>
                <rule id="1">**Exact Segments:** Be precise in identifying the exact text segments that justify each domain object</rule>
                <rule id="2">**Avoid Vagueness:** Avoid generic or vague references that don't clearly support the domain object</rule>
                <rule id="3">**Verification:** Ensure that the referenced text actually justifies the domain object's existence and characteristics</rule>
                <rule id="4">**Comprehensive Mapping:** If multiple requirement sections contribute to a single domain object, include all relevant references</rule>
            </section>
        </guidelines>

        <refs_format_example>
            <title>Example of refs Format for Domain Objects</title>
            <description>If functional requirements contain:</description>
            <example_requirements>
<1># Hotel Room Management System</1>
<2></2>
<3>Room registration with room number, type, and capacity</3>
<4>Room status tracking (Available, Occupied, Cleaning)</4>
<5>Maintenance scheduling for room repairs</5>
<6>Housekeeping staff can update cleaning status</6>
            </example_requirements>
            <example_traceability>
- **"Room" aggregate** based on room registration → refs: [[[3, "Room"], [3, "capacity"]]]
- **"RoomStatus" enumeration** for status values → refs: [[[4, "status"], [4, "Cleaning"]]]
- **"MaintenanceSchedule" value object** for repair tracking → refs: [[[5, "Maintenance"], [5, "repairs"]]]
            </example_traceability>
            <format_explanation>
- The refs array contains ranges where each range is [[startLine, startPhrase], [endLine, endPhrase]]
- Phrases should be MINIMAL words (1-2 words) that uniquely identify the position
- Use the shortest possible phrase that can locate the specific part of requirements
- Multiple ranges can be included if a domain object references multiple requirement sections
- **CRITICAL:** In `[3, "Room"]`, the first element `3` is a NUMBER (line number), the second element `"Room"` is a STRING (phrase)
            </format_explanation>
        </refs_format_example>
    </core_instructions>
    
    <output_format>
        <title>JSON Output Format</title>
        <description>The output must be a JSON object structured as follows:</description>
        <schema>
{
    "aggregates": [
        {
            "name": "(Aggregate name matching provided domain objects)",
            "refs": [[[startLineNumber, "minimal_start_phrase"], [endLineNumber, "minimal_end_phrase"]]]
        }
    ],
    "enumerations": [
        {
            "name": "(Enumeration name matching provided domain objects)",
            "refs": [[[startLineNumber, "minimal_start_phrase"], [endLineNumber, "minimal_end_phrase"]]]
        }
    ],
    "valueObjects": [
        {
            "name": "(Value object name matching provided domain objects)",
            "refs": [[[startLineNumber, "minimal_start_phrase"], [endLineNumber, "minimal_end_phrase"]]]
        }
    ]
}
        </schema>
        <field_requirements>
            <requirement id="1">All domain object names must exactly match the names provided in the input</requirement>
            <requirement id="2">Every domain object from the input must be included in the output with refs</requirement>
            <requirement id="3">Line numbers in refs must be valid (exist in the requirements document)</requirement>
            <requirement id="4">Phrases in refs must be minimal (1-2 words) and accurately identify the location</requirement>
            <requirement id="5">**CRITICAL:** You MUST output ONLY valid JSON, no explanations, no markdown code fences. Do NOT include ```json or ``` markers. Do NOT include any text before or after the JSON object.</requirement>
            <requirement id="6">**CRITICAL:** Every domain object MUST have non-empty refs array. Empty refs arrays [] are NOT allowed. Empty strings "" in phrases are STRICTLY FORBIDDEN.</requirement>
        </field_requirements>
    </output_format>
</instruction>"""

        user_prompt = f"""**Generated Draft Options:**
{json.dumps(filtered_options, ensure_ascii=False, indent=2)}

**Bounded Context Name:** {bounded_context_name}

**Functional Requirements:**
{line_numbered_requirements}

**Domain Objects to Trace:**
{json.dumps(all_domain_objects, ensure_ascii=False, indent=2)}

Please provide traceability mappings for all domain objects listed above."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def _get_response_schema(self) -> Dict:
        """Structured output을 위한 JSON 스키마 정의 (refs 필수 및 비어있지 않도록 강제)"""
        return {
            "title": "TraceabilityResponse",
            "description": "Response schema for traceability generation with refs for domain objects",
            "type": "object",
            "properties": {
                "aggregates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "refs": {
                                "type": "array",
                                "minItems": 1,  # 최소 1개 refs 필수
                                "items": {
                                    "type": "array",
                                    "minItems": 2,
                                    "maxItems": 2,
                                    "items": {
                                        "type": "array",
                                        "minItems": 2,
                                        "maxItems": 2,
                                        "items": {
                                            "anyOf": [
                                                {"type": "number"},  # line number 또는 phrase
                                                {"type": "string", "minLength": 1}  # phrase는 최소 1자 이상
                                            ]
                                        }
                                    }
                                }
                            }
                        },
                        "required": ["name", "refs"],
                        "additionalProperties": False
                    }
                },
                "enumerations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "refs": {
                                "type": "array",
                                "minItems": 1,  # 최소 1개 refs 필수
                                "items": {
                                    "type": "array",
                                    "minItems": 2,
                                    "maxItems": 2,
                                    "items": {
                                        "type": "array",
                                        "minItems": 2,
                                        "maxItems": 2,
                                        "items": {
                                            "anyOf": [
                                                {"type": "number"},  # line number 또는 phrase
                                                {"type": "string", "minLength": 1}  # phrase는 최소 1자 이상
                                            ]
                                        }
                                    }
                                }
                            }
                        },
                        "required": ["name", "refs"],
                        "additionalProperties": False
                    }
                },
                "valueObjects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "refs": {
                                "type": "array",
                                "minItems": 1,  # 최소 1개 refs 필수
                                "items": {
                                    "type": "array",
                                    "minItems": 2,
                                    "maxItems": 2,
                                    "items": {
                                        "type": "array",
                                        "minItems": 2,
                                        "maxItems": 2,
                                        "items": {
                                            "anyOf": [
                                                {"type": "number"},  # line number 또는 phrase
                                                {"type": "string", "minLength": 1}  # phrase는 최소 1자 이상
                                            ]
                                        }
                                    }
                                }
                            }
                        },
                        "required": ["name", "refs"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["aggregates", "enumerations", "valueObjects"],
            "additionalProperties": False
        }

    def _convert_refs_to_indexes(
        self,
        output: Dict,
        raw_requirements: str,
        line_numbered_requirements: str,
        trace_map: Dict
    ) -> Dict:
        """refs를 phrase → indexes로 변환 (프론트엔드 RefsTraceUtil과 동일)"""
        
        for object_type in ['aggregates', 'enumerations', 'valueObjects']:
            if object_type not in output:
                continue

            for domain_object in output[object_type]:
                if 'refs' not in domain_object or not domain_object['refs']:
                    continue

                # 1. sanitizeAndConvertRefs: phrase → [[[line, col], [line, col]]]
                original_refs = domain_object['refs']
                # 공통 유틸리티 사용 (프론트엔드와 동일)
                sanitized_data = RefsTraceUtil.sanitize_and_convert_refs(
                    {'refs': original_refs},
                    line_numbered_requirements,
                    is_use_xml_base=True
                )
                sanitized_refs = sanitized_data.get('refs', original_refs) if isinstance(sanitized_data, dict) else sanitized_data
                
                if not sanitized_refs:
                    LoggingUtil.warning("TraceabilityGenerator", 
                        f"{object_type} '{domain_object.get('name', 'unknown')}'의 refs 변환 실패: 원본={original_refs}")
                    domain_object['refs'] = []
                    continue
                
                domain_object['refs'] = sanitized_refs

                # 2. validateRefs: 범위 검증 (예외 발생 시 빈 배열로 처리)
                try:
                    self._validate_refs(domain_object['refs'], raw_requirements)
                except Exception as e:
                    LoggingUtil.warning("TraceabilityGenerator", 
                        f"{object_type} '{domain_object.get('name', 'unknown')}'의 refs 검증 실패: {str(e)}")
                    domain_object['refs'] = []
                    continue

                # 3. convertToOriginalRefsUsingTraceMap: traceMap 사용해 원본 라인으로 역변환
                # 공통 유틸리티 사용 (프론트엔드와 동일, 클램핑 없음)
                converted_refs = RefsTraceUtil.convert_to_original_refs_using_trace_map(
                    domain_object['refs'],
                    trace_map
                )
                
                if not converted_refs:
                    LoggingUtil.warning("TraceabilityGenerator", 
                        f"{object_type} '{domain_object.get('name', 'unknown')}'의 traceMap 변환 실패: sanitized_refs={sanitized_refs}")
                    domain_object['refs'] = []
                else:
                    domain_object['refs'] = converted_refs

        return output

    def _sanitize_and_convert_refs(self, refs: List, line_numbered_requirements: str) -> List:
        """Frontend RefsTraceUtil.convertRefsToIndexes와 동일 (phrase → column indexes)"""
        if not refs or not line_numbered_requirements:
            return refs
        
        lines = line_numbered_requirements.split('\n')
        
        # 최소 라인 번호 찾기
        min_line = None
        for line in lines:
            match = re.match(r'^<(\d+)>.*</\1>$', line)
            if match:
                line_num = int(match.group(1))
                if min_line is None or line_num < min_line:
                    min_line = line_num
        
        if min_line is None:
            min_line = 1
        
        start_line_offset = min_line - 1
        
        sanitized = []
        for ref in refs:
            if not isinstance(ref, list):
                continue
            
            # 단일 ref 형식: [[lineNumber, phrase]]
            if len(ref) == 1:
                single_ref = ref[0]
                if not isinstance(single_ref, list) or len(single_ref) < 2:
                    continue
                
                line_number = single_ref[0]
                phrase = single_ref[1] if isinstance(single_ref[1], str) else ''
                
                try:
                    line_num = int(line_number)
                except (ValueError, TypeError):
                    continue
                
                # 이미 변환된 경우 (phrase가 number인 경우)
                if isinstance(phrase, (int, float)):
                    sanitized.append([[line_num, int(phrase)]])
                    continue
                
                # 빈 phrase 처리: 전체 라인으로 처리
                if not phrase or not phrase.strip():
                    # 라인 컨텐츠 가져오기
                    adjusted_line_num = line_num - start_line_offset
                    if adjusted_line_num < 1 or adjusted_line_num > len(lines):
                        continue
                    line_content = lines[adjusted_line_num - 1]
                    # XML 태그 제거
                    match = re.match(rf'^<{line_num}>(.*)</{line_num}>$', line_content)
                    if match:
                        line_content = match.group(1)
                    line_end_index = len(line_content) if line_content else 1
                    sanitized.append([[line_num, 1], [line_num, line_end_index]])
                    continue
                
                # 라인 컨텐츠 가져오기
                adjusted_line_num = line_num - start_line_offset
                if adjusted_line_num < 1 or adjusted_line_num > len(lines):
                    continue
                
                line_content = lines[adjusted_line_num - 1]
                # XML 태그 제거
                match = re.match(rf'^<{line_num}>(.*)</{line_num}>$', line_content)
                if match:
                    line_content = match.group(1)
                
                # phrase 찾기 (대소문자 무시)
                phrase_index = line_content.find(phrase) if phrase else -1
                if phrase_index == -1 and phrase:
                    # 대소문자 무시해서 다시 시도
                    phrase_lower = phrase.lower()
                    line_content_lower = line_content.lower()
                    phrase_index_lower = line_content_lower.find(phrase_lower)
                    if phrase_index_lower != -1:
                        phrase_index = phrase_index_lower
                
                if phrase_index != -1:
                    start_index = phrase_index + 1  # 1-based
                    end_index = phrase_index + len(phrase)
                    sanitized.append([[line_num, start_index], [line_num, end_index]])
                else:
                    # phrase를 찾지 못하면 전체 라인
                    line_end_index = len(line_content) if line_content else 1
                    sanitized.append([[line_num, 1], [line_num, line_end_index]])
            
            # 이중 ref 형식: [[start], [end]]
            elif len(ref) == 2:
                start_ref = ref[0]
                end_ref = ref[1]
                
                if not isinstance(start_ref, list) or len(start_ref) < 2:
                    continue
                if not isinstance(end_ref, list) or len(end_ref) < 2:
                    continue
                
                start_line = start_ref[0]
                start_phrase = start_ref[1] if isinstance(start_ref[1], str) else ''
                end_line = end_ref[0]
                end_phrase = end_ref[1] if isinstance(end_ref[1], str) else ''
                
                try:
                    start_line_num = int(start_line)
                    end_line_num = int(end_line)
                except (ValueError, TypeError):
                    continue
                
                # 이미 변환된 경우
                if isinstance(start_phrase, (int, float)) and isinstance(end_phrase, (int, float)):
                    sanitized.append([[start_line_num, int(start_phrase)], [end_line_num, int(end_phrase)]])
                    continue
                
                # 빈 phrase 처리: 전체 라인으로 처리
                if (not start_phrase or not start_phrase.strip()) and (not end_phrase or not end_phrase.strip()):
                    # 시작 라인 컨텐츠
                    start_adjusted = start_line_num - start_line_offset
                    if start_adjusted < 1 or start_adjusted > len(lines):
                        continue
                    start_line_content = lines[start_adjusted - 1]
                    start_match = re.match(rf'^<{start_line_num}>(.*)</{start_line_num}>$', start_line_content)
                    if start_match:
                        start_line_content = start_match.group(1)
                    
                    # 끝 라인 컨텐츠
                    end_adjusted = end_line_num - start_line_offset
                    if end_adjusted < 1 or end_adjusted > len(lines):
                        continue
                    end_line_content = lines[end_adjusted - 1]
                    end_match = re.match(rf'^<{end_line_num}>(.*)</{end_line_num}>$', end_line_content)
                    if end_match:
                        end_line_content = end_match.group(1)
                    
                    start_index = 1
                    end_index = len(end_line_content) if end_line_content else 1
                    sanitized.append([[start_line_num, start_index], [end_line_num, end_index]])
                    continue
                
                # 시작 라인 컨텐츠
                start_adjusted = start_line_num - start_line_offset
                if start_adjusted < 1 or start_adjusted > len(lines):
                    continue
                start_line_content = lines[start_adjusted - 1]
                start_match = re.match(rf'^<{start_line_num}>(.*)</{start_line_num}>$', start_line_content)
                if start_match:
                    start_line_content = start_match.group(1)
                
                # 끝 라인 컨텐츠
                end_adjusted = end_line_num - start_line_offset
                if end_adjusted < 1 or end_adjusted > len(lines):
                    continue
                end_line_content = lines[end_adjusted - 1]
                end_match = re.match(rf'^<{end_line_num}>(.*)</{end_line_num}>$', end_line_content)
                if end_match:
                    end_line_content = end_match.group(1)
                
                # phrase 위치 찾기
                start_index = self._find_column_for_words(start_line_content, start_phrase, False)
                end_index = self._find_column_for_words(end_line_content, end_phrase, True)
                
                sanitized.append([[start_line_num, start_index], [end_line_num, end_index]])
        
        return sanitized
    
    def _find_column_for_words(self, line_content: str, phrase: str, is_end_position: bool) -> int:
        """프론트엔드 _findColumnForWords와 동일"""
        if not line_content:
            return 1
        if not phrase:
            return len(line_content) if is_end_position else 1
        
        index = line_content.find(phrase)
        if index == -1:
            return len(line_content) if is_end_position else 1
        
        return (index + len(phrase)) if is_end_position else (index + 1)  # 1-based

    def _validate_refs(self, refs: List, raw_requirements: str):
        """refs 범위 검증"""
        lines = raw_requirements.split('\n')
        for ref in refs:
            if not isinstance(ref, list) or len(ref) < 2:
                continue
            start_line = ref[0][0] if isinstance(ref[0], list) and len(ref[0]) > 0 else 1
            end_line = ref[1][0] if isinstance(ref[1], list) and len(ref[1]) > 0 else 1
            
            # 1-based → 0-based
            if start_line < 1 or start_line > len(lines):
                raise ValueError(f"Invalid start line: {start_line}")
            if end_line < 1 or end_line > len(lines):
                raise ValueError(f"Invalid end line: {end_line}")

    def _convert_to_original_refs_using_trace_map(self, refs: List, trace_map: Dict, raw_requirements: str = None) -> List:
        """traceMap을 사용해 원본 라인으로 역변환 (Frontend RefsTraceUtil.convertToOriginalRefsUsingTraceMap와 동일)"""
        if not refs or not trace_map:
            return refs

        # traceMap: { lineNumber: { "refs": [[[line, col], [line, col]]], "isDirectMatching": bool } }
        # 프론트엔드와 동일하게 라인 번호를 키로 사용
        
        original_refs = []
        processed_trace_infos = set()  # 중복 처리 방지
        
        for ref in refs:
            if not isinstance(ref, list) or len(ref) < 2:
                continue

            start_line = ref[0][0] if isinstance(ref[0], list) and len(ref[0]) > 0 else 1
            start_index = ref[0][1] if isinstance(ref[0], list) and len(ref[0]) > 1 else 0
            end_line = ref[1][0] if isinstance(ref[1], list) and len(ref[1]) > 0 else 1
            end_index = ref[1][1] if isinstance(ref[1], list) and len(ref[1]) > 1 else 0

            # 현재 ref 범위에서 고유한 traceInfo들을 수집
            unique_trace_infos = {}
            missing_keys = []
            for i in range(start_line, end_line + 1):
                # traceMap 키가 문자열일 수 있으므로 정수와 문자열 모두 시도
                trace_info = trace_map.get(i) or trace_map.get(str(i))
                if not trace_info:
                    missing_keys.append(i)
                    continue
                
                # traceKey 생성 (refs를 JSON 문자열로 변환)
                import json
                trace_key = json.dumps(trace_info.get('refs', []), sort_keys=True)
                if trace_key not in unique_trace_infos:
                    unique_trace_infos[trace_key] = {'traceInfo': trace_info, 'line': i}

            # 각 고유한 traceInfo에 대해 처리
            for trace_key, trace_data in unique_trace_infos.items():
                if trace_key in processed_trace_infos:
                    continue
                processed_trace_infos.add(trace_key)

                trace_info = trace_data['traceInfo']
                line = trace_data['line']

                if (trace_info.get('isDirectMatching') and 
                    trace_info.get('refs') and 
                    len(trace_info['refs']) == 1 and 
                    len(trace_info['refs'][0]) == 2):
                    
                    trace_ref = trace_info['refs'][0]
                    trace_start_line = trace_ref[0][0] if isinstance(trace_ref[0], list) and len(trace_ref[0]) > 0 else 1
                    trace_start_index = trace_ref[0][1] if isinstance(trace_ref[0], list) and len(trace_ref[0]) > 1 else 0
                    trace_end_line = trace_ref[1][0] if isinstance(trace_ref[1], list) and len(trace_ref[1]) > 0 else 1
                    trace_end_index = trace_ref[1][1] if isinstance(trace_ref[1], list) and len(trace_ref[1]) > 1 else 0
                    
                    if trace_start_line != trace_end_line:
                        if end_line - start_line == trace_end_line - trace_start_line:
                            # 여러 줄에 걸쳐 있고 줄 수가 같은 경우: 라인별 매핑
                            line_offset = line - start_line
                            corresponding_trace_line = trace_start_line + line_offset
                            
                            if line == start_line:
                                # 첫 번째 라인: 원본의 startIndex 사용
                                calculated_start_index = start_index
                                calculated_end_index = trace_end_index
                            elif line == end_line:
                                # 마지막 라인: 원본의 endIndex 사용
                                calculated_start_index = trace_start_index
                                calculated_end_index = end_index
                            else:
                                # 중간 라인: trace의 전체 범위 사용
                                calculated_start_index = trace_start_index
                                calculated_end_index = trace_end_index
                            
                            # 인덱스 유효성 검증
                            if calculated_start_index <= calculated_end_index:
                                original_refs.append([[corresponding_trace_line, calculated_start_index], 
                                                      [corresponding_trace_line, calculated_end_index]])
                            else:
                                original_refs.append([[corresponding_trace_line, trace_start_index], 
                                                      [corresponding_trace_line, trace_end_index]])
                        else:
                            # 줄 수가 다른 경우: 전체 trace refs 사용 (프론트엔드와 동일)
                            original_refs.extend(trace_info['refs'])
                    elif start_line == end_line:
                        # 단일 라인 처리 (프론트엔드와 동일)
                        if start_index > end_index or start_index < 1:
                            original_refs.append([[trace_start_line, trace_start_index], 
                                                  [trace_end_line, trace_end_index]])
                        else:
                            original_refs.append([[trace_start_line, start_index], 
                                                  [trace_end_line, end_index]])
                    else:
                        # 원본이 여러 줄, trace가 단일 줄인 경우 (프론트엔드와 동일)
                        if line == start_line:
                            calculated_start_index = start_index
                            calculated_end_index = trace_end_index
                        elif line == end_line:
                            calculated_start_index = trace_start_index
                            calculated_end_index = end_index
                        else:
                            calculated_start_index = trace_start_index
                            calculated_end_index = trace_end_index

                        if calculated_start_index <= calculated_end_index:
                            original_refs.append([[trace_start_line, calculated_start_index], 
                                                  [trace_end_line, calculated_end_index]])
                        else:
                            original_refs.append([[trace_start_line, trace_start_index], 
                                                  [trace_end_line, trace_end_index]])
                else:
                    # directMatching이 아니거나 복잡한 refs 구조인 경우 (프론트엔드와 동일)
                    if trace_info.get('refs'):
                        original_refs.extend(trace_info['refs'])
            
            # 변환 실패 시 로깅 (디버깅용)
            if not original_refs and missing_keys:
                trace_map_sample_keys = list(trace_map.keys())[:10] if trace_map else []
                LoggingUtil.warning("TraceabilityGenerator", 
                    f"refs 변환 실패: ref=[{start_line}:{start_index}, {end_line}:{end_index}], "
                    f"missing_traceMap_keys={missing_keys}, "
                    f"traceMap_sample_keys={trace_map_sample_keys}, "
                    f"traceMap_total_keys={len(trace_map) if trace_map else 0}")
        
        # 프론트엔드와 동일하게: 변환이 실패하면 원본 refs 반환
        # (프론트엔드도 변환 실패 시 원본 refs 반환하지만, 이후 ESDialogerTraceUtil에서 다시 변환 시도)
        # 하지만 여기서는 변환 실패 시 원본 refs를 반환하여 프론트엔드에서 재시도할 수 있도록 함
        return original_refs if original_refs else refs
