"""
Preview Fields Generator - LangGraph Backend
애그리거트 초안에 대한 필수 속성(Preview Fields)을 생성하는 LangGraph 워크플로우
"""
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import json
from datetime import datetime
from project_generator.config import Config
from project_generator.utils.logging_util import LoggingUtil
from project_generator.utils.llm_factory import create_chat_llm


# ==================== Pydantic Models ====================

class PreviewField(BaseModel):
    """개별 필드"""
    fieldName: str = Field(..., description="The field name (e.g., user_id, title)")
    refs: List[List[List[Any]]] = Field(..., description="Traceability references in format [[[startLineNumber, 'start_phrase'], [endLineNumber, 'end_phrase']]]")


class AggregateFieldAssignment(BaseModel):
    """애그리거트별 필드 할당"""
    aggregateName: str = Field(..., description="The name of the aggregate")
    previewFields: List[PreviewField] = Field(..., description="List of fields assigned to this aggregate")


class PreviewFieldsOutput(BaseModel):
    """최종 출력 형식"""
    inference: str = Field(..., description="Detailed reasoning for field generation")
    result: Dict[str, List[AggregateFieldAssignment]] = Field(
        ..., 
        description="Result containing aggregateFieldAssignments"
    )


# ==================== State ====================

class PreviewFieldsState(TypedDict):
    """Preview Fields Generator의 상태"""
    # Input
    description: str
    aggregateDrafts: List[Dict[str, Any]]
    generatorKey: str
    traceMap: Dict[str, Any]
    originalRequirements: str  # 원본 요구사항 (userStory + ddl)
    
    # Processing
    lineNumberedRequirements: str
    prompt: str
    
    # Output
    inference: str
    aggregateFieldAssignments: List[Dict[str, Any]]
    
    # Metadata
    logs: List[Dict[str, str]]
    progress: int
    isCompleted: bool
    isFailed: bool


# ==================== Preview Fields Generator ====================

class PreviewFieldsGenerator:
    """Preview Fields 생성 워크플로우"""
    
    def __init__(self, model_name: str = None, temperature: float = 0.7):
        self.model_name = model_name or Config.DEFAULT_LLM_MODEL
        self.temperature = temperature
        self.llm = None
        
    def _add_line_numbers(self, text: str) -> str:
        """텍스트에 줄 번호 추가"""
        lines = text.split('\n')
        numbered_lines = []
        for i, line in enumerate(lines, 1):
            numbered_lines.append(f"<{i}>{line}</{i}>")
        return '\n'.join(numbered_lines)
    
    def _get_line_number_range(self, line_numbered_requirements: str) -> tuple:
        """라인 번호 범위 계산 (프론트엔드 TextTraceUtil.getLineNumberRangeOfRequirements와 동일)"""
        import re
        
        if not line_numbered_requirements:
            return (1, 1)
        
        lines = line_numbered_requirements.strip().split('\n')
        if not lines:
            return (1, 1)
        
        # 첫 번째 라인에서 라인 번호 추출
        first_line = lines[0]
        first_match = re.match(r'^<(\d+)>', first_line)
        min_line = int(first_match.group(1)) if first_match else 1
        
        # 마지막 라인에서 라인 번호 추출
        last_line = lines[-1]
        last_match = re.match(r'^<(\d+)>', last_line)
        max_line = int(last_match.group(1)) if last_match else len(lines)
        
        return (min_line, max_line)
    
    def _get_line_number_validation_prompt(self, line_numbered_requirements: str) -> str:
        """라인 번호 검증 프롬프트 생성 (프론트엔드 TextTraceUtil.getLineNumberValidationPrompt와 동일)"""
        if not line_numbered_requirements:
            return ''
        
        min_line, max_line = self._get_line_number_range(line_numbered_requirements)
        return f"""
IMPORTANT - LINE NUMBER VALIDATION RULE:
When creating 'refs' arrays in your response, you MUST only use line numbers that exist in the Requirements.
Valid line number range: {min_line} ~ {max_line}
"""
    
    def _build_persona_info(self) -> Dict[str, str]:
        """Persona 정보 생성"""
        return {
            "persona": "Domain-Driven Design (DDD) Field Generation Specialist",
            "goal": "To analyze functional requirements and aggregate draft structures within a bounded context, then intelligently generate appropriate field names for each aggregate based on domain semantics and business logic.",
            "backstory": "With extensive experience in domain modeling and database design, I specialize in translating business requirements into concrete data structures."
        }
    
    def _build_guidelines(self) -> str:
        """가이드라인 생성 (프론트엔드와 동일)"""
        return """<instruction>
    <core_instructions>
        <title>Aggregate Field Generation Task</title>
        <task_description>Your task is to analyze functional requirements and generate appropriate field names with traceability information for each aggregate draft in a bounded context. Generate comprehensive and semantically correct field sets that accurately represent the data each aggregate should contain according to the business requirements.</task_description>
        
        <input_description>
            <title>You will be given:</title>
            <item id="1">**Functional Requirements:** The business context and domain description for the bounded context with line numbers for traceability</item>
            <item id="2">**Aggregate Drafts:** A list of planned aggregates with their names and basic structure information</item>
        </input_description>

        <guidelines>
            <title>Field Generation Guidelines</title>
            
            <section id="field_generation_rules">
                <title>Field Generation Rules</title>
                <rule id="1">**Business-Driven Generation:** Generate fields that directly support the business operations described in the functional requirements</rule>
                <rule id="2">**Complete Coverage:** Each aggregate should have a comprehensive set of fields that covers its full responsibility scope</rule>
                <rule id="3">**Avoid Duplication:** Each field concept should appear in only one aggregate unless there's a clear business need for duplication</rule>
                <rule id="4">**Domain Semantics:** Field names should reflect the business language and concepts used in the domain</rule>
                <rule id="5">**Consistent Naming:** Use consistent naming patterns across all aggregates (e.g., snake_case, clear prefixes/suffixes)</rule>
            </section>

            <section id="field_categories">
                <title>Standard Field Categories</title>
                <rule id="1">**Identity Fields:** Each aggregate should have a primary identifier (e.g., "user_id", "order_id")</rule>
                <rule id="2">**Core Business Fields:** Fields that represent the main business data for the aggregate</rule>
                <rule id="3">**Lifecycle Fields:** Standard fields like "created_at", "updated_at", "status" where relevant</rule>
                <rule id="4">**Relationship Fields:** Foreign key references to other aggregates when relationships exist</rule>
                <rule id="5">**Metadata Fields:** Additional fields for versioning, auditing, or technical requirements when needed</rule>
            </section>

            <section id="domain_context">
                <title>Domain Context Considerations</title>
                <rule id="1">**Business Operations:** Consider what operations will be performed on each aggregate and what data they require</rule>
                <rule id="2">**State Management:** Include fields needed to track the aggregate's state through its lifecycle</rule>
                <rule id="3">**Business Rules:** Generate fields that support the business rules and constraints mentioned in the requirements</rule>
                <rule id="4">**User Interactions:** Consider what data users will need to view, create, or modify for each aggregate</rule>
            </section>

            <section id="inference_process">
                <title>Inference Process</title>
                <rule id="1">**Domain Analysis:** Begin by understanding the business domain and identifying the core responsibilities of each aggregate</rule>
                <rule id="2">**Extract Data Requirements:** From the functional requirements, identify what data is needed to support the described business operations</rule>
                <rule id="3">**Map Data to Aggregates:** Assign data concepts to the most appropriate aggregate based on business ownership and responsibility</rule>
                <rule id="4">**Apply Domain Language:** Use terminology and naming conventions that match the business domain vocabulary</rule>
                <rule id="5">**Validate Business Logic:** Ensure the generated fields support the business operations and rules described in the requirements</rule>
                <rule id="6">**Document Reasoning:** Clearly explain your field generation decisions, especially for complex or ambiguous cases</rule>
            </section>

            <section id="traceability">
                <title>Source Traceability Requirements</title>
                <rule id="1">**Mandatory Refs:** Each field MUST include a 'refs' array that traces back to specific parts of the functional requirements</rule>
                <rule id="2">**Refs Format:** Use format [[[startLineNumber, "minimal_start_phrase"], [endLineNumber, "minimal_end_phrase"]]]</rule>
                <rule id="3">**Minimal Phrases:** Use 1-2 word phrases that uniquely identify the position in the requirement text</rule>
                <rule id="4">**Valid Line Numbers:** Refs must reference valid line numbers from the provided functional requirements section</rule>
                <rule id="5">**Multiple References:** Multiple reference ranges can be included if a field is derived from multiple requirement sections</rule>
                <rule id="6">**Complete Traceability:** Every generated field must have at least one traceability reference</rule>
            </section>

            <section id="quality">
                <title>Quality Guidelines</title>
                <rule id="1">**Appropriate Granularity:** Generate fields at the right level of detail - not too generic, not overly specific</rule>
                <rule id="2">**Future-Friendly:** Consider reasonable extensions and modifications that might be needed</rule>
                <rule id="3">**Precision and Accuracy:** Be precise in identifying the exact text segments that justify each field</rule>
            </section>
        </guidelines>

        <refs_format_example>
            <title>Example of refs Format for Field Traceability</title>
            <description>If functional requirements contain:</description>
            <example_requirements>
<1># Course Management System</1>
<2></2> 
<3>As an instructor, I want to create and manage my courses. When creating a course, I need to provide a title, description, and price.</3>
<4>Students can enroll in published courses.</4>
<5></5>
<6>## Key Events</6>
<7>- CourseCreated</7>
<8>- CoursePublished</8>
<9>- StudentEnrolled</9>
            </example_requirements>
            <example_refs>
- "course_id" field for Course aggregate → refs: [[[3, "create"], [3, "courses"]]]
- "title" field based on course creation → refs: [[[3, "title"], [3, "price"]]]
- "status" field for course lifecycle → refs: [[[7, "CourseCreated"], [8, "CoursePublished"]]]
- "student_id" field for enrollment → refs: [[[4, "Students"], [9, "StudentEnrolled"]]]
            </example_refs>
            <note>The refs array contains ranges where each range is [[startLine, startPhrase], [endLine, endPhrase]]. Use the shortest possible phrase that can locate the specific part of requirements.</note>
        </refs_format_example>
    </core_instructions>
    
    <output_format>
        <title>JSON Output Format</title>
        <description>The output must be a JSON object structured as follows:</description>
        <schema>
{
    "inference": "(Detailed reasoning for the field generation, including analysis of the domain context and explanation of field choices for each aggregate)",
    "result": {
        "aggregateFieldAssignments": [
            {
                "aggregateName": "(name_of_aggregate)",
                "previewFields": [
                    {
                        "fieldName": "(field_name_1)",
                        "refs": [[[startLineNumber, "minimal start phrase"], [endLineNumber, "minimal end phrase"]]]
                    },
                    {
                        "fieldName": "(field_name_2)",
                        "refs": [[[startLineNumber, "minimal start phrase"], [endLineNumber, "minimal end phrase"]]]
                    }
                ]
            }
        ]
    }
}
        </schema>
        <field_requirements>
            <requirement id="1">All field names must use consistent naming patterns (e.g., snake_case)</requirement>
            <requirement id="2">Every field must include a refs array with valid line number references</requirement>
            <requirement id="3">Aggregate names must match the names provided in the input Aggregate Drafts</requirement>
        </field_requirements>
    </output_format>
</instruction>"""
    
    def _build_prompt(self, state: PreviewFieldsState) -> str:
        """프롬프트 생성"""
        persona = self._build_persona_info()
        guidelines = self._build_guidelines()
        
        aggregates_xml = self._format_aggregates(state['aggregateDrafts'])
        
        # 라인 번호 검증 프롬프트 추가 (프론트엔드와 동일하게)
        line_number_validation = self._get_line_number_validation_prompt(state['lineNumberedRequirements'])
        
        prompt = f"""You are a {persona['persona']}.

{persona['goal']}

{guidelines}

{line_number_validation}

## Your Task:
Analyze the following functional requirements and generate appropriate field names with traceability for each aggregate draft.

### Functional Requirements:
{state['lineNumberedRequirements']}

### Aggregate Drafts:
{aggregates_xml}

### Generator Key: {state.get('generatorKey', 'N/A')}

Please generate comprehensive field sets for each aggregate with proper traceability references.
For each field in `previewFields`, include both `fieldName` (English name) and `fieldAlias` (Korean name/alias)."""
        
        return prompt
    
    def _format_aggregates(self, aggregates: List[Dict[str, Any]]) -> str:
        """Aggregate 리스트를 XML 형식으로 변환"""
        xml_parts = []
        for agg in aggregates:
            xml_parts.append(f"  <aggregate>")
            xml_parts.append(f"    <name>{agg.get('name', 'Unknown')}</name>")
            if 'alias' in agg:
                xml_parts.append(f"    <alias>{agg['alias']}</alias>")
            xml_parts.append(f"  </aggregate>")
        return "<aggregates>\n" + "\n".join(xml_parts) + "\n</aggregates>"
    
    def _get_response_schema(self) -> Dict[str, Any]:
        """Structured Output 스키마 정의"""
        return {
            "type": "object",
            "title": "PreviewFieldsOutput",
            "description": "Output format for preview fields generation",
            "properties": {
                "inference": {
                    "type": "string",
                    "description": "Detailed reasoning for the field generation"
                },
                "result": {
                    "type": "object",
                    "properties": {
                        "aggregateFieldAssignments": {
                            "type": "array",
                            "description": "Field assignments for each aggregate",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "aggregateName": {
                                        "type": "string",
                                        "description": "The name of the aggregate"
                                    },
                                    "previewFields": {
                                        "type": "array",
                                        "description": "List of fields for this aggregate",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "fieldName": {
                                                    "type": "string",
                                                    "description": "The field name (English name)"
                                                },
                                                "fieldAlias": {
                                                    "type": "string",
                                                    "description": "The field alias (Korean name)"
                                                },
                                                "refs": {
                                                    "type": "array",
                                                    "description": "Traceability references",
                                                    "minItems": 1,
                                                    "items": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "array",
                                                            "items": {
                                                                "anyOf": [
                                                                    {"type": "number"},
                                                                    {"type": "string"}
                                                                ]
                                                            }
                                                        }
                                                    }
                                                }
                                            },
                                            "required": ["fieldName", "fieldAlias", "refs"],
                                            "additionalProperties": False
                                        }
                                    }
                                },
                                "required": ["aggregateName", "previewFields"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["aggregateFieldAssignments"],
                    "additionalProperties": False
                }
            },
            "required": ["inference", "result"],
            "additionalProperties": False
        }
    
    # ==================== Workflow Nodes ====================
    
    def prepare_input(self, state: PreviewFieldsState) -> PreviewFieldsState:
        """입력 데이터 준비"""
        # Add line numbers to requirements
        description = state.get('description', '')
        state['lineNumberedRequirements'] = self._add_line_numbers(description)
        
        # Build prompt
        state['prompt'] = self._build_prompt(state)
        
        state['logs'] = state.get('logs', [])
        state['logs'].append({
            "message": "Input data prepared",
            "timestamp": datetime.now().isoformat()
        })
        state['progress'] = 10
        
        return state
    
    def generate_fields(self, state: PreviewFieldsState) -> PreviewFieldsState:
        """LLM을 사용하여 Preview Fields 생성"""
        aggregates = state.get('aggregateDrafts', [])
        
        try:
            # LLM 초기화
            if not self.llm:
                self.llm = create_chat_llm(
                    model=self.model_name,
                    temperature=self.temperature
                )
            
            # Structured Output 설정
            structured_llm = self.llm.with_structured_output(
                self._get_response_schema(),
                method="json_schema",
                strict=True
            )
            
            # LLM 호출
            response = structured_llm.invoke(state['prompt'])
            
            # 응답 처리
            state['inference'] = response.get('inference', '')
            assignments = response.get('result', {}).get('aggregateFieldAssignments', [])
            state['aggregateFieldAssignments'] = assignments
            
            # 생성된 결과 로깅 및 refs 확인
            # LLM이 refs를 생성했는지 확인
            # Refs 생성 확인
            total_fields = 0
            fields_with_refs = 0
            for assignment in assignments:
                for field in assignment.get('previewFields', []):
                    total_fields += 1
                    if field.get('refs') and len(field.get('refs', [])) > 0:
                        fields_with_refs += 1
            if fields_with_refs == 0 and total_fields > 0:
                LoggingUtil.warning("PreviewFieldsGenerator", 
                    "⚠️ LLM이 refs를 생성하지 않았습니다! 스키마에 minItems: 1이 적용되었는지 확인 필요")
            
            # 누락된 aggregate 검사
            input_aggregate_names = set(agg.get('name') for agg in aggregates)
            output_aggregate_names = set(a.get('aggregateName') for a in assignments)
            missing_aggregates = input_aggregate_names - output_aggregate_names
            
            if missing_aggregates:
                LoggingUtil.error(
                    "PreviewFieldsGenerator",
                    f"⚠️ Missing aggregates in output: {list(missing_aggregates)}"
                )
            
            state['logs'].append({
                "message": f"Generated fields for {len(state['aggregateFieldAssignments'])} aggregates",
                "timestamp": datetime.now().isoformat()
            })
            state['progress'] = 90
            
        except Exception as e:
            LoggingUtil.error("PreviewFieldsGenerator", f"Error generating fields: {str(e)}")
            state['isFailed'] = True
            state['logs'].append({
                "message": f"Error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
        
        return state
    
    def finalize_output(self, state: PreviewFieldsState) -> PreviewFieldsState:
        """최종 출력 정리"""
        
        # refs 변환: phrase → index → 원본 라인
        # 프론트엔드와 동일: sanitize/validate는 description 기준, convertToOriginalRefsUsingTraceMap은 traceMap 사용
        description = state.get('description', '')  # BC description (sanitize/validate용)
        original_requirements = state.get('originalRequirements', '')  # 원본 요구사항 (최종 검증용)
        line_numbered_requirements = state.get('lineNumberedRequirements', '')
        trace_map = state.get('traceMap', {})
        
        if state.get('aggregateFieldAssignments'):
            self._convert_refs_to_indexes(
                state['aggregateFieldAssignments'],
                description,  # sanitize/validate는 description 기준 (프론트엔드와 동일)
                line_numbered_requirements,
                trace_map,
                original_requirements  # 최종 검증용
            )
        
        # 검증: 각 aggregate가 최소한의 필드를 가지고 있는지 확인
        # LLM이 특정 aggregate에 필드를 생성하지 않은 경우 진단을 위한 상세 로깅
        input_aggregate_names = set(agg.get('name') for agg in state.get('aggregateDrafts', []))
        output_aggregate_names = set()
        
        for assignment in state.get('aggregateFieldAssignments', []):
            aggregate_name = assignment.get('aggregateName', 'Unknown')
            output_aggregate_names.add(aggregate_name)
            preview_fields = assignment.get('previewFields', [])
            
            if not preview_fields or len(preview_fields) == 0:
                # 상세 진단 정보 로깅
                LoggingUtil.warning("PreviewFieldsGenerator", 
                    f"Aggregate '{aggregate_name}' has no generated fields. "
                    f"Input aggregates: {sorted(input_aggregate_names)}, "
                    f"Output aggregates: {sorted(output_aggregate_names)}, "
                    f"Total assignments: {len(state.get('aggregateFieldAssignments', []))}. "
                    f"프로세스를 계속 진행합니다.")
                
                # LLM 응답 원본 확인
                if state.get('inference'):
                    LoggingUtil.debug("PreviewFieldsGenerator", 
                        f"Inference for '{aggregate_name}': {state['inference'][:500]}...")
                
                # 경고만 로깅하고 프로세스 계속 진행 (프론트엔드에서 처리)
                state['logs'].append({
                    "message": f"Aggregate '{aggregate_name}' has no generated fields (경고)",
                    "timestamp": datetime.now().isoformat(),
                    "level": "warning"
                })
                # 프로세스 중단하지 않고 계속 진행
                continue
            
            # 각 필드의 refs 검증
            for field in assignment.get('previewFields', []):
                if not field.get('refs') or len(field.get('refs', [])) == 0:
                    LoggingUtil.warning("PreviewFieldsGenerator", 
                        f"Field '{field.get('fieldName', 'unknown')}' in aggregate '{assignment.get('aggregateName', 'unknown')}' has empty refs")
        
        state['isCompleted'] = True
        state['progress'] = 100
        state['logs'].append({
            "message": "Preview fields generation completed",
            "timestamp": datetime.now().isoformat()
        })
        
        return state
    
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
            for index, item in enumerate(trace_map):
                if not item:
                    continue
                # {"key":"4","value":{...}} 형태인 경우
                if isinstance(item, dict) and 'key' in item and 'value' in item:
                    key = item['key']
                    restored_trace_map[str(key)] = item['value']
                    # 숫자로 변환 가능하면 숫자 키로도 저장 (양쪽 모두 접근 가능)
                    try:
                        num_key = int(key)
                        if str(num_key) == str(key):
                            restored_trace_map[num_key] = item['value']
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
        
        return restored_trace_map
    
    def _convert_refs_to_indexes(
        self,
        aggregate_field_assignments: List[Dict],
        description: str,  # BC description (sanitize/validate용)
        line_numbered_requirements: str,
        trace_map: Dict,
        original_requirements: str = ''  # 원본 요구사항 (최종 검증용, 선택적)
    ) -> None:
        """refs를 phrase → indexes로 변환 (프론트엔드 RefsTraceUtil과 동일)"""
        from project_generator.workflows.aggregate_draft.traceability_generator import TraceabilityGenerator
        
        # 디버깅: 입력 상태 확인
        LoggingUtil.info("PreviewFieldsGenerator", 
            f"🔍 [변환 시작] description 길이={len(description) if description else 0}, "
            f"line_numbered_requirements 길이={len(line_numbered_requirements) if line_numbered_requirements else 0}, "
            f"trace_map keys={len(trace_map) if isinstance(trace_map, dict) else 0 if trace_map else 0}")
        
        if not description:
            LoggingUtil.warning("PreviewFieldsGenerator", 
                "⚠️ description이 비어있습니다!")
        
        if not line_numbered_requirements:
            LoggingUtil.error("PreviewFieldsGenerator", 
                "❌ line_numbered_requirements가 비어있습니다! 변환 불가능합니다.")
            return
        
        # traceMap 복원 (Firebase가 배열로 변환한 경우 처리)
        restored_trace_map = self._restore_trace_map(trace_map)
        
        # TraceabilityGenerator의 변환 메서드 재사용
        temp_generator = TraceabilityGenerator()
        
        total_fields = 0
        converted_fields = 0
        failed_fields = 0
        sanitize_failed = 0
        validate_failed = 0
        trace_map_failed = 0
        
        from project_generator.utils.refs_trace_util import RefsTraceUtil
        
        def convert_field_refs(field, field_type='previewFields'):
            """단일 필드의 refs 변환 (previewFields와 previewAttributes 공통)"""
            if 'refs' not in field or not field['refs']:
                return False
            
            nonlocal total_fields, converted_fields, failed_fields, sanitize_failed, validate_failed, trace_map_failed
            total_fields += 1
            original_refs = field['refs']
            
            # 첫 번째 필드만 상세 로깅
            is_first = total_fields == 1
            
            try:
                # 1. sanitizeAndConvertRefs: phrase → [[[line, col], [line, col]]]
                if is_first:
                    LoggingUtil.info("PreviewFieldsGenerator", 
                        f"🔍 [sanitize 시도] field='{field.get('fieldName', 'unknown')}', "
                        f"original_refs={original_refs[:1] if original_refs else []}")
                
                sanitized_data = RefsTraceUtil.sanitize_and_convert_refs(
                    {'refs': original_refs},
                    line_numbered_requirements,
                    is_use_xml_base=True
                )
                sanitized_refs = sanitized_data.get('refs', original_refs) if isinstance(sanitized_data, dict) else sanitized_data
                
                if not sanitized_refs:
                    if is_first:
                        LoggingUtil.warning("PreviewFieldsGenerator", 
                            f"❌ [sanitize 실패] field='{field.get('fieldName', 'unknown')}', "
                            f"original_refs={original_refs[:1] if original_refs else []}, "
                            f"sanitized_data={sanitized_data}")
                    sanitize_failed += 1
                    failed_fields += 1
                    return False
                
                if is_first:
                    LoggingUtil.info("PreviewFieldsGenerator", 
                        f"✅ [sanitize 성공] field='{field.get('fieldName', 'unknown')}', "
                        f"sanitized_refs={sanitized_refs[:1] if sanitized_refs else []}")
                
                field['refs'] = sanitized_refs
                
                # 2. validateRefs: 범위 검증 (description 기준, 프론트엔드와 동일)
                try:
                    temp_generator._validate_refs(field['refs'], description)
                except Exception as e:
                    if is_first:
                        LoggingUtil.warning("PreviewFieldsGenerator", 
                            f"❌ [validate 실패] field='{field.get('fieldName', 'unknown')}', "
                            f"error={str(e)}, sanitized_refs={sanitized_refs[:1] if sanitized_refs else []}")
                    validate_failed += 1
                    # 원본 refs로 복원
                    field['refs'] = original_refs
                    failed_fields += 1
                    return False
                
                # 3. convertToOriginalRefsUsingTraceMap: traceMap 사용해 원본 라인으로 역변환
                if restored_trace_map:
                    converted_refs = RefsTraceUtil.convert_to_original_refs_using_trace_map(
                        field['refs'],
                        restored_trace_map
                    )
                    
                    if not converted_refs:
                        if is_first:
                            LoggingUtil.warning("PreviewFieldsGenerator", 
                                f"❌ [traceMap 변환 실패] field='{field.get('fieldName', 'unknown')}', "
                                f"sanitized_refs={sanitized_refs[:1] if sanitized_refs else []}")
                        trace_map_failed += 1
                        # 변환 실패 시 sanitized_refs 유지
                        field['refs'] = sanitized_refs
                        failed_fields += 1
                        return False
                    else:
                        if is_first:
                            LoggingUtil.info("PreviewFieldsGenerator", 
                                f"✅ [변환 완료] field='{field.get('fieldName', 'unknown')}', "
                                f"converted_refs={converted_refs[:1] if converted_refs else []}")
                        field['refs'] = converted_refs
                        converted_fields += 1
                        return True
                else:
                    # trace_map이 없으면 sanitized_refs 그대로 사용
                    converted_fields += 1
                    return True
                    
            except Exception as e:
                if is_first:
                    LoggingUtil.error("PreviewFieldsGenerator", 
                        f"❌ [예외 발생] field='{field.get('fieldName', 'unknown')}', "
                        f"error={str(e)}")
                # 원본 refs 유지
                field['refs'] = original_refs
                failed_fields += 1
                return False
        
        for assignment in aggregate_field_assignments:
            # previewFields 변환
            for field in assignment.get('previewFields', []):
                convert_field_refs(field, 'previewFields')
            
            # previewAttributes 변환 (previewFields와 동일한 구조)
            for field in assignment.get('previewAttributes', []):
                convert_field_refs(field, 'previewAttributes')
        
        if failed_fields > 0:
            LoggingUtil.warning("PreviewFieldsGenerator", 
                f"Refs 변환: {converted_fields}/{total_fields} 성공, {failed_fields} 실패 "
                f"(sanitize: {sanitize_failed}, validate: {validate_failed}, traceMap: {trace_map_failed})")
    
    # ==================== Workflow Construction ====================
    
    def create_workflow(self) -> StateGraph:
        """워크플로우 생성"""
        workflow = StateGraph(PreviewFieldsState)
        
        # 노드 추가
        workflow.add_node("prepare_input", self.prepare_input)
        workflow.add_node("generate_fields", self.generate_fields)
        workflow.add_node("finalize_output", self.finalize_output)
        
        # 엣지 추가
        workflow.set_entry_point("prepare_input")
        workflow.add_edge("prepare_input", "generate_fields")
        workflow.add_edge("generate_fields", "finalize_output")
        workflow.add_edge("finalize_output", END)
        
        return workflow.compile()
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """워크플로우 실행"""
        
        # 초기 상태 설정 (프론트엔드 에이전트 방식과 동일하게 description만 사용)
        initial_state: PreviewFieldsState = {
            'description': input_data.get('description', ''),
            'aggregateDrafts': input_data.get('aggregateDrafts', []),
            'generatorKey': input_data.get('generatorKey', 'default'),
            'traceMap': input_data.get('traceMap', {}),
            'originalRequirements': input_data.get('originalRequirements', ''),  # 원본 요구사항 (userStory + ddl)
            'lineNumberedRequirements': '',
            'prompt': '',
            'inference': '',
            'aggregateFieldAssignments': [],
            'logs': [],
            'progress': 0,
            'isCompleted': False,
            'isFailed': False
        }
        
        # 워크플로우 실행
        workflow = self.create_workflow()
        final_state = workflow.invoke(initial_state)
        
        # 결과 반환
        return {
            'inference': final_state.get('inference', ''),
            'aggregateFieldAssignments': final_state.get('aggregateFieldAssignments', []),
            'logs': final_state.get('logs', []),
            'progress': final_state.get('progress', 0),
            'isCompleted': final_state.get('isCompleted', False),
            'isFailed': final_state.get('isFailed', False)
        }

