from typing import Any, Dict, List, TypedDict
from datetime import datetime
import json
import re

from project_generator.utils.logging_util import LoggingUtil
from project_generator.utils.refs_trace_util import RefsTraceUtil
from project_generator.utils.llm_factory import create_chat_llm


class DDLExtractorState(TypedDict):
    inputs: Dict[str, Any]
    inference: str
    progress: int
    result: Dict[str, Any]
    error: str
    timestamp: str


class DDLExtractor:
    """
    LangGraph workflow for extracting DDL fields from CREATE TABLE statements.
    Identifies all column names and their positions in the DDL text.
    """

    def __init__(self):
        self.llm = create_chat_llm(
            temperature=0,
            streaming=False,
            model_kwargs={
                "response_format": {"type": "json_object"}
            }
        )

    def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            LoggingUtil.info("DDLExtractor", "DDL field extraction started")

            # 1. 입력 추출
            ddl_requirements = input_data.get('ddlRequirements', [])
            bounded_context_name = input_data.get('boundedContextName', '')

            if not ddl_requirements:
                LoggingUtil.info("DDLExtractor", "No DDL requirements provided")
                return {
                    'ddlFieldRefs': [],
                    'inference': 'No DDL requirements provided',
                    'progress': 100
                }

            # 2. DDL 텍스트 병합 및 라인 번호 맵 생성
            ddl_text = '\n'.join([req.get('text', '') for req in ddl_requirements])
            line_trace_map = self._build_line_trace_map(ddl_requirements)

            # 3. 라인 번호 추가
            line_numbered_ddl = self._add_line_numbers(ddl_text)

            # 4. 프롬프트 생성 및 LLM 호출
            prompt = self._build_prompt(line_numbered_ddl, bounded_context_name)
            response = self.llm.invoke(prompt)

            # 5. JSON 파싱
            response_text = response.content if hasattr(response, 'content') else str(response)
            result_data = json.loads(response_text)
            
            # LLM이 refs를 생성했는지 확인
            raw_ddl_field_refs = result_data.get('result', {}).get('ddlFieldRefs', [])
            total_fields = len(raw_ddl_field_refs)
            fields_with_refs = sum(1 for field in raw_ddl_field_refs if field.get('refs') and len(field.get('refs', [])) > 0)
            LoggingUtil.info("DDLExtractor", 
                f"Refs 생성 확인: {fields_with_refs}/{total_fields} 필드에 refs 생성됨")
            if fields_with_refs == 0 and total_fields > 0:
                LoggingUtil.warning("DDLExtractor", 
                    "⚠️ LLM이 refs를 생성하지 않았습니다! 프롬프트를 확인하거나 structured output을 고려하세요")

            # 6. refs 변환 (phrase → indexes → absolute refs)
            ddl_field_refs = self._convert_refs_to_indexes(
                raw_ddl_field_refs,
                ddl_text,
                line_numbered_ddl,
                line_trace_map
            )

            LoggingUtil.info(
                "DDLExtractor",
                f"DDL extraction completed: {len(ddl_field_refs)} fields extracted"
            )

            return {
                'ddlFieldRefs': ddl_field_refs,
                'inference': result_data.get('inference', ''),
                'progress': 100
            }

        except Exception as e:
            LoggingUtil.error("DDLExtractor", f"Failed: {str(e)}")
            return {
                'ddlFieldRefs': [],
                'inference': f'Error: {str(e)}',
                'progress': 100,
                'error': str(e)
            }

    def _build_line_trace_map(self, ddl_requirements: List[Dict]) -> Dict[int, int]:
        """DDL 요구사항별 라인 번호 맵 생성 (병합된 라인 → 원본 라인)"""
        line_trace_map = {}
        current_merged_line = 1

        for ddl_req in ddl_requirements:
            text = ddl_req.get('text', '')
            lines = text.split('\n')

            # refs에서 시작 라인 번호 추출
            refs = ddl_req.get('refs', [])
            if refs and len(refs) > 0 and len(refs[0]) > 0:
                original_start_line = refs[0][0][0]

                # 각 라인 매핑
                for i in range(len(lines)):
                    line_trace_map[current_merged_line + i] = original_start_line + i

                current_merged_line += len(lines)

        return line_trace_map

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

    def _build_prompt(self, line_numbered_ddl: str, bounded_context_name: str) -> List[Dict]:
        """프론트엔드와 동일한 프롬프트 생성"""

        system_prompt = """You are an Expert SQL Parsing Specialist.

Your goal is to accurately extract all field (column) names from provided Data Definition Language (DDL) text with precise position traceability, regardless of SQL dialect or formatting inconsistencies.

You are a highly specialized parsing engine designed to understand and deconstruct SQL DDL. You have been trained on a vast corpus of SQL schemas from various databases like MySQL, PostgreSQL, Oracle, and SQL Server. You can cut through complex formatting, comments, and varied syntax to reliably identify the fundamental column definitions that form the blueprint of a database table.

**Task:** Analyze the provided line-numbered DDL text and identify every valid field (column) name with its precise position references.

**Parsing Strategy:**
1. Focus on CREATE TABLE statements
2. Parse column definitions between parentheses ()
3. Extract only column names (ignore data types, constraints, default values, comments)
4. Consolidate all fields from all tables into one single list
5. Do not infer or create any names - only extract what is explicitly defined

**Filtering Rules:**
- Uniqueness: No duplicates - each field name appears only once
- Ignore Comments: Do not extract from SQL comments (-- or /* ... */)
- Field Name Language: Extract only English field names (column names themselves must be English)
- Valid Characters: Field names consist of letters, numbers, underscores (_), typically start with letter
- Ignore ENUM Values: Extract only the field name, not the values in ENUM(...)
- Handle Formatting: Be robust to inconsistent formatting and SQL dialects

**Traceability (refs):**
- Format: [[[startLineNumber, "minimal_start_phrase"], [endLineNumber, "minimal_end_phrase"]]]
- Use MINIMAL phrases (1-2 words) that uniquely identify the position in the DDL text
- Use the shortest possible phrase that can accurately locate the specific part of the DDL
- Must reference valid line numbers from the DDL text
- **CRITICAL:** startLineNumber and endLineNumber MUST be NUMBERS (not strings). Phrases MUST be NON-EMPTY strings (at least 1 character). Empty strings "" in phrases are STRICTLY FORBIDDEN.

**Precision and Accuracy Standards:**
- **Exact Segments:** Be precise in identifying the exact text segments that contain each field name
- **Avoid Vagueness:** Avoid generic or vague references that don't clearly point to the field location
- **Verification:** Ensure that the referenced text actually contains the field name
- **Comprehensive Mapping:** If a field appears in multiple places, reference all occurrences

**CRITICAL OUTPUT REQUIREMENTS:**
- You MUST output ONLY valid JSON, no explanations, no markdown code fences
- Do NOT include ```json or ``` markers
- Do NOT include any text before or after the JSON object

**Output Format:**
{
    "inference": "(Brief explanation of extraction process)",
    "result": {
        "ddlFieldRefs": [
            {
                "fieldName": "(ColumnName)",
                "refs": [[[startLine, "phrase"], [endLine, "phrase"]]]
            }
        ]
    }
}

**Requirements:**
- All field names must match exactly as they appear in the DDL
- Each field must include valid refs pointing to its location
- Return empty ddlFieldRefs array if no valid fields are found
- Output ONLY the JSON object, nothing else"""

        user_prompt = f"""**Line-Numbered DDL:**
{line_numbered_ddl}

**Bounded Context:** {bounded_context_name}

Please extract all DDL field names with their position references."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def _convert_refs_to_indexes(
        self,
        ddl_field_refs: List[Dict],
        raw_ddl: str,
        line_numbered_ddl: str,
        line_trace_map: Dict[int, int]
    ) -> List[Dict]:
        """refs를 phrase → indexes → absolute refs로 변환"""

        converted_fields = []

        for field_ref in ddl_field_refs:
            field_name = field_ref.get('fieldName', '')
            refs = field_ref.get('refs', [])

            if not field_name:
                continue
            
            # refs가 없어도 필드는 포함 (빈 배열로 설정)
            if not refs:
                LoggingUtil.warning("DDLExtractor", 
                    f"Field '{field_name}' has no refs, will be included with empty refs array")
                converted_fields.append({
                    'fieldName': field_name,
                    'refs': []
                })
                continue

            # 1. sanitizeAndConvertRefs: phrase → [[[line, col], [line, col]]]
            # 공통 유틸리티 사용 (프론트엔드와 동일)
            sanitized_data = RefsTraceUtil.sanitize_and_convert_refs(
                {'refs': refs},
                line_numbered_ddl,
                is_use_xml_base=True
            )
            sanitized_refs = sanitized_data.get('refs', refs) if isinstance(sanitized_data, dict) else sanitized_data

            # 2. absolute refs로 변환
            absolute_refs = self._convert_to_absolute_refs(
                sanitized_refs,
                line_trace_map
            )

            converted_fields.append({
                'fieldName': field_name,
                'refs': absolute_refs
            })

        return converted_fields

    def _sanitize_and_convert_refs(self, refs: List, line_numbered_ddl: str) -> List:
        """Frontend RefsTraceUtil.sanitizeAndConvertRefs와 동일"""
        lines = line_numbered_ddl.split('\n')

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

    def _convert_to_absolute_refs(
        self,
        sanitized_refs: List,
        line_trace_map: Dict[int, int]
    ) -> List:
        """병합된 라인 번호를 원본 절대 라인 번호로 변환"""

        absolute_refs = []

        for ref in sanitized_refs:
            if not isinstance(ref, list) or len(ref) < 2:
                continue

            start_line = ref[0][0] if isinstance(ref[0], list) and len(ref[0]) > 0 else 1
            start_col = ref[0][1] if isinstance(ref[0], list) and len(ref[0]) > 1 else 0
            end_line = ref[1][0] if isinstance(ref[1], list) and len(ref[1]) > 0 else 1
            end_col = ref[1][1] if isinstance(ref[1], list) and len(ref[1]) > 1 else 0

            # line_trace_map으로 절대 라인 번호로 변환
            abs_start_line = line_trace_map.get(start_line, start_line)
            abs_end_line = line_trace_map.get(end_line, end_line)

            absolute_refs.append([[abs_start_line, start_col], [abs_end_line, end_col]])

        return absolute_refs

