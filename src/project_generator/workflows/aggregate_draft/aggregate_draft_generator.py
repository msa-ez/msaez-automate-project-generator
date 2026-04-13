"""
Aggregate Draft Generator Backend
요구사항 및 BC 정보를 기반으로 여러 Aggregate 초안 옵션을 생성
"""
from typing import Dict, TypedDict, List
from datetime import datetime
import json
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.output_parsers import JsonOutputParser
from project_generator.utils.logging_util import LoggingUtil
from project_generator.utils.llm_factory import create_chat_llm
from project_generator.systems.storage_system_factory import StorageSystemFactory
from project_generator.utils.refs_trace_util import RefsTraceUtil
from langgraph.graph import StateGraph, END

class AggregateDraftState(TypedDict):
    # Inputs
    bounded_context: Dict
    description: str
    accumulated_drafts: Dict
    analysis_result: Dict  # Optional
    
    # Working state
    requirements_text: str
    context_relations_text: str
    
    # Output
    inference: str
    options: List[Dict]
    default_option_index: int
    conclusions: str
    
    # Progress tracking
    progress: int
    logs: List[Dict]
    is_completed: bool
    error: str


class AggregateDraftGenerator:
    """Aggregate 초안 생성기"""
    
    def __init__(self):
        """초기화"""
        self.llm = create_chat_llm(
            model="gpt-4.1-2025-04-14",
            temperature=0.3,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        self.llm_structured = self.llm.with_structured_output(
            self._get_response_schema(),
            strict=True
        )
        
        self.workflow = self._build_workflow()
    
    def _get_response_schema(self) -> Dict:
        """응답 스키마 정의 (프론트엔드와 동일)"""
        return {
            "title": "AggregateDraftResponse",
            "description": "Response schema for aggregate draft generation with multiple design options",
            "type": "object",
            "properties": {
                "inference": {
                    "type": "string",
                    "description": "Detailed reasoning for aggregate design including analysis of business requirements, events, and context relationships"
                },
                "result": {
                    "type": "object",
                    "properties": {
                        "options": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "structure": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "aggregate": {
                                                    "type": "object",
                                                    "properties": {
                                                        "name": {"type": "string"},
                                                        "alias": {"type": "string"}
                                                    },
                                                    "required": ["name", "alias"],
                                                    "additionalProperties": False
                                                },
                                                "enumerations": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "name": {"type": "string"},
                                                            "alias": {"type": "string"}
                                                        },
                                                        "required": ["name", "alias"],
                                                        "additionalProperties": False
                                                    }
                                                },
                                                "valueObjects": {
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "name": {"type": "string"},
                                                            "alias": {"type": "string"},
                                                            "referencedAggregateName": {"type": "string"}
                                                        },
                                                        "required": ["name", "alias", "referencedAggregateName"],
                                                        "additionalProperties": False
                                                    }
                                                }
                                            },
                                            "required": ["aggregate", "enumerations", "valueObjects"],
                                            "additionalProperties": False
                                        }
                                    },
                                    "pros": {
                                        "type": "object",
                                        "properties": {
                                            "cohesion": {"type": "string"},
                                            "coupling": {"type": "string"},
                                            "consistency": {"type": "string"},
                                            "encapsulation": {"type": "string"},
                                            "complexity": {"type": "string"},
                                            "independence": {"type": "string"},
                                            "performance": {"type": "string"}
                                        },
                                        "required": ["cohesion", "coupling", "consistency", "encapsulation", "complexity", "independence", "performance"],
                                        "additionalProperties": False
                                    },
                                    "cons": {
                                        "type": "object",
                                        "properties": {
                                            "cohesion": {"type": "string"},
                                            "coupling": {"type": "string"},
                                            "consistency": {"type": "string"},
                                            "encapsulation": {"type": "string"},
                                            "complexity": {"type": "string"},
                                            "independence": {"type": "string"},
                                            "performance": {"type": "string"}
                                        },
                                        "required": ["cohesion", "coupling", "consistency", "encapsulation", "complexity", "independence", "performance"],
                                        "additionalProperties": False
                                    }
                                },
                                "required": ["structure", "pros", "cons"],
                                "additionalProperties": False
                            }
                        },
                        "defaultOptionIndex": {
                            "type": "number",
                            "description": "The index of the recommended option (starts from 1)"
                        },
                        "conclusions": {
                            "type": "string",
                            "description": "Conclusion explaining when each option would be best to choose"
                        }
                    },
                    "required": ["options", "defaultOptionIndex", "conclusions"],
                    "additionalProperties": False
                }
            },
            "required": ["inference", "result"],
            "additionalProperties": False
        }
    
    def _build_workflow(self) -> StateGraph:
        """워크플로우 구성"""
        workflow = StateGraph(AggregateDraftState)
        
        # 노드 추가
        workflow.add_node("prepare_inputs", self._prepare_inputs)
        workflow.add_node("generate_draft", self._generate_draft)
        workflow.add_node("finalize", self._finalize)
        
        # 엣지 추가
        workflow.set_entry_point("prepare_inputs")
        workflow.add_edge("prepare_inputs", "generate_draft")
        workflow.add_edge("generate_draft", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _prepare_inputs(self, state: AggregateDraftState) -> Dict:
        """입력 데이터 준비"""
        LoggingUtil.info("AggregateDraftGenerator", "Preparing inputs for aggregate draft generation")
        
        # BC 정보에서 requirements 추출
        bc = state.get('bounded_context', {})
        requirements = bc.get('requirements', [])
        
        # Requirements를 text로 변환
        requirements_text = self._format_requirements(requirements)
        
        # BC 관계 정보 추출
        context_relations = state.get('bounded_context', {}).get('context_relations', [])
        context_relations_text = self._format_context_relations(context_relations)
        
        return {
            "requirements_text": requirements_text,
            "context_relations_text": context_relations_text,
            "progress": 30,
            "logs": [{
                "timestamp": datetime.now().isoformat(),
                "message": "Input preparation completed"
            }]
        }
    
    def _format_requirements(self, requirements) -> str:
        """Requirements를 텍스트로 포맷팅"""
        if not requirements:
            return "No specific requirements mapped for this bounded context."
        
        # Requirements가 object 형태인 경우 (Frontend 형식)
        if isinstance(requirements, dict):
            text = "## Requirements\n\n"
            
            if requirements.get('userStory'):
                text += "### User Stories:\n"
                text += requirements.get('userStory', '') + "\n\n"
            
            if requirements.get('event'):
                text += "### Events:\n"
                text += requirements.get('event', '') + "\n\n"
            
            if requirements.get('ddl'):
                text += "### DDL Schemas:\n"
                text += requirements.get('ddl', '') + "\n\n"
            
            if requirements.get('eventNames'):
                text += "### Related Events:\n"
                text += requirements.get('eventNames', '') + "\n\n"
            
            return text
        
        # Requirements가 배열 형태인 경우 (기존 형식)
        text = "## Mapped Requirements\n\n"
        
        user_stories = [r for r in requirements if r.get('type') == 'userStory']
        if user_stories:
            text += "### User Stories:\n"
            for req in user_stories:
                text += f"- {req.get('text', '')}\n"
            text += "\n"
        
        events = [r for r in requirements if r.get('type') == 'Event']
        if events:
            text += "### Events:\n"
            for req in events:
                text += f"- {req.get('text', '')}\n"
            text += "\n"
        
        ddls = [r for r in requirements if r.get('type') == 'DDL']
        if ddls:
            text += "### DDL Schemas:\n"
            for req in ddls:
                text += f"- {req.get('text', '')}\n"
        
        return text
    
    def _format_context_relations(self, context_relations: List[Dict]) -> str:
        """Context relations을 텍스트로 포맷팅"""
        if not context_relations:
            return "No explicit context relationships defined."
        
        text = "## Context Relations\n\n"
        for relation in context_relations:
            upstream = relation.get('upstream', {})
            downstream = relation.get('downstream', {})
            rel_type = relation.get('type', 'Unknown')
            
            text += f"- **{upstream.get('name', 'Unknown')}** → **{downstream.get('name', 'Unknown')}** ({rel_type})\n"
        
        return text
    
    def _generate_draft(self, state: AggregateDraftState) -> Dict:
        """Aggregate 초안 생성"""
        LoggingUtil.info("AggregateDraftGenerator", "Generating aggregate draft options")
        
        bc = state.get('bounded_context', {})
        bc_name = bc.get('name', 'Unknown')
        bc_alias = bc.get('alias', '')
        description = state.get('description', '')
        accumulated_drafts = state.get('accumulated_drafts', {})
        bc_aggregates = bc.get('aggregates', [])
        
        # 프론트엔드 DraftGeneratorByFunctions.__sanitizeAccumulatedDrafts와 동일
        # accumulated_drafts에서 refs 제거 (복사본 사용, 원본 state는 수정하지 않음)
        sanitized_accumulated_drafts = accumulated_drafts
        if accumulated_drafts:
            import copy
            sanitized_accumulated_drafts = copy.deepcopy(accumulated_drafts)
            sanitized_accumulated_drafts = RefsTraceUtil.remove_refs_attributes(sanitized_accumulated_drafts)
        
        # 프롬프트 구성
        prompt = self._build_prompt(
            bc_name, bc_alias, description,
            state.get('requirements_text', ''),
            state.get('context_relations_text', ''),
            sanitized_accumulated_drafts,  # refs가 제거된 복사본 사용
            bc_aggregates
        )
        
        try:
            # LLM 호출
            result = self.llm_structured.invoke(prompt)
            
            # LLM이 생성한 options 가져오기
            options = result.get('result', {}).get('options', [])
            
            # 각 option에 boundedContext.requirements.traceMap 추가
            # 입력으로 받은 bounded_context에서 requirements.traceMap 가져오기
            bc_requirements = bc.get('requirements', {})
            trace_map = bc_requirements.get('traceMap') if isinstance(bc_requirements, dict) else None
            
            # 각 option에 boundedContext 필드 추가/보완
            for option in options:
                # boundedContext 필드가 없으면 생성
                if 'boundedContext' not in option:
                    option['boundedContext'] = {}
                
                # requirements 필드가 없으면 생성
                if 'requirements' not in option['boundedContext']:
                    option['boundedContext']['requirements'] = {}
                
                # traceMap이 있으면 추가 (없으면 빈 객체라도 추가)
                if trace_map is not None:
                    option['boundedContext']['requirements']['traceMap'] = trace_map
                elif 'traceMap' not in option['boundedContext']['requirements']:
                    option['boundedContext']['requirements']['traceMap'] = {}
                
                # structure의 각 항목에 enumerations, valueObjects 보장
                if 'structure' in option:
                    for structure_item in option['structure']:
                        # enumerations가 없으면 빈 배열로 초기화
                        if 'enumerations' not in structure_item:
                            structure_item['enumerations'] = []
                        
                        # valueObjects가 없으면 빈 배열로 초기화
                        if 'valueObjects' not in structure_item:
                            structure_item['valueObjects'] = []
            
            LoggingUtil.info("AggregateDraftGenerator", f"Completed: {len(options)} options for {bc_name}")
            
            return {
                "inference": result.get('inference', ''),
                "options": options,
                "default_option_index": result.get('result', {}).get('defaultOptionIndex', 1),
                "conclusions": result.get('result', {}).get('conclusions', ''),
                "progress": 80,
                "logs": [{
                    "timestamp": datetime.now().isoformat(),
                    "message": f"Generated aggregate drafts for {bc_name}"
                }]
            }
            
        except Exception as e:
            LoggingUtil.exception("AggregateDraftGenerator", f"Failed to generate drafts for {bc_name}", e)
            return {
                "options": [],
                "default_option_index": 0,
                "conclusions": "",
                "inference": f"Error: {str(e)}",
                "error": str(e),
                "progress": 80,
                "logs": [{
                    "timestamp": datetime.now().isoformat(),
                    "message": f"Failed to generate drafts: {str(e)}"
                }]
            }
    
    def _build_prompt(self, bc_name: str, bc_alias: str, description: str, 
                     requirements_text: str, context_relations_text: str, 
                     accumulated_drafts: Dict, bc_aggregates: List) -> str:
        """프롬프트 구성 (프론트엔드와 동일한 구조)"""
        
        # Format accumulated drafts
        accumulated_info = ""
        if accumulated_drafts:
            accumulated_info = f"""
<accumulated_drafts>
{json.dumps(accumulated_drafts, ensure_ascii=False, indent=2)}
</accumulated_drafts>
"""
        
        prompt = f"""You are tasked with drafting a proposal to define multiple Aggregates within a specified Bounded Context based on provided functional requirements and business rules.

## Guidelines

1. **Alignment with Functional Requirements and Business Rules**
   - Ensure that all design proposals fully satisfy the given functional requirements
   - Accurately address every business rule and constraint within your design

2. **Event-Driven Design Considerations**
   - Analyze the provided events to understand the domain behaviors and state transitions
   - Design aggregates that can naturally produce and handle the identified domain events
   - Use events to identify aggregate boundaries - operations that should be atomic typically belong to the same aggregate

3. **Context Relationship Analysis**
   - Examine context relations to understand how this bounded context interacts with other contexts
   - Design aggregates that support the identified interaction patterns (Pub/Sub, API calls, etc.)
   - Consider the direction of relationships when defining aggregate dependencies and references

4. **Transactional Consistency**
   - Consolidate transaction-critical data within a single Aggregate to preserve atomicity
   - Avoid splitting core transactional data (e.g., do not separate loan/loan details or order/order items)
   - Define Aggregate boundaries that respect inherent business invariants and support identified events

5. **Design for Maintainability**
   - Distribute properties across well-defined Value Objects to improve maintainability
   - Avoid creating Value Objects with only one property unless they represent a significant domain concept
   - Do not derive an excessive number of Value Objects

6. **Proper Use of Enumerations**
   - When storing state or similar information, always use Enumerations
   - Ensure that all Enumerations are directly associated with the Aggregate

7. **Naming and Language Conventions**
   - Use English for all object names
   - Do not include type information in names or aliases (e.g., use "Book" instead of "BookAggregate")
   - Within a single option, each name and alias must be unique

8. **Reference Handling and Duplication Avoidance**
   - Before creating an Aggregate, check if an Aggregate with the same core concept already exists in accumulated drafts or other contexts
   - If it exists, reference it using a Value Object with a foreign key rather than duplicating its definition

9. **Aggregate References**
   - Aggregates that relate to other Aggregates should use Value Objects to hold these references
   - When referencing another Aggregate via a ValueObject, write the name as '<Referenced Aggregate Name> + Reference'
   - Avoid bidirectional references: ensure that references remain unidirectional

10. **High-Quality Evaluation of Options**
    - For each design option, provide specific and concrete pros and cons
    - Evaluate options based on design quality attributes with specific consequences and examples:
      * Cohesion: How focused each aggregate is on a single responsibility
      * Coupling: Dependencies between aggregates and their impact on system flexibility
      * Consistency: How well business invariants are protected within transaction boundaries
      * Encapsulation: How effectively domain rules are hidden
      * Complexity: Cognitive load for developers
      * Independence: How autonomously each aggregate can evolve
      * Performance: Query efficiency, memory usage, and operational characteristics
    - Ensure pros and cons are meaningfully different between options

## Input

<target_bounded_context>
Name: {bc_name}
Alias: {bc_alias}
Description: {description}
</target_bounded_context>

<functional_requirements>
{requirements_text}
</functional_requirements>

<context_relations>
{context_relations_text}
</context_relations>

{accumulated_info}

<suggested_aggregates>
Please consider including aggregates with the following names: {', '.join([f"{agg.get('name', '')}({agg.get('alias', '')})" for agg in bc_aggregates]) if bc_aggregates else 'None'}
</suggested_aggregates>

## Task
Generate 2-3 aggregate design options for this bounded context. Each option should propose different approaches to organizing aggregates.

Default Option Selection Priority: Consistency > Event Alignment > Context Integration > Domain Alignment > Performance > Maintainability > Flexibility

Generate now."""
        
        return prompt
    
    def _finalize(self, state: AggregateDraftState) -> Dict:
        """최종 결과 정리"""
        return {
            "is_completed": True,
            "progress": 100,
            "logs": [{
                "timestamp": datetime.now().isoformat(),
                "message": "Aggregate draft generation completed"
            }]
        }
    
    def run(self, inputs: Dict) -> Dict:
        """워크플로우 실행"""
        initial_state: AggregateDraftState = {
            "bounded_context": inputs.get("bounded_context", {}),
            "description": inputs.get("description", ""),
            "accumulated_drafts": inputs.get("accumulated_drafts", {}),
            "analysis_result": inputs.get("analysis_result", {}),
            "requirements_text": "",
            "context_relations_text": "",
            "inference": "",
            "options": [],
            "default_option_index": 0,
            "conclusions": "",
            "progress": 0,
            "logs": [],
            "is_completed": False,
            "error": ""
        }
        
        result = self.workflow.invoke(initial_state)
        return result

