from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from src.project_generator.utils.logging_util import LoggingUtil
from src.project_generator.systems.storage_system_factory import StorageSystemFactory
from src.project_generator.utils.refs_trace_util import RefsTraceUtil
from src.project_generator.utils.trace_markdown_util import TraceMarkdownUtil
from src.project_generator.utils.llm_factory import create_chat_llm
import json
import re

class CommandReadModelState(TypedDict):
    """Command/ReadModel 추출 워크플로우 상태"""
    job_id: str
    requirements: str
    bounded_contexts: list
    extracted_data: dict
    logs: Annotated[list, lambda x, y: x + y]
    progress: int
    is_completed: bool
    is_failed: bool
    error: str
    current_chunk: int
    total_chunks: int
    chunks: list

def split_requirements_into_chunks(requirements: str, chunk_size: int = 12000) -> list:
    """
    요구사항을 청크로 분할 (문장 단위로 분할하여 문맥 유지)
    """
    # 문장 단위로 분할 (., !, ?, \n\n 기준)
    sentences = re.split(r'(?<=[.!?\n])\s+', requirements)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # 현재 청크에 문장을 추가했을 때 크기 확인
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # 마지막 청크 추가
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [requirements]

def merge_extracted_data(existing_data: dict, new_data: dict) -> dict:
    """
    기존 추출 데이터와 새로운 추출 데이터를 병합
    """
    if not existing_data or not existing_data.get("boundedContexts"):
        return new_data
    
    merged_bcs = {bc["name"]: bc for bc in existing_data.get("boundedContexts", [])}
    
    for new_bc in new_data.get("boundedContexts", []):
        bc_name = new_bc["name"]
        
        if bc_name in merged_bcs:
            # 기존 BC에 commands와 readModels 추가 (중복 제거)
            existing_bc = merged_bcs[bc_name]
            
            # Commands 병합
            existing_command_names = {cmd["name"] for cmd in existing_bc.get("commands", [])}
            new_commands = [cmd for cmd in new_bc.get("commands", []) if cmd["name"] not in existing_command_names]
            existing_bc["commands"] = existing_bc.get("commands", []) + new_commands
            
            # ReadModels 병합
            existing_readmodel_names = {rm["name"] for rm in existing_bc.get("readModels", [])}
            new_readmodels = [rm for rm in new_bc.get("readModels", []) if rm["name"] not in existing_readmodel_names]
            existing_bc["readModels"] = existing_bc.get("readModels", []) + new_readmodels
        else:
            # 새로운 BC 추가
            merged_bcs[bc_name] = new_bc
    
    return {"boundedContexts": list(merged_bcs.values())}

def extract_commands_and_readmodels(state: CommandReadModelState) -> CommandReadModelState:
    """
    요구사항에서 Command와 ReadModel 추출 (청크별 순차 처리)
    """
    try:
        # 첫 호출 시 청크 분할
        if state.get("current_chunk", 0) == 0:
            requirements = state["requirements"]
            
            # 요구사항 길이 체크 및 청크 분할 결정
            if len(requirements) > 12000:
                chunks = split_requirements_into_chunks(requirements, chunk_size=12000)
                LoggingUtil.info(state["job_id"], f"Requirements split into {len(chunks)} chunks")
                state["chunks"] = chunks
                state["total_chunks"] = len(chunks)
                state["current_chunk"] = 0
                state["extracted_data"] = {"boundedContexts": []}
            else:
                # 단일 청크로 처리
                state["chunks"] = [requirements]
                state["total_chunks"] = 1
                state["current_chunk"] = 0
                state["extracted_data"] = {"boundedContexts": []}
        
        current_chunk_index = state["current_chunk"]
        total_chunks = state["total_chunks"]
        chunk_text = state["chunks"][current_chunk_index]
        
        LoggingUtil.info(state["job_id"], f"Processing chunk {current_chunk_index + 1}/{total_chunks}...")
        
        # 언어 감지 (한국어 여부 체크)
        has_korean = any('\uac00' <= c <= '\ud7a3' for c in chunk_text[:500])
        language = "Korean" if has_korean else "English"
        
        # Structured Output Schema (Frontend의 Zod schema와 동일)
        response_schema = {
            "type": "object",
            "title": "CommandReadModelExtraction",
            "description": "Extracted commands and read models organized by bounded contexts",
            "properties": {
                "extractedData": {
                    "type": "object",
                    "properties": {
                        "boundedContexts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "commands": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "actor": {"type": "string", "enum": ["user", "admin", "system", "external"]},
                                                "aggregate": {"type": "string"},
                                                "description": {"type": "string"}
                                            },
                                            "required": ["name", "actor", "aggregate", "description"],
                                            "additionalProperties": False
                                        }
                                    },
                                    "readModels": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "actor": {"type": "string", "enum": ["user", "admin", "system"]},
                                                "aggregate": {"type": "string"},
                                                "isMultipleResult": {"type": "boolean"},
                                                "description": {"type": "string"}
                                            },
                                            "required": ["name", "actor", "aggregate", "isMultipleResult", "description"],
                                            "additionalProperties": False
                                        }
                                    }
                                },
                                "required": ["name", "commands", "readModels"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["boundedContexts"],
                    "additionalProperties": False
                }
            },
            "required": ["extractedData"],
            "additionalProperties": False
        }
        
        llm = create_chat_llm(
            temperature=0.2  # Frontend와 동일 (0.3 → 0.2)
        )
        
        # Structured Output 적용
        llm_structured = llm.with_structured_output(response_schema, strict=True)
        
        bounded_contexts_json = json.dumps(state["bounded_contexts"], ensure_ascii=False, indent=2)
        
        # 기존 추출 데이터가 있으면 프롬프트에 추가 (Frontend와 동일한 요약 방식)
        existing_data_prompt = ""
        if state["extracted_data"].get("boundedContexts"):
            # Frontend의 _createAccumulatedSummary와 동일하게 이름만 추출
            accumulated_summary = []
            for bc in state["extracted_data"].get("boundedContexts", []):
                bc_name = bc.get("name", "")
                accumulated_summary.append(f"## Bounded Context: {bc_name}")
                
                # Commands 이름만
                commands = bc.get("commands", [])
                if commands:
                    command_names = [cmd.get("name", "") for cmd in commands if cmd.get("name")]
                    accumulated_summary.append(f"Commands: {', '.join(command_names)}")
                
                # ReadModels 이름만
                readmodels = bc.get("readModels", [])
                if readmodels:
                    readmodel_names = [rm.get("name", "") for rm in readmodels if rm.get("name")]
                    accumulated_summary.append(f"ReadModels: {', '.join(readmodel_names)}")
                
                accumulated_summary.append("")  # 빈 줄
            
            accumulated_summary_text = "\n".join(accumulated_summary)
            
            # Frontend의 RecursiveCommandReadModelExtractor와 동일한 구조
            existing_data_prompt = f"""
<recursive_extraction_context>
    <previous_extracted_data_summary>
        <title>Previously Accumulated Results - DO NOT RETURN THESE AGAIN</title>
        <warning>⚠️ The items listed below have ALREADY been extracted and processed. DO NOT include them in your output.</warning>
        <description>The following Commands and ReadModels were already extracted from previous requirement chunks and must be EXCLUDED from your current extraction:</description>
        <exclusion_list>
{accumulated_summary_text}
        </exclusion_list>
        <reminder>Extract ONLY new operations from the current chunk that are NOT listed above</reminder>
    </previous_extracted_data_summary>
    
    <task_objectives>
        <objective id="1">Extract ONLY NEW Commands and ReadModels from the current chunk that are NOT already in previous results</objective>
        <objective id="2">DO NOT return any Commands or ReadModels that already exist in the previously accumulated results</objective>
        <objective id="3">Maintain Bounded Context categorization consistency with previous extractions</objective>
        <objective id="4">The system will handle merging automatically - you only provide new items</objective>
    </task_objectives>
    
    <duplicate_avoidance_rules>
        <rule id="1">Skip if Command/ReadModel name already exists (case-sensitive comparison)</rule>
        <rule id="2">Skip if the same name exists within the same Bounded Context</rule>
        <rule id="3">Skip if functionally equivalent operation already exists, even with slightly different naming</rule>
        <rule id="4">If no new operations are found in the current chunk, return empty commands and readModels arrays</rule>
    </duplicate_avoidance_rules>
</recursive_extraction_context>
"""
        
        # 프론트엔드와 동일한 상세한 XML 기반 프롬프트
        task_guidelines = """<instruction>
    <core_instructions>
        <title>Command and ReadModel Extraction Task</title>
        <task_description>Analyze business requirements and extract all business operations, categorizing them into Commands (state-changing operations) and ReadModels (query operations), organized by their corresponding Bounded Contexts.</task_description>
        
        <input_description>
            <title>You will be given:</title>
            <item id="1">**Requirements:** Business requirements describing the system functionality</item>
            <item id="2">**Bounded Contexts:** List of identified bounded contexts with their aggregates</item>
        </input_description>

        <guidelines>
            <title>Extraction Guidelines</title>
            
            <section id="command_extraction">
                <title>Command Extraction Rules</title>
                <description>Commands represent state-changing operations that modify the system's data or state.</description>
                
                <rule id="1">
                    <name>State-Changing Operations</name>
                    <description>Extract all operations that modify system state or data</description>
                    <examples>
                        <category name="Create">CreateReservation, RegisterUser, CreateFlight</category>
                        <category name="Update">UpdateProfile, UpdateReservation, UpdateFlight</category>
                        <category name="Delete">CancelReservation, DeleteFlight, DeleteSeat</category>
                        <category name="Process">ProcessPayment, ConfirmReservation, VerifyEmail</category>
                    </examples>
                </rule>
                
                <rule id="2">
                    <name>Business Processes</name>
                    <description>Operations that execute specific business rules or logic</description>
                    <examples>
                        <category name="Validate">ValidateReservation, AuthenticateUser</category>
                        <category name="Calculate">CalculatePrice, ComputeRefund</category>
                        <category name="Notify">SendNotification, IssueAuthToken</category>
                    </examples>
                </rule>
                
                <rule id="3">
                    <name>External System Integration</name>
                    <description>Operations involving interaction with external systems</description>
                    <examples>
                        <category name="Synchronize">SyncFlightInfo, SyncSeatInfo</category>
                        <category name="Detect">DetectFraudulentReservation</category>
                    </examples>
                </rule>
                
                <rule id="4">
                    <name>Naming Convention</name>
                    <description>Use Verb + Noun pattern in PascalCase (e.g., CreateOrder, UpdateProfile)</description>
                </rule>
                
                <rule id="5">
                    <name>Actor Identification</name>
                    <description>Assign appropriate actor: user, admin, system, or external</description>
                </rule>
            </section>

            <section id="readmodel_extraction">
                <title>ReadModel (View) Extraction Rules</title>
                <description>ReadModels represent query operations that retrieve data without changing state.</description>
                
                <rule id="1">
                    <name>Data Retrieval</name>
                    <description>Operations to fetch data without modifying it</description>
                    <examples>
                        <category name="Single Retrieval">UserProfile, ReservationDetail, FlightDetail</category>
                        <category name="List Retrieval">FlightList, ReservationHistory, InquiryList</category>
                    </examples>
                </rule>
                
                <rule id="2">
                    <name>Search and Filtering</name>
                    <description>Data retrieval based on specific conditions or criteria</description>
                    <examples>
                        <category name="Search">FlightSearch, SearchReservations</category>
                        <category name="Filtering">FilteredFlightList, AvailableSeats</category>
                    </examples>
                </rule>
                
                <rule id="3">
                    <name>Statistics and Reports</name>
                    <description>Aggregated data or summary information</description>
                    <examples>
                        <category name="Statistics">ReservationStatistics, SalesReport</category>
                        <category name="Status">SeatAvailability, FlightStatus</category>
                    </examples>
                </rule>
                
                <rule id="4">
                    <name>UI Support Data</name>
                    <description>Data required for screen composition and user interface</description>
                    <examples>
                        <category name="Option Lists">AirportList, SeatClassOptions</category>
                        <category name="Configuration">UserPreferences, SystemSettings</category>
                    </examples>
                </rule>
                
                <rule id="5">
                    <name>Naming Convention</name>
                    <description>Use Noun + Purpose pattern in PascalCase (e.g., FlightList, UserProfile)</description>
                </rule>
                
                <rule id="6">
                    <name>Multiple Result Indicator</name>
                    <description>Set isMultipleResult to true for list/collection results, false for single item results</description>
                </rule>
                
                <rule id="7">
                    <name>Actor Identification</name>
                    <description>Assign appropriate actor: user, admin, or system</description>
                </rule>
            </section>

            <section id="bounded_context_assignment">
                <title>Bounded Context Assignment Strategy</title>
                
                <rule id="1">
                    <name>Domain Alignment</name>
                    <description>Assign commands and readModels to the most appropriate Bounded Context based on domain responsibility and business capability</description>
                </rule>
                
                <rule id="2">
                    <name>Aggregate Alignment</name>
                    <description>Consider which aggregates within each Bounded Context are most relevant to the operation</description>
                </rule>
                
                <rule id="3">
                    <name>Business Logic Grouping</name>
                    <description>Group related operations within the same Bounded Context to maintain cohesion</description>
                </rule>
                
                <rule id="4">
                    <name>Comprehensive Coverage</name>
                    <description>Extract ALL business operations from requirements without omission</description>
                </rule>
            </section>

            <section id="quality_standards">
                <title>Quality Standards</title>
                
                <rule id="1">
                    <name>Completeness</name>
                    <description>Extract ALL business operations mentioned in requirements</description>
                </rule>
                
                <rule id="2">
                    <name>Domain-Specific Focus</name>
                    <description>Focus on domain-specific operations, not generic CRUD unless explicitly required</description>
                </rule>
                
                <rule id="3">
                    <name>Clear Naming</name>
                    <description>Use clear, descriptive names that reflect business intent and purpose</description>
                </rule>
                
                <rule id="4">
                    <name>Proper Categorization</name>
                    <description>Ensure commands and readModels are correctly distinguished and categorized</description>
                </rule>
                
                <rule id="5">
                    <name>Meaningful Descriptions</name>
                    <description>Provide clear, concise descriptions explaining the purpose and business value of each operation</description>
                </rule>
            </section>
        </guidelines>
    </core_instructions>
    
    <output_format>
        <title>JSON Output Format</title>
        <description>The output must be a JSON object structured as follows:</description>
        <schema>
{
    "extractedData": {
        "boundedContexts": [
            {
                "name": "(BoundedContextName)",
                "commands": [
                    {
                        "name": "(CommandName in PascalCase, Verb+Noun)",
                        "actor": "(user|admin|system|external)",
                        "aggregate": "(AggregateName)",
                        "description": "(Clear description of what this command does and its business purpose)"
                    }
                ],
                "readModels": [
                    {
                        "name": "(ReadModelName in PascalCase, Noun+Purpose)",
                        "actor": "(user|admin|system)",
                        "aggregate": "(AggregateName)",
                        "isMultipleResult": (true for lists/collections, false for single items),
                        "description": "(Clear description of what data this readModel retrieves and its purpose)"
                    }
                ]
            }
        ]
    }
}
        </schema>
        <field_requirements>
            <requirement id="1">All field names must match exactly as shown in the schema</requirement>
            <requirement id="2">Command names must follow Verb+Noun pattern (e.g., CreateOrder, ProcessPayment)</requirement>
            <requirement id="3">ReadModel names must follow Noun+Purpose pattern (e.g., OrderList, UserProfile)</requirement>
            <requirement id="4">Actor values must be exactly one of: user, admin, system, external (for commands) or user, admin, system (for readModels)</requirement>
            <requirement id="5">Aggregate names must match those defined in the bounded contexts</requirement>
            <requirement id="6">Descriptions must be clear and explain business purpose</requirement>
        </field_requirements>
    </output_format>
</instruction>"""
        
        prompt = f"""{task_guidelines}

<user_input>
<current_chunk>Chunk {current_chunk_index + 1} of {total_chunks}</current_chunk>

<requirements>
{chunk_text}
</requirements>

<bounded_contexts>
{bounded_contexts_json}
</bounded_contexts>

{existing_data_prompt}
</user_input>

LANGUAGE REQUIREMENT:
Please generate the response in {language}. All descriptions and text fields should be in {language}, while technical names (command names, aggregate names) should remain in English.

CRITICAL INSTRUCTIONS:
- Extract ALL business operations mentioned in requirements
- Focus on domain-specific operations, not generic CRUD unless explicitly required
- Ensure commands and readModels are correctly distinguished and categorized
- Use clear, descriptive names that reflect business intent
- Assign appropriate actors for each operation
- Return ONLY the JSON object, no additional text or explanations"""

        # Structured Output 사용 (Frontend의 response_format과 동일)
        try:
            result_data = llm_structured.invoke(prompt)
            
            # extractedData 필드 추출
            new_extracted_data = result_data.get('extractedData', {})
            
            # 기존 데이터와 병합
            merged_data = merge_extracted_data(state["extracted_data"], new_extracted_data)
            
            LoggingUtil.info(state["job_id"], f"Chunk {current_chunk_index + 1}/{total_chunks} processed: {len(new_extracted_data.get('boundedContexts', []))} BCs")
            
            # 다음 청크 처리 여부 결정
            next_chunk = current_chunk_index + 1
            progress = int((next_chunk / total_chunks) * 80)  # 80%까지는 추출 단계
            
            # Firebase에 중간 결과 업데이트 (UI에 실시간 반영)
            try:
                storage = StorageSystemFactory.instance()
                job_path = f"jobs/command_readmodel_extractor/{state['job_id']}"
                
                # 중간 결과를 Firebase에 업데이트 (outputs 경로에 저장)
                output_path = f"{job_path}/state/outputs"
                storage.update_data(output_path, {
                    "extractedData": merged_data,
                    "progress": progress,
                    "logs": state["logs"] + [f"Chunk {current_chunk_index + 1}/{total_chunks} completed"]
                })
                LoggingUtil.info(state["job_id"], f"Firebase updated with chunk {current_chunk_index + 1}/{total_chunks} results")
            except Exception as e:
                LoggingUtil.info(state["job_id"], f"Failed to update Firebase: {str(e)}")
            
            if next_chunk < total_chunks:
                # 다음 청크 처리
                return {
                    **state,
                    "extracted_data": merged_data,
                    "current_chunk": next_chunk,
                    "logs": [f"Chunk {next_chunk}/{total_chunks} extraction in progress"],
                    "progress": progress
                }
            else:
                # 모든 청크 처리 완료
                return {
                    **state,
                    "extracted_data": merged_data,
                    "logs": [f"All chunks processed. Total {len(merged_data.get('boundedContexts', []))} bounded contexts"],
                    "progress": 80
                }
            
        except Exception as e:
            error_msg = f"Failed to parse JSON response: {str(e)}"
            LoggingUtil.info(state["job_id"], error_msg)
            return {
                **state,
                "is_failed": True,
                "error": error_msg,
                "logs": [error_msg]
            }
            
    except Exception as e:
        error_msg = f"Error in extract_commands_and_readmodels: {str(e)}"
        LoggingUtil.info(state["job_id"], error_msg)
        return {
            **state,
            "is_failed": True,
            "error": error_msg,
            "logs": [error_msg]
        }

def finalize_result(state: CommandReadModelState) -> CommandReadModelState:
    """
    최종 결과 구성 (프론트엔드와 동일하게 refs 변환 및 검증)
    """
    try:
        LoggingUtil.info(state["job_id"], "Finalizing Command/ReadModel extraction result...")
        
        # 프론트엔드 CommandReadModelExtractor.onModelCreatedWithThinking와 동일한 처리
        # 1. sanitizeAndConvertRefs: phrase → index 변환
        # 2. validateRefs: 검증
        
        extracted_data = state.get("extracted_data", {})
        requirements = state.get("requirements", "")
        
        if extracted_data and requirements:
            # 라인 번호 추가 (프론트엔드 TextTraceUtil.addLineNumbers와 동일)
            # XML 형식: <1>line1</1>\n<2>line2</2>...
            lines = requirements.split('\n')
            line_numbered_requirements = '\n'.join([f"<{i+1}>{line}</{i+1}>" for i, line in enumerate(lines)])
            
            # sanitizeAndConvertRefs 적용
            extracted_data = RefsTraceUtil.sanitize_and_convert_refs(
                extracted_data,
                line_numbered_requirements,
                is_use_xml_base=True
            )
            
            # validateRefs 적용
            try:
                RefsTraceUtil.validate_refs(extracted_data, requirements)
            except ValueError as e:
                LoggingUtil.warning(state["job_id"], f"Refs validation warning: {str(e)}")
                # 검증 실패해도 계속 진행 (프론트엔드와 동일)
        
        return {
            **state,
            "extracted_data": extracted_data,
            "logs": state["logs"] + ["Finalization completed"],
            "progress": 100,
            "is_completed": True
        }
        
    except Exception as e:
        error_msg = f"Error in finalize_result: {str(e)}"
        LoggingUtil.info(state["job_id"], error_msg)
        return {
            **state,
            "is_failed": True,
            "error": error_msg,
            "logs": state["logs"] + [error_msg]
        }

def should_continue_extraction(state: CommandReadModelState) -> str:
    """
    청크 처리를 계속할지 결정
    
    IMPORTANT: current_chunk는 이미 처리된 청크의 인덱스
    다음 청크가 존재하는지 확인하려면 current_chunk + 1 < total_chunks
    """
    if state.get("is_failed"):
        return "finalize_result"
    
    # extract_commands_and_readmodels 함수가 다음 청크 인덱스를 이미 설정했는지 확인
    # 로그에서 "All chunks processed"가 있으면 완료된 것
    logs = state.get("logs", [])
    if logs and any("All chunks processed" in log for log in logs):
        return "finalize_result"
    
    current_chunk = state.get("current_chunk", 0)
    total_chunks = state.get("total_chunks", 1)
    
    # current_chunk는 다음에 처리할 청크의 인덱스
    # 예: chunk 0 처리 후 current_chunk = 1 설정됨
    if current_chunk < total_chunks:
        return "extract_commands_and_readmodels"
    else:
        return "finalize_result"

def create_command_readmodel_workflow():
    """
    Command/ReadModel 추출 워크플로우 생성 (루프 지원)
    """
    workflow = StateGraph(CommandReadModelState)
    
    # 노드 추가
    workflow.add_node("extract_commands_and_readmodels", extract_commands_and_readmodels)
    workflow.add_node("finalize_result", finalize_result)
    
    # 엣지 추가
    workflow.set_entry_point("extract_commands_and_readmodels")
    
    # 조건부 엣지: 다음 청크가 있으면 extract로, 없으면 finalize로
    workflow.add_conditional_edges(
        "extract_commands_and_readmodels",
        should_continue_extraction,
        {
            "extract_commands_and_readmodels": "extract_commands_and_readmodels",
            "finalize_result": "finalize_result"
        }
    )
    
    workflow.add_edge("finalize_result", END)
    
    return workflow.compile()

