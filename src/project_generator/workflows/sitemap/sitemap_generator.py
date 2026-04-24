from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from src.project_generator.utils.logging_util import LoggingUtil
from src.project_generator.systems.storage_system_factory import StorageSystemFactory
from src.project_generator.utils.refs_trace_util import RefsTraceUtil
from src.project_generator.utils.llm_factory import create_chat_llm
import json
import re

class SiteMapState(TypedDict):
    """SiteMap 생성 워크플로우 상태"""
    job_id: str
    requirements: str
    bounded_contexts: list
    command_readmodel_data: dict
    existing_navigation: list
    site_map: dict
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
    sentences = re.split(r'(?<=[.!?\n])\s+', requirements)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [requirements]

def merge_sitemap_data(existing_sitemap: dict, new_sitemap: dict) -> dict:
    """
    기존 사이트맵과 새로운 사이트맵 데이터를 병합 (안전하게 path 처리)
    """
    if not existing_sitemap or not existing_sitemap.get("pages"):
        return new_sitemap
    
    # 기존 페이지들을 path를 키로 하는 딕셔너리로 변환 (path가 없는 경우 id나 title 사용)
    merged_pages = {}
    for page in existing_sitemap.get("pages", []):
        key = page.get("path") or page.get("id") or page.get("title", "unknown")
        merged_pages[key] = page
    
    # 새로운 페이지 병합
    for new_page in new_sitemap.get("pages", []):
        # path가 없으면 id나 title로 대체
        page_key = new_page.get("path") or new_page.get("id") or new_page.get("title", "unknown")
        
        if page_key in merged_pages:
            # 기존 페이지에 children 추가 (중복 제거)
            existing_page = merged_pages[page_key]
            existing_children = existing_page.get("children", [])
            new_children = new_page.get("children", [])
            
            # children의 키 결정 (path, id, title 순서로)
            existing_child_keys = {
                child.get("path") or child.get("id") or child.get("title", "unknown") 
                for child in existing_children
            }
            
            unique_new_children = [
                child for child in new_children 
                if (child.get("path") or child.get("id") or child.get("title", "unknown")) not in existing_child_keys
            ]
            
            existing_page["children"] = existing_children + unique_new_children
        else:
            # 새로운 페이지 추가
            merged_pages[page_key] = new_page
    
    return {
        "title": existing_sitemap.get("title") or new_sitemap.get("title", "새로운 웹사이트"),
        "description": existing_sitemap.get("description") or new_sitemap.get("description", "웹사이트 설명"),
        "pages": list(merged_pages.values())
    }

def generate_sitemap(state: SiteMapState) -> SiteMapState:
    """
    Command/ReadModel 데이터를 기반으로 SiteMap 생성 (청크별 순차 처리)
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
                state["site_map"] = {"pages": []}
            else:
                # 단일 청크로 처리
                state["chunks"] = [requirements]
                state["total_chunks"] = 1
                state["current_chunk"] = 0
                state["site_map"] = {"pages": []}
        
        current_chunk_index = state["current_chunk"]
        total_chunks = state["total_chunks"]
        chunk_text = state["chunks"][current_chunk_index]
        
        LoggingUtil.info(state["job_id"], f"Processing chunk {current_chunk_index + 1}/{total_chunks}...")
        
        # 언어 감지 (한국어 여부 체크)
        has_korean = any('\uac00' <= c <= '\ud7a3' for c in chunk_text[:500])
        language = "Korean" if has_korean else "English"
        
        llm = create_chat_llm(
            temperature=0.2  # Frontend와 동일
        )
        
        # 'ui' BC 필터링
        filtered_bcs = [bc for bc in state["bounded_contexts"] if bc.get("name") != "ui"]
        bounded_contexts_json = json.dumps(filtered_bcs, ensure_ascii=False, indent=2)
        # 프론트엔드와 동일하게 refs 제거 (RefsTraceUtil.removeRefsAttributes)
        command_readmodel_data_without_refs = RefsTraceUtil.remove_refs_attributes(
            state["command_readmodel_data"] or {}
        )
        command_readmodel_json = json.dumps(command_readmodel_data_without_refs, ensure_ascii=False, indent=2)
        
        # 기존 사이트맵 데이터를 existing_navigation으로 변환
        existing_navigation = state.get("site_map", {}).get("pages", [])
        
        # 기존 Navigation이 있는 경우 프롬프트 추가
        existing_navigation_prompt = ""
        if existing_navigation and len(existing_navigation) > 0:
            existing_titles = extract_existing_titles(existing_navigation)
            existing_navigation_prompt = f"""
ALREADY GENERATED PAGES (DO NOT DUPLICATE):
{', '.join(existing_titles)}

IMPORTANT: 
- Only generate NEW pages and components from the current requirements chunk
- Do not regenerate pages that are already in the list above
- If a requirement relates to an existing page, skip it
- Focus only on extracting NEW content from the current chunk
"""
        
        # 프론트엔드와 동일한 상세한 프롬프트
        prompt = f"""You are an expert UX designer and web architect. Generate a comprehensive website sitemap JSON structure based on user requirements.

<chunk_info>Chunk {current_chunk_index + 1} of {total_chunks}</chunk_info>

REQUIREMENTS:
{chunk_text}

{existing_navigation_prompt}

BOUNDED CONTEXTS:
Bounded Contexts List: {bounded_contexts_json}
Command/ReadModel Data: {command_readmodel_json}

TASK:
Generate a user-friendly website sitemap that represents actual web pages and sections.
Focus on user experience and logical page organization, not technical implementation.
Create a hierarchical structure that makes sense to end users.

COMMAND/READMODEL INTEGRATION:
CRITICAL: You MUST use the provided Command/ReadModel data to create accurate UI components with proper references.

COMMAND/READMODEL TYPES:

Command Types (Business Logic):
- Creation Commands: Create[Entity], Register[Entity], Add[Entity]
- Update Commands: Update[Entity], Modify[Entity], Change[Entity]
- Deletion Commands: Delete[Entity], Remove[Entity], Cancel[Entity]
- Process Commands: Process[Action], Confirm[Entity], Validate[Entity]
- Integration Commands: Sync[Entity], Notify[Entity], Detect[Condition]

ReadModel Types (Query Operations):
- Entity Queries: [Entity]Detail, [Entity]Profile, [Entity]Info
- List Queries: [Entity]List, [Entity]History, [Entity]Search
- Admin Queries: [Entity]ListAdmin, [Entity]Statistics, [Entity]Report
- UI Support: [Entity]FormView, [Entity]SearchView, [Entity]Options
- Status Queries: [Entity]Status, [Entity]Availability, [Entity]Alert

REFERENCE RULES:
- Each component must reference exactly ONE Command or ReadModel from the provided data
- Use the actual Command/ReadModel names from the Command/ReadModel Data section
- Exclude pure UI components (Header, Footer, Navbar) - focus on business logic components only

STRUCTURE GUIDELINES:
- Home page as the root with main navigation sections as children
- Main sections (e.g., Rooms, Amenities, Contact Us) as direct children of Home
- Each main section contains business logic components only
- Page components are organized vertically under each main section
- Clear 3-level hierarchy: Home → Main Pages → Business Components
- Focus on actual website structure with Home as entry point
- Group business logic components under their parent page for better organization
- Each component should have only ONE reference (single Command or ReadModel)
- Exclude pure UI components (Header, Footer, Navbar) - focus on business logic components only
- Create separate components for different business operations (e.g., separate login form and register form)

OUTPUT FORMAT:
{{
  "siteMap": {{
    "title": "Website Title",
    "description": "Website purpose and description",
    "pages": [
      {{
        "id": "page-id",
        "title": "Page Title",
        "url": "/page-url",
        "description": "Page description",
        "children": [
          {{
            "id": "component-id",
            "title": "Component Title",
            "url": "",
            "description": "Component description",
            "reference": "CommandOrReadModelName"
          }}
        ]
      }}
    ]
  }}
}}

LANGUAGE REQUIREMENT:
Please generate the response in {language}. All descriptions, titles, and text fields should be in {language}, while technical names (page paths, component names) should remain in English.

RULES:
- Create logical page structure based on requirements
- Each component must reference exactly ONE Command or ReadModel
- Exclude pure UI components (Header, Footer, Navbar)
- Return ONLY JSON, no explanations"""

        response = llm.invoke(prompt)
        response_text = response.content
        
        # JSON 파싱
        try:
            # JSON 코드 블록 제거
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            parsed_json = json.loads(response_text)
            
            # siteMap 필드 추출 (LLM이 { "siteMap": {...} } 형식으로 반환)
            new_site_map_data = parsed_json.get('siteMap', {})
            
            # 기존 데이터와 병합
            merged_sitemap = merge_sitemap_data(state["site_map"], new_site_map_data)
            
            LoggingUtil.info(state["job_id"], f"Chunk {current_chunk_index + 1}/{total_chunks} processed: {len(new_site_map_data.get('pages', []))} pages")
            
            # 다음 청크 처리 여부 결정
            next_chunk = current_chunk_index + 1
            progress = int((next_chunk / total_chunks) * 80)  # 80%까지는 생성 단계
            
            # Firebase에 중간 결과 업데이트 (UI에 실시간 반영)
            try:
                storage = StorageSystemFactory.instance()
                job_path = f"jobs/sitemap_generator/{state['job_id']}"
                
                # 중간 결과를 Firebase에 업데이트 (outputs 경로에 저장)
                output_path = f"{job_path}/state/outputs"
                storage.update_data(output_path, {
                    "siteMap": merged_sitemap,
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
                    "site_map": merged_sitemap,
                    "current_chunk": next_chunk,
                    "logs": [f"Chunk {next_chunk}/{total_chunks} generation in progress"],
                    "progress": progress
                }
            else:
                # 모든 청크 처리 완료
                return {
                    **state,
                    "site_map": merged_sitemap,
                    "logs": [f"All chunks processed. Total {len(merged_sitemap.get('pages', []))} pages"],
                    "progress": 80
                }
            
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON response: {str(e)}"
            LoggingUtil.info(state["job_id"], error_msg)
            return {
                **state,
                "is_failed": True,
                "error": error_msg,
                "logs": [error_msg]
            }
            
    except Exception as e:
        error_msg = f"Error in generate_sitemap: {str(e)}"
        LoggingUtil.info(state["job_id"], error_msg)
        return {
            **state,
            "is_failed": True,
            "error": error_msg,
            "logs": [error_msg]
        }

def extract_existing_titles(navigation):
    """
    기존 Navigation에서 모든 제목 추출
    """
    titles = []
    
    def extract_recursive(nodes):
        for node in nodes:
            if node.get("title"):
                titles.append(node["title"])
            if node.get("children") and isinstance(node["children"], list):
                extract_recursive(node["children"])
    
    extract_recursive(navigation)
    return titles

def finalize_result(state: SiteMapState) -> SiteMapState:
    """
    최종 결과 구성
    """
    try:
        LoggingUtil.info(state["job_id"], "Finalizing SiteMap generation result...")
        
        return {
            **state,
            "site_map": state["site_map"],
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

def should_continue_generation(state: SiteMapState) -> str:
    """
    청크 처리를 계속할지 결정
    
    IMPORTANT: current_chunk는 이미 처리된 청크의 인덱스
    다음 청크가 존재하는지 확인하려면 current_chunk + 1 < total_chunks
    """
    if state.get("is_failed"):
        return "finalize_result"
    
    # generate_sitemap 함수가 다음 청크 인덱스를 이미 설정했는지 확인
    # 로그에서 "All chunks processed"가 있으면 완료된 것
    logs = state.get("logs", [])
    if logs and any("All chunks processed" in log for log in logs):
        return "finalize_result"
    
    current_chunk = state.get("current_chunk", 0)
    total_chunks = state.get("total_chunks", 1)
    
    # current_chunk는 다음에 처리할 청크의 인덱스
    # 예: chunk 0 처리 후 current_chunk = 1 설정됨
    if current_chunk < total_chunks:
        return "generate_sitemap"
    else:
        return "finalize_result"

def create_sitemap_workflow():
    """
    SiteMap 생성 워크플로우 생성 (루프 지원)
    """
    workflow = StateGraph(SiteMapState)
    
    # 노드 추가
    workflow.add_node("generate_sitemap", generate_sitemap)
    workflow.add_node("finalize_result", finalize_result)
    
    # 엣지 추가
    workflow.set_entry_point("generate_sitemap")
    
    # 조건부 엣지: 다음 청크가 있으면 generate로, 없으면 finalize로
    workflow.add_conditional_edges(
        "generate_sitemap",
        should_continue_generation,
        {
            "generate_sitemap": "generate_sitemap",
            "finalize_result": "finalize_result"
        }
    )
    
    workflow.add_edge("finalize_result", END)
    
    return workflow.compile()

