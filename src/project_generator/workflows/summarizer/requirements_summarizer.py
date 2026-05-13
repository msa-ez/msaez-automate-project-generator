"""
RequirementsSummarizer - LangGraph 워크플로우
재귀적 청크 기반 요구사항 요약 및 라인 추적
"""
from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, END
import json
import re
from datetime import datetime
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.project_generator.config import Config
from src.project_generator.utils.logging_util import LoggingUtil
from src.project_generator.utils.llm_factory import create_chat_llm

class SummarizerState(TypedDict):
    """요약 생성 상태 (camelCase for Frontend compatibility)"""
    # Inputs
    requirements: str                    # 입력 요구사항 (라인 번호 포함)
    iteration: int                       # 현재 요약 반복 횟수
    
    # Outputs
    summarizedRequirements: List[Dict]   # 요약된 요구사항 목록 (text, refs)
    
    # Metadata
    progress: int                       # 진행률 (0-100)
    logs: Annotated[List[Dict], "append"]  # 로그 (append only)
    isCompleted: bool                   # 완료 여부
    error: str                         # 에러 메시지

class RequirementsSummarizerWorkflow:
    """
    요구사항 요약 워크플로우
    """
    def __init__(self):
        self.llm = create_chat_llm(
            model=Config.DEFAULT_LLM_MODEL,
            temperature=Config.DEFAULT_LLM_TEMPERATURE,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """워크플로우 그래프 구성"""
        workflow = StateGraph(SummarizerState)
        
        # 노드 추가
        workflow.add_node("summarize_requirements", self.summarize_requirements)
        workflow.add_node("finalize", self.finalize)

        # 그래프 연결
        workflow.set_entry_point("summarize_requirements")
        workflow.add_edge("summarize_requirements", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def run(self, inputs: Dict) -> Dict:
        """
        워크플로우 실행
        """
        initial_state: SummarizerState = {
            "requirements": inputs.get("requirements", ""),
            "iteration": inputs.get("iteration", 1),
            "summarizedRequirements": [],
            "progress": 0,
            "logs": [],
            "isCompleted": False,
            "error": ""
        }
        
        LoggingUtil.info("SummarizerWorkflow", f"워크플로우 시작 (Iteration {initial_state['iteration']})")
        final_state = self.workflow.invoke(initial_state)
        LoggingUtil.info("SummarizerWorkflow", f"워크플로우 완료")
        
        return self.finalize_result(final_state)

    def summarize_requirements(self, state: SummarizerState) -> Dict:
        """
        요구사항 텍스트를 요약
        """
        requirements = state["requirements"]
        iteration = state["iteration"]
        
        LoggingUtil.info("SummarizerWorkflow", f"요약 시작 (Iteration {iteration})")
        
        # 요구사항 언어 감지 (간단한 휴리스틱)
        has_korean = any('\uac00' <= c <= '\ud7a3' for c in requirements[:500])
        language = "Korean" if has_korean else "English"

        # 라인 수 계산
        total_lines = len(requirements.splitlines())

        prompt = f"""An AI agent that summarizes the following requirements by grouping related items and tracking source line references.

Your primary goal is to synthesize and consolidate, not just rephrase.
Analyze all the requirements provided below. Identify requirements that deal with similar features, subjects, or data.
Group these related requirements together.
For each group, create a single, concise summary sentence that captures the essence of that group.

The requirements text is provided with line numbers in the format "lineNumber: content".

Requirements with line numbers:
{requirements}

Output in the following JSON format:
{{
    "summarizedRequirements": [
        {{
            "text": "A summary sentence representing a group of related requirements.",
            "source_lines": [1, 5, 8] // All source line numbers that contributed to this summary.
        }},
        {{
            "text": "Another summary sentence for a different group.", 
            "source_lines": [2, 3, 4] // All relevant source line numbers.
        }}
    ]
}}

Guidelines:
- CRITICAL: The 'source_lines' array must only contain line numbers that are present in the 'Requirements with line numbers' section. For the provided text, the valid range of line numbers is from 1 to {total_lines}. Do not under any circumstances invent or use line numbers outside this range.
- CRITICAL: Avoid rephrasing single lines. The goal is to produce significantly fewer summary sentences than the number of input lines by grouping related topics.
- The final summary should be less than 50% of the original text's length.
- Each summary sentence must reference multiple source lines where possible.
- Ensure all important functional requirements from the original text are covered in the summaries.
- DDL statements should be ignored and not included in the summary.

<language_guide>Please generate the response in {language} while ensuring that all code elements (e.g., variable names, function names) remain in English.</language_guide>. Please generate the json in valid json format and if there's a property its value is null, don't contain the property. also, Please return only the json without any natural language.
"""
        
        try:
            # POSCO P-GPT 게이트웨이가 openai SDK 의 streaming 요청에 404 를 돌려주므로
            # non-streaming(invoke) 으로 호출. 자세한 배경은 user_story_generator 참조.
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content=prompt)]

            response = self.llm.invoke(messages).content
            
            response_clean = self._extract_json(response)
            result_data = json.loads(response_clean)
            
            LoggingUtil.info("SummarizerWorkflow", f"요약 완료: {len(result_data.get('summarizedRequirements', []))}개")
            
            return {
                "summarizedRequirements": result_data.get("summarizedRequirements", []),
                "progress": 50,
                "logs": state["logs"] + [{"timestamp": datetime.now().isoformat(), "message": "요약 생성 완료"}]
            }
        except Exception as e:
            LoggingUtil.exception("SummarizerWorkflow", "요약 생성 중 오류 발생", e)
            return {
                "summarizedRequirements": [],
                "error": str(e),
                "progress": 50,
                "logs": state["logs"] + [{"timestamp": datetime.now().isoformat(), "message": f"요약 생성 오류: {str(e)}"}]
            }

    def finalize(self, state: SummarizerState) -> Dict:
        """
        최종 결과 정리
        """
        LoggingUtil.info("SummarizerWorkflow", "요약 워크플로우 최종 정리")
        
        final_summaries = []
        for summary in state.get("summarizedRequirements", []):
            # Frontend의 refs 형식으로 변환
            source_lines = summary.get("source_lines", [])
            refs = [[[line, 1], [line, -1]] for line in source_lines]
            final_summaries.append({
                "text": summary.get("text", ""),
                "refs": refs,
                "source_lines": source_lines # 원본 source_lines도 유지
            })

        return {
            "summarizedRequirements": final_summaries,
            "progress": 100,
            "isCompleted": True,
            "logs": state["logs"] + [{"timestamp": datetime.now().isoformat(), "message": "요약 워크플로우 완료"}]
        }

    def finalize_result(self, state: SummarizerState) -> Dict:
        """
        최종 결과를 반환하기 전에 필요한 후처리
        """
        return {
            "summarizedRequirements": state.get("summarizedRequirements", []),
            "originalRequirements": state.get("requirements", ""),
            "refs": state.get("summarizedRequirements", []),  # Frontend 호환성
            "progress": state.get("progress", 0),
            "logs": state.get("logs", []),
            "isCompleted": state.get("isCompleted", False),
            "error": state.get("error", "")
        }

    def _extract_json(self, text: str) -> str:
        """LLM 응답에서 JSON 부분만 추출"""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

