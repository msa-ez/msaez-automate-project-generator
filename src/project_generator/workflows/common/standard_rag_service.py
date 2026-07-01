"""
표준 RAG 검색 서비스
회사 표준(DB 테이블명/컬럼명/API/용어)을 카테고리별로 명확히 구분하여 검색

핵심 아이디어:
- "테이블명 표준" vs "필드명 표준" vs "API 표준" vs "용어 표준"은 명확히 다른 타입
- 메타데이터(category, domain 등)로 검색 영역을 먼저 좁힌 뒤
- 그 안에서만 임베딩 유사도 검색
"""
from typing import Literal, TypedDict, Optional, List, Dict
from dataclasses import dataclass
from pathlib import Path

from src.project_generator.workflows.common.rag_retriever import RAGRetriever, DEFAULT_SIM_THRESHOLD
from src.project_generator.utils.logging_util import LoggingUtil

# 표준 카테고리 정의
StandardCategory = Literal[
    "table_name",    # 테이블명 표준 (T_ODR_M, T_CPN_M 등)
    "column_name",   # 컬럼명 표준 (ODR_ID, CRT_DTM 등)
    "api_path",      # API 경로 표준 (/v1/odr/crt 등)
    "terminology"    # 도메인 용어 표준 (쿠폰, 알림, 정산 등)
]

StandardType = Literal[
    "database_standard",
    "api_standard",
    "terminology_standard"
]

class StandardQuery(TypedDict):
    """
    "무엇을 찾을지"를 표현하는 검색 쿼리 단위.
    
    - query: 실제 RAG에 던질 텍스트 (가능하면 짧은 키워드)
    - domain_hint: ODR/CPN/NTF 등 도메인 힌트 (선택사항)
    """
    query: str
    domain_hint: Optional[str]

class StandardSearchResult(TypedDict):
    """
    표준 검색 결과 공통 포맷.
    
    - content: 자연어 텍스트 (LLM 프롬프트에 넣을 본문)
    - metadata: 원본 메타데이터 (type, category, source, structured_data 등)
    - score: 유사도 (0.0~1.0, optional)
    """
    content: str
    metadata: Dict
    score: Optional[float]


@dataclass
class StandardRAGService:
    """
    회사 표준 전용 RAG 검색 서비스.
    
    역할:
    - category/type/domain_hint 기반으로 VectorStore 검색 필터 구성
    - 각 카테고리(table_name, column_name, api_path, terminology)에 맞는 검색 메서드 제공
    - 표준 JSON은 인덱싱 시 메타데이터에 type/category/domain을 이미 넣어준다는 전제
    """
    retriever: RAGRetriever
    score_threshold: float = DEFAULT_SIM_THRESHOLD
    k_per_query: int = 1  # top-k 1개만 반환
    
    # ---- Public API: 단일 검색 메서드 (카테고리 구분 없음) ------------------------
    
    def search_standards(
        self,
        query: str,
        domain_hint: Optional[str] = None,
        k: Optional[int] = None,
        transformation_session_id: Optional[str] = None
    ) -> List[StandardSearchResult]:
        """
        표준 검색 (카테고리 구분 없음):
        - 모든 표준 문서에서 유사도 검색
        - transformation_session_id: (선택) 현재 세션의 문서만 검색
        """
        return self._search_with_metadata(
            query=query,
            type_filter=None,  # 모든 타입
            category_filter=None,  # 모든 카테고리
            domain_hint=domain_hint,
            k=k or self.k_per_query,
            transformation_session_id=transformation_session_id
        )
    
    # ---- 내부 공통 로직 -------------------------------------------
    
    def _search_with_metadata(
        self,
        query: str,
        type_filter: Optional[StandardType] = None,
        category_filter: Optional[StandardCategory] = None,
        domain_hint: Optional[str] = None,
        k: int = 1,
        transformation_session_id: Optional[str] = None,
    ) -> List[StandardSearchResult]:
        """
        공통 검색 로직:
        - type, category, domain_hint로 Chroma filter 구성
        - RAGRetriever의 VectorStore에서 similarity_search_with_score 호출
        - 점수 → 유사도 변환 로직은 RAGRetriever 쪽 구현을 재사용
        """
        if not self.retriever._initialized or not self.retriever.vectorstore:
            return []
        
        vs = self.retriever.vectorstore
        
        # 모든 제약 제거: type, category, domain 등 필터 없이 검색
        # 키워드 쿼리로 내용을 검색하고 가장 높은 1개의 구조화된 데이터만 사용
        # ChromaDB는 빈 필터 {}를 허용하지 않으므로 None으로 설정
        metadata_filter = None
        
        # 검색 범위 설정:
        # - k=1 (top-1 채택)이면 search_k=3 (3개 후보 중에서 1개 선택)
        # - k=3 (top-3 채택)이면 search_k=9 (9개 후보 중에서 3개 선택)
        # 이유: 충분한 후보를 가져온 후 유사도 순으로 정렬하여 top-k만 선택
        # transformation_session_id가 있으면 더 많이 가져와서 수동 필터링
        # k=1일 때 search_k=5는 과도하므로 k * 3으로 조정
        search_k = k * 3 if transformation_session_id else k * 2
        results_with_scores = []
        
        try:
            # 필터 없이 먼저 검색해서 문서가 있는지 확인 (디버깅용)
            all_results = vs.similarity_search_with_score(query, k=min(20, search_k))
            
            # 필터 없이 검색 (metadata_filter=None)
            results_with_scores = vs.similarity_search_with_score(
                query,
                k=search_k,
                filter=metadata_filter
            )
            
            if not results_with_scores:
                if all_results:
                    print(f"   ❌ 유사도 검색: 검색 결과 없음 (쿼리: '{query[:50]}...', 필터 없이 {len(all_results)}개 발견, 필터 적용 후 0개)")
                else:
                    print(f"   ❌ 유사도 검색: 검색 결과 없음 (쿼리: '{query[:50]}...', 요청: {search_k}개)")
                return []
            
            # transformation_session_id 수동 필터링
            # 기본 표준 문서(has_draft_context=False)는 항상 포함
            # 초안 정보 포함 문서(has_draft_context=True)는 현재 세션의 것만 포함
            if transformation_session_id:
                filtered_results = []
                for doc, score in results_with_scores:
                    doc_metadata = doc.metadata
                    has_draft_context = doc_metadata.get("has_draft_context", False)
                    
                    if has_draft_context:
                        # 초안 정보 포함 문서는 현재 세션의 것만 허용
                        doc_session_id = doc_metadata.get("transformation_session_id")
                        if doc_session_id != transformation_session_id:
                            continue  # 다른 세션의 문서 제외
                    
                    # 기본 표준 문서(has_draft_context=False)는 항상 포함
                    filtered_results.append((doc, score))
                
                results_with_scores = filtered_results
                
                if not results_with_scores:
                    print(f"   ❌ 유사도 검색: 세션 필터링 후 결과 없음")
        except Exception as e:
            # 필터 실패 시 필터 없이 재시도
            # ChromaDB 동시성 문제("Failed to get segments")는 일시적이므로 조용히 처리
            error_msg = str(e)
            if "Failed to get segments" not in error_msg:
                print(f"⚠️  Search failed with filter: {e}")
            try:
                # 필터 없이 검색 (모든 제약 제거)
                results_with_scores = vs.similarity_search_with_score(query, k=search_k)
                # transformation_session_id 필터링만 수행 (선택사항)
                if transformation_session_id:
                    filtered_results = []
                    for doc, score in results_with_scores:
                        doc_metadata = doc.metadata
                        has_draft_context = doc_metadata.get("has_draft_context", False)
                        if has_draft_context:
                            if doc_metadata.get("transformation_session_id") != transformation_session_id:
                                continue
                        filtered_results.append((doc, score))
                    results_with_scores = filtered_results
            except Exception as e2:
                error_msg2 = str(e2).lower()
                if "Failed to get segments" not in error_msg2:
                    print(f"⚠️  Search failed: {e2}")
                # "no such table: collections" 오류 감지 시 자동 복구 시도
                if "no such table" in error_msg2 or "collections" in error_msg2 or "database" in error_msg2:
                    print(f"⚠️  Vector Store database corrupted during search. Attempting to repair...")
                    try:
                        if self.retriever._repair_vectorstore():
                            print(f"✅ Vector Store repaired. Retrying search...")
                            # 복구 후 재시도 (1회만)
                            try:
                                results_with_scores = vs.similarity_search_with_score(query, k=search_k)
                                if transformation_session_id:
                                    filtered_results = []
                                    for doc, score in results_with_scores:
                                        doc_metadata = doc.metadata
                                        has_draft_context = doc_metadata.get("has_draft_context", False)
                                        if has_draft_context:
                                            if doc_metadata.get("transformation_session_id") != transformation_session_id:
                                                continue
                                        filtered_results.append((doc, score))
                                    results_with_scores = filtered_results
                            except Exception as retry_error:
                                print(f"⚠️  Search still failed after repair: {retry_error}")
                                return []
                        else:
                            print(f"⚠️  Vector Store repair failed. Returning empty results.")
                            return []
                    except Exception as repair_error:
                        print(f"⚠️  Failed to repair Vector Store: {repair_error}")
                        return []
                else:
                    return []
        
        candidates: List[StandardSearchResult] = []
        filtered_count = 0  # 필터링된 결과 수 추적
        
        if not results_with_scores:
            print(f"   [DEBUG] 검색 결과 없음: query='{query[:50]}...'")
            return []
        
        # distance -> similarity 변환 (0~2 거리 → 0~1 유사도)
        distances = [abs(float(s)) for _, s in results_with_scores]
        if distances:
            dist_max = max(distances)
            dist_min = min(distances)
            distance_range = 2.0 if dist_max > 1.0 else 1.0
        else:
            distance_range = 2.0
        
        max_similarity = 0.0
        min_similarity = 1.0
        for doc, raw_score in results_with_scores:
            doc_metadata = doc.metadata
            
            # transformation_session_id 필터링 (수동 필터링)
            if transformation_session_id:
                has_draft_context = doc_metadata.get("has_draft_context", False)
                if has_draft_context:
                    doc_session_id = doc_metadata.get("transformation_session_id")
                    if doc_session_id != transformation_session_id:
                        continue
            
            distance = abs(float(raw_score))
            
            # ChromaDB cosine distance 변환
            if distance_range == 2.0:
                similarity = max(0.0, 1.0 - (distance / 2.0))
            else:
                similarity = max(0.0, 1.0 - distance)
            
            # 최고/최저 유사도 추적
            if similarity > max_similarity:
                max_similarity = similarity
            if similarity < min_similarity:
                min_similarity = similarity
            
            # 임계값 체크 제거: 모든 결과를 candidates에 추가
            candidates.append(
                StandardSearchResult(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=similarity
                )
            )
        
        # 유사도 내림차순 정렬 후 top-k만 반환 (임계값 무관)
        candidates.sort(key=lambda x: x.get("score") or 0.0, reverse=True)
        result = candidates[:k]
        
        # 쿼리별 상세 로그는 제거 (요약 로그만 표시)
        # 필요시 디버깅을 위해 주석 해제 가능
        
        return result

