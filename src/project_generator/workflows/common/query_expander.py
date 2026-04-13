"""
Query Expander
검색 쿼리를 LLM으로 확장하여 semantic_text와 유사한 구조로 만듦

목적:
- 인덱싱된 semantic_text와 유사도 검색 성능 향상
- "Customer" → "Customer order 주문 customer_order m_odr 주문 도메인 표준 prefix"
"""
from typing import Optional, Dict
from src.project_generator.config import Config
from src.project_generator.utils.logging_util import LoggingUtil
from src.project_generator.utils.llm_factory import create_chat_llm


class QueryExpander:
    """
    검색 쿼리 확장기
    LLM을 사용하여 단순 키워드를 의미가 풍부한 쿼리로 확장
    """
    
    def __init__(self):
        """초기화"""
        self.llm = create_chat_llm(
            model=Config.DEFAULT_LLM_MODEL,
            temperature=0.2,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
    
    def expand(
        self,
        base_keyword: str,
        category: str,
        context: Optional[Dict] = None,
        bounded_context: Optional[Dict] = None
    ) -> str:
        """
        검색 쿼리 확장
        
        목적:
        - 단순 키워드("Customer")를 semantic_text와 비슷한 구조로 확장
        - 인덱싱된 semantic_text와의 유사도 향상
        
        Args:
            base_keyword: 기본 키워드 (예: "Customer", "customer_id")
            category: 카테고리 ("table_name" 또는 "column_name")
            context: 추가 컨텍스트 (aggregate_alias 등)
            bounded_context: Bounded Context 정보
            
        Returns:
            확장된 쿼리 문자열
        """
        if not base_keyword or not base_keyword.strip():
            return ""
        
        # 컨텍스트 정보 추출
        domain_hint = None
        aggregate_alias = None
        if bounded_context:
            domain_hint = bounded_context.get("domain")
        if context:
            aggregate_alias = context.get("aggregate_alias")
        
        # 프롬프트 구성
        if category == "table_name":
            prompt = self._build_table_name_expansion_prompt(
                base_keyword, domain_hint, aggregate_alias
            )
        else:  # column_name
            prompt = self._build_column_name_expansion_prompt(
                base_keyword, domain_hint, aggregate_alias
            )
        
        try:
            response = self.llm.invoke(prompt)
            expanded_query = response.content.strip()
            
            # 따옴표 제거
            expanded_query = expanded_query.strip('"').strip("'").strip()
            
            LoggingUtil.info("QueryExpander", 
                           f"🔍 쿼리 확장: '{base_keyword}' → '{expanded_query[:100]}...' ({category})")
            
            return expanded_query
        except Exception as e:
            LoggingUtil.warning("QueryExpander", 
                              f"⚠️  쿼리 확장 실패: {e}, 원본 키워드 사용: '{base_keyword}'")
            return base_keyword
    
    def _build_table_name_expansion_prompt(
        self,
        base_keyword: str,
        domain_hint: Optional[str],
        aggregate_alias: Optional[str]
    ) -> str:
        """테이블명 쿼리 확장 프롬프트"""
        context_info = ""
        if domain_hint:
            context_info += f"\n도메인 힌트: {domain_hint}"
        if aggregate_alias:
            context_info += f"\nAggregate 별칭: {aggregate_alias}"
        
        return f"""다음 키워드를 기반으로 회사 표준 도메인/테이블 검색에 적합한 확장 쿼리를 생성하세요.

기본 키워드: {base_keyword}
{context_info}

요구사항 (중요: 인덱싱된 semantic_text 형식과 일치):
1. 핵심 키워드를 반복 포함: "{base_keyword} [한글명] [표준명]" 형식으로 시작
2. 간결하게 작성: 50-100자 내외
3. 형식: "{base_keyword} [한글명] [표준명]는 [도메인] 도메인의 table_name이다" (semantic_text와 동일한 형식)
4. 불필요한 설명 제거: "도메인", "표준", "prefix" 같은 메타 키워드 제거
5. 인덱싱된 semantic_text와 정확히 유사한 구조로 작성

인덱싱된 semantic_text 예시:
"고객정보 CustomerCore m_cst는 CST 도메인의 table_name이다."
"주문이력 OrderHistory m_odr_hist는 주문 이력을 관리하는 ODR 도메인 테이블이다."

입력: "{base_keyword}"
출력: (semantic_text와 동일한 형식으로)"""
    
    def _build_column_name_expansion_prompt(
        self,
        base_keyword: str,
        domain_hint: Optional[str],
        aggregate_alias: Optional[str]
    ) -> str:
        """컬럼명 쿼리 확장 프롬프트"""
        context_info = ""
        if domain_hint:
            context_info += f"\n도메인 힌트: {domain_hint}"
        if aggregate_alias:
            context_info += f"\nAggregate 별칭: {aggregate_alias}"
        
        return f"""다음 키워드를 기반으로 회사 표준 필드/컬럼 검색에 적합한 확장 쿼리를 생성하세요.

기본 키워드: {base_keyword}
{context_info}

요구사항 (중요: 인덱싱된 semantic_text 형식과 일치):
1. 핵심 키워드를 반복 포함: "{base_keyword} [한글명] [표준명]" 형식으로 시작
2. 간결하게 작성: 30-70자 내외
3. 형식: "{base_keyword} [한글명] [표준명]는 [용도] 필드이다" (semantic_text와 동일한 형식)
4. 불필요한 설명 제거: "필드", "컬럼", "표준" 같은 메타 키워드 제거
5. 인덱싱된 semantic_text와 정확히 유사한 구조로 작성

인덱싱된 semantic_text 예시:
"주문식별자 order_id fld_order_id는 주문을 식별하는 필드이다."

입력: "{base_keyword}"
출력: (semantic_text와 동일한 형식으로)"""

