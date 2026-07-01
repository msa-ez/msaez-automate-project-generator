#!/usr/bin/env python3
"""
Vector Store 조회 스크립트
인덱싱된 표준 문서를 검색하고 조회
"""
import sys
from pathlib import Path
import json

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# .env 파일 로드
from dotenv import load_dotenv
load_dotenv(dotenv_path=project_root / '.env')

from src.project_generator.workflows.common.rag_retriever import RAGRetriever
from src.project_generator.config import Config


def list_all_documents(category_filter: str = None):
    """Vector Store에 저장된 모든 문서 목록 조회"""
    retriever = RAGRetriever()
    
    if not retriever._initialized or not retriever.vectorstore:
        print("❌ Vector Store가 초기화되지 않았습니다.")
        return
    
    try:
        # ChromaDB에서 모든 문서 가져오기
        collection = retriever.vectorstore._collection
        count = collection.count()
        
        print(f"📊 Vector Store에 저장된 문서 수: {count}개")
        if category_filter:
            print(f"🔍 카테고리 필터: {category_filter}\n")
        else:
            print()
        
        if count == 0:
            print("⚠️  저장된 문서가 없습니다.")
            return
        
        # 필터 적용 여부에 따라 가져오기
        if category_filter:
            results = collection.get(
                limit=count,
                where={"category": category_filter}
            )
        else:
            results = collection.get(limit=count)
        
        print("=" * 80)
        print("📚 저장된 문서 목록 (전체):")
        print("=" * 80)
        
        # 실제 필터링된 결과 개수
        filtered_count = len(results.get('ids', []))
        
        for i, (doc_id, metadata, document) in enumerate(zip(
            results.get('ids', []),
            results.get('metadatas', []),
            results.get('documents', [])
        ), 1):
            print(f"\n[{i}/{filtered_count}] ID: {doc_id}")
            print(f"    출처: {Path(metadata.get('source', '')).name}")
            if metadata.get('sheet'):
                print(f"    시트: {metadata.get('sheet')}")
            if metadata.get('section'):
                print(f"    섹션: {metadata.get('section')}")
            
            # 문서 내용 전체 표시
            print(f"    내용:")
            # 내용이 길면 줄바꿈하여 표시
            content_lines = document.split('\n')
            if len(content_lines) > 10:
                # 처음 10줄 + 마지막 3줄 표시
                for line in content_lines[:10]:
                    print(f"      {line}")
                print(f"      ... (중간 {len(content_lines) - 13}줄 생략) ...")
                for line in content_lines[-3:]:
                    print(f"      {line}")
            else:
                for line in content_lines:
                    print(f"      {line}")
            
            # 구조화된 데이터가 있으면 표시
            if metadata.get('structured_data'):
                try:
                    structured = json.loads(metadata.get('structured_data'))
                    print(f"    구조화된 데이터:")
                    print(f"    {json.dumps(structured, ensure_ascii=False, indent=4)}")
                except Exception as e:
                    print(f"    구조화된 데이터 (파싱 실패): {metadata.get('structured_data')[:200]}...")
        
        print(f"\n{'=' * 80}")
        if category_filter:
            print(f"✅ 총 {filtered_count}개 문서 조회 완료 (카테고리: {category_filter}, 전체: {count}개)")
        else:
            print(f"✅ 총 {filtered_count}개 문서 조회 완료")
        print(f"{'=' * 80}")
        
    except Exception as e:
        print(f"❌ 조회 실패: {e}")
        import traceback
        traceback.print_exc()


def search_documents(query: str, k: int = 5):
    """쿼리로 문서 검색"""
    retriever = RAGRetriever()
    
    if not retriever._initialized or not retriever.vectorstore:
        print("❌ Vector Store가 초기화되지 않았습니다.")
        return
    
    print(f"🔍 검색 쿼리: '{query}'")
    print(f"📊 반환할 결과 수: {k}개\n")
    
    try:
        results = retriever.search_company_standards(query, k=k)
        
        if not results:
            print("⚠️  검색 결과가 없습니다.")
            return
        
        print("=" * 80)
        print(f"📚 검색 결과 ({len(results)}개):")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            content = result.get("content", "")
            metadata = result.get("metadata", {})
            
            print(f"\n[{i}] 출처: {Path(metadata.get('source', '')).name}")
            print(f"    내용:")
            print(f"    {content}")
            
            if metadata.get('structured_data'):
                try:
                    structured = json.loads(metadata.get('structured_data'))
                    print(f"    구조화된 데이터:")
                    print(f"    {json.dumps(structured, ensure_ascii=False, indent=4)}")
                except:
                    pass
        
    except Exception as e:
        print(f"❌ 검색 실패: {e}")
        import traceback
        traceback.print_exc()


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Vector Store 조회 및 검색')
    parser.add_argument(
        '--list',
        action='store_true',
        help='모든 문서 목록 조회'
    )
    parser.add_argument(
        '--search',
        type=str,
        default=None,
        help='검색 쿼리'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='검색 결과 수 (기본: 5)'
    )
    parser.add_argument(
        '--category',
        type=str,
        default=None,
        help='카테고리 필터 (예: table_name, column_name)'
    )
    
    args = parser.parse_args()
    
    print("🚀 Vector Store 조회 도구")
    print(f"📁 Vector Store 경로: {Config.VECTORSTORE_PATH}\n")
    
    if args.list:
        list_all_documents(category_filter=args.category)
    elif args.search:
        search_documents(args.search, args.k)
    else:
        print("사용법:")
        print("  모든 문서 목록: python scripts/query_vectorstore.py --list")
        print("  카테고리별 목록: python scripts/query_vectorstore.py --list --category table_name")
        print("  검색: python scripts/query_vectorstore.py --search 'Order aggregate table naming standard'")
        print("  검색 (결과 수 지정): python scripts/query_vectorstore.py --search 'Order' --k 10")


if __name__ == '__main__':
    main()

