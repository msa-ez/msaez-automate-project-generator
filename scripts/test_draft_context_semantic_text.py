"""
초안 정보를 포함한 semantic_text 생성 테스트
"""
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.project_generator.workflows.common.standard_loader import StandardLoader
from src.project_generator.config import Config
import pandas as pd
import json

def test_draft_context_semantic_text():
    """초안 정보를 포함한 semantic_text 생성 테스트"""
    
    # StandardLoader 초기화 (LLM 활성화)
    loader = StandardLoader(enable_llm=True)
    
    # 표준 문서 경로
    standards_path = Config.COMPANY_STANDARDS_PATH
    
    if not standards_path.exists():
        print(f"❌ 표준 문서를 찾을 수 없습니다: {standards_path}")
        return
    
    # 초안 정보 구성 (예시: 상점 Aggregate)
    draft_context = {
        "bounded_context": {
            "name": "StoreManagement",
            "domain": "STR",
            "alias": "상점관리"
        },
        "aggregates": [
            {
                "alias": "상점",
                "name": "Store",
                "previewAttributes": [
                    {"fieldName": "store_id"},
                    {"fieldName": "store_name"},
                    {"fieldName": "address"},
                    {"fieldName": "is_receiving_orders"},
                    {"fieldName": "created_at"}
                ],
                "valueObjects": [
                    {"alias": "영업시간", "name": "OperatingHours"},
                    {"alias": "브레이크타임", "name": "BreakTime"}
                ]
            }
        ]
    }
    
    # 엑셀 파일 읽기
    try:
        excel_file = pd.ExcelFile(standards_path)
        
        # 테이블표준 시트 찾기
        table_sheet = None
        for sheet_name in excel_file.sheet_names:
            if "테이블" in sheet_name or "table" in sheet_name.lower():
                table_sheet = sheet_name
                break
        
        if not table_sheet:
            print("❌ 테이블표준 시트를 찾을 수 없습니다.")
            return
        
        print(f"📄 시트 '{table_sheet}' 읽는 중...")
        df = pd.read_excel(excel_file, sheet_name=table_sheet)
        
        if df.empty:
            print("❌ 시트가 비어있습니다.")
            return
        
        # "상점" 또는 "Store" 관련 행 찾기
        test_rows = []
        for idx, row in df.iterrows():
            row_str = row.astype(str).str.lower().str.cat(sep=' ')
            if '상점' in row_str or 'store' in row_str.lower():
                test_rows.append((idx, row))
                if len(test_rows) >= 3:  # 최대 3개만
                    break
        
        if not test_rows:
            print("❌ '상점' 또는 'Store' 관련 행을 찾을 수 없습니다.")
            print("   첫 번째 행으로 테스트합니다...")
            test_rows = [(0, df.iloc[0])]
        
        print(f"\n{'='*80}")
        print(f"🧪 초안 정보를 포함한 semantic_text 생성 테스트")
        print(f"{'='*80}\n")
        
        # 각 행 테스트
        for row_idx, row in test_rows:
            print(f"\n--- 테스트 {row_idx + 1}: Row {row_idx} ---")
            
            # 1. 초안 정보 없이 semantic_text 생성
            print("\n[1] 초안 정보 없이 semantic_text 생성:")
            text_without_draft, structured_data_without = loader._format_excel_row_as_standard_text(
                row, 
                table_sheet,
                draft_context=None
            )
            
            semantic_text_without = structured_data_without.get('semantic_text', '')
            if semantic_text_without:
                print(f"✅ semantic_text ({len(semantic_text_without)} chars):")
                print(f"   {semantic_text_without[:200]}...")
            else:
                print("⚠️  semantic_text가 생성되지 않았습니다.")
            
            # 2. 초안 정보 포함하여 semantic_text 생성
            print("\n[2] 초안 정보 포함하여 semantic_text 생성:")
            text_with_draft, structured_data_with = loader._format_excel_row_as_standard_text(
                row, 
                table_sheet,
                draft_context=draft_context
            )
            
            semantic_text_with = structured_data_with.get('semantic_text', '')
            if semantic_text_with:
                print(f"✅ semantic_text ({len(semantic_text_with)} chars):")
                print(f"   {semantic_text_with[:200]}...")
            else:
                print("⚠️  semantic_text가 생성되지 않았습니다.")
            
            # 3. 비교
            print("\n[3] 비교:")
            if semantic_text_without and semantic_text_with:
                if semantic_text_without != semantic_text_with:
                    print("✅ 초안 정보가 semantic_text에 반영되었습니다!")
                    print(f"   차이점:")
                    print(f"   - 초안 정보 없음: {len(semantic_text_without)} chars")
                    print(f"   - 초안 정보 포함: {len(semantic_text_with)} chars")
                    
                    # 초안 키워드 포함 여부 확인
                    draft_keywords = ['store_name', 'store_id', '영업시간', 'OperatingHours', '브레이크타임', 'BreakTime']
                    found_keywords = [kw for kw in draft_keywords if kw in semantic_text_with]
                    if found_keywords:
                        print(f"   - 초안 키워드 포함: {', '.join(found_keywords)}")
                else:
                    print("⚠️  초안 정보가 semantic_text에 반영되지 않았습니다.")
            else:
                print("⚠️  semantic_text를 비교할 수 없습니다.")
            
            print("\n" + "-"*80)
        
        print(f"\n{'='*80}")
        print("✅ 테스트 완료")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_draft_context_semantic_text()

