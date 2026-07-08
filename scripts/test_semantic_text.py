#!/usr/bin/env python3
"""
semantic_text 생성 테스트 스크립트
StandardLoader의 LLM 기반 semantic_text 생성 기능 테스트
"""
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# .env 파일 로드
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=project_root / '.env')
except ImportError:
    print("⚠️  python-dotenv not installed. Environment variables may not be loaded.")

import pandas as pd
from src.project_generator.workflows.common.standard_loader import StandardLoader
from src.project_generator.config import Config


def test_semantic_text_generation():
    """semantic_text 생성 테스트"""
    print("🧪 Testing semantic_text generation...")
    print(f"🤖 LLM Model: {Config.DEFAULT_LLM_MODEL}")
    print()
    
    # StandardLoader 초기화 (LLM 활성화)
    loader = StandardLoader(enable_llm=True)
    
    if not loader.enable_llm:
        print("❌ LLM is not enabled. Check LLM initialization.")
        return False
    
    print("✅ StandardLoader initialized with LLM")
    print()
    
    # 테스트용 엑셀 파일 경로
    standards_path = Config.COMPANY_STANDARDS_PATH
    excel_file = standards_path / "table_field_standards.xlsx"
    
    if not excel_file.exists():
        print(f"❌ Excel file not found: {excel_file}")
        return False
    
    print(f"📄 Reading Excel file: {excel_file}")
    print()
    
    try:
        # 엑셀 파일 읽기
        excel_file_obj = pd.ExcelFile(excel_file)
        
        # 각 시트별로 테스트
        for sheet_name in excel_file_obj.sheet_names:
            print(f"📊 Testing sheet: {sheet_name}")
            print("-" * 60)
            
            df = pd.read_excel(excel_file_obj, sheet_name=sheet_name)
            
            if df.empty:
                print("  ⚠️  Empty sheet, skipping...")
                print()
                continue
            
            # 첫 3개 행만 테스트
            test_rows = min(3, len(df))
            
            for idx in range(test_rows):
                row = df.iloc[idx]
                print(f"\n  Row {idx + 1}:")
                
                # semantic_text 생성 테스트
                text, structured_data = loader._format_excel_row_as_standard_text(row, sheet_name)
                
                print(f"    한글명: {structured_data.get('korean_name', 'N/A')}")
                print(f"    영문명: {structured_data.get('english_name', 'N/A')}")
                print(f"    표준명: {structured_data.get('table_name', 'N/A')}")
                print(f"    카테고리: {structured_data.get('category', 'N/A')}")
                print()
                print(f"    📝 Generated semantic_text:")
                print(f"    {text}")
                print()
                
                # semantic_text가 생성되었는지 확인
                if text and text.strip():
                    if "내부 표준명" in text or "표준명" in text:
                        print(f"    ✅ semantic_text generated successfully!")
                    else:
                        print(f"    ⚠️  semantic_text generated but may not be in expected format")
                else:
                    print(f"    ❌ semantic_text is empty!")
                
                print("-" * 60)
            
            print()
    
    except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, OSError, ImportError) as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✅ Test completed!")
    return True


if __name__ == '__main__':
    success = test_semantic_text_generation()
    sys.exit(0 if success else 1)

