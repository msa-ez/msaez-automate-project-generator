"""
RefsTraceUtil - 프론트엔드 RefsTraceUtil.js와 동일한 기능
refs 변환 및 클램핑 처리
"""
import re
from typing import List, Dict, Any, Union


class RefsTraceUtil:
    """프론트엔드 RefsTraceUtil과 동일한 기능 제공"""
    
    @staticmethod
    def sanitize_and_convert_refs(data: Any, line_numbered_requirements: str, is_use_xml_base: bool = True) -> Any:
        """
        프론트엔드 RefsTraceUtil.sanitizeAndConvertRefs와 동일
        phrase → index 변환 + 클램핑 처리
        """
        if not data or not line_numbered_requirements:
            return data
        
        # minLine, maxLine 찾기
        min_line, max_line = RefsTraceUtil._get_line_number_range(line_numbered_requirements, is_use_xml_base)
        
        # 라인 컨텐츠 추출 (XML 태그 제거)
        lines = line_numbered_requirements.split('\n')
        line_contents = []
        for line in lines:
            if is_use_xml_base:
                match = re.match(r'^<\d+>(.*)</\d+>$', line)
                if match:
                    line_contents.append(match.group(1))
                else:
                    line_contents.append(line)
            else:
                idx = line.find(': ')
                line_contents.append(line[idx + 2:] if idx != -1 else line)
        
        # 클램핑 함수
        def clamp(n, lo, hi):
            return max(lo, min(hi, n))
        
        # 1단계: refs의 구조에 맞지 않은 데이터 제거
        filtered_data = RefsTraceUtil._filter_invalid_refs(data)
        
        # 2단계: refs의 라인번호가 실제 구문을 포함하는지 검증 및 가능한 경우 교정 작업
        sanitized_data = RefsTraceUtil._search_refs_array_recursively(
            filtered_data, 
            lambda refs_array: RefsTraceUtil._sanitize_refs_array(refs_array, line_contents, min_line, max_line, clamp)
        )
        
        # 3단계: convertRefsToIndexes로 구문을 실제 칼럼 좌표로 변환
        try:
            converted_data = RefsTraceUtil.convert_refs_to_indexes(
                sanitized_data, 
                line_numbered_requirements, 
                is_use_xml_base
            )
        except (OSError, ValueError, TypeError, LookupError, AttributeError, RuntimeError, ImportError, ArithmeticError, AssertionError, StopIteration, StopAsyncIteration, BufferError) as e:
            import logging
            logging.warning(f'Failed to convert refs to indexes: {e}')
            converted_data = sanitized_data
        
        # 4단계: 최종 클램핑 처리
        final_data = RefsTraceUtil._search_refs_array_recursively(
            converted_data,
            lambda refs_array: RefsTraceUtil._clamp_refs_array(refs_array, line_contents, min_line, max_line, clamp)
        )

        # 5단계: 구조 라인 ref relocate — LLM 이 헤더 / TOC 표 row 를 anchor 로 잡은 경우
        # 같은 user story 의 본문 narrative 라인으로 ref 위치만 옮긴다. zero-length 와
        # 비-user-story 헤더 / FR-ID 없는 구분선 / 빈 라인은 drop. 본문 라인은 그대로.
        final_data = RefsTraceUtil._search_refs_array_recursively(
            final_data,
            lambda refs_array: RefsTraceUtil._relocate_structural_refs(refs_array, line_contents)
        )

        return final_data

    @staticmethod
    def _relocate_structural_refs(refs_array: List, line_contents: List[str]) -> List:
        """구조 라인 ref (헤더 / TOC 표 row) 를 같은 user story 의 본문으로 옮긴다.

        - zero-length / 빈줄 / 구분선 → drop
        - 본문 라인 → 그대로
        - markdown 헤더 / 표 row 에서 [PROJ-US-FR/NFR-XXX] 추출됨 → 그 user story 의
          `##### [...]` 본문 narrative ('> *As a ...') 또는 첫 content 라인으로 relocate
        - 헤더/표 row 인데 FR-ID 없음 → drop
        """
        if not refs_array or not line_contents:
            return refs_array

        # user story 본문 위치 인덱스 (1-회 계산)
        us_index = {}
        us_header_re = re.compile(r'^\s*#{4,6}\s+\[([A-Za-z][\w-]*US-(?:FR|NFR)-\d+)\]')
        narrative_re = re.compile(r'^\s*>\s*\*?\s*As a')
        for i, ln in enumerate(line_contents):
            m = us_header_re.match(ln or '')
            if not m: continue
            us_id = m.group(1)
            body_end = len(line_contents)
            for j in range(i+1, len(line_contents)):
                nxt = (line_contents[j] or '').strip()
                if not nxt: continue
                if us_header_re.match(nxt) or nxt.startswith('---'):
                    body_end = j
                    break
            narrative_line = None
            first_content = None
            for j in range(i+1, body_end):
                raw = line_contents[j] or ''
                stripped = raw.strip()
                if not stripped: continue
                if stripped.startswith('#'): continue
                if stripped.startswith('|'): continue
                if all(c in '- \t' for c in stripped) and len(stripped) >= 3: continue
                if first_content is None: first_content = j + 1
                if narrative_re.match(raw):
                    narrative_line = j + 1
                    break
            target = narrative_line or first_content
            if target:
                us_index[us_id] = target

        def _classify(text):
            if text is None: return 'oob'
            s = text.strip()
            if not s: return 'empty'
            if s.startswith('#'): return 'header'
            if s.startswith('|'): return 'table'
            if all(c in '- \t' for c in s) and len(s) >= 3: return 'sep'
            return 'content'

        result = []
        for mono in refs_array:
            if not isinstance(mono, list) or len(mono) != 2:
                result.append(mono)
                continue
            s, e = mono
            if not (isinstance(s, list) and len(s) == 2 and isinstance(e, list) and len(e) == 2):
                result.append(mono)
                continue
            try:
                sL = int(s[0]); sC = int(s[1])
                eL = int(e[0]); eC = int(e[1])
            except (TypeError, ValueError):
                result.append(mono)
                continue
            # zero-length
            if sL == eL and sC == eC:
                continue
            idx = sL - 1
            if not (0 <= idx < len(line_contents)):
                result.append(mono)
                continue
            src = line_contents[idx]
            cls = _classify(src)
            if cls == 'content':
                result.append(mono)
                continue
            if cls in ('empty', 'sep'):
                continue
            # header / table — FR-ID 추출 (브래킷 optional — TOC 표 row 는 `| ... PROJ-US-FR-011 | ...` 처럼 브래킷 없음)
            m = re.search(r'\[?([A-Za-z][\w-]*US-(?:FR|NFR)-\d+)\]?', src or '')
            if not m: continue
            target_line = us_index.get(m.group(1))
            if not target_line: continue
            tcontent = line_contents[target_line - 1] if 0 <= target_line - 1 < len(line_contents) else ''
            result.append([[target_line, 1], [target_line, max(1, len(tcontent))]])
        return result
    
    @staticmethod
    def convert_to_original_refs_using_trace_map(refs: List, trace_map: Dict) -> List:
        """
        프론트엔드 RefsTraceUtil.convertToOriginalRefsUsingTraceMap와 동일
        BC 라인 번호 → 원본 라인 번호 변환 (클램핑 없음)
        """
        if not refs or not trace_map:
            return refs
        
        original_refs = []
        processed_trace_infos = set()  # 중복 처리 방지
        
        for ref in refs:
            if not isinstance(ref, list) or len(ref) < 2:
                continue
            
            start_line = ref[0][0] if isinstance(ref[0], list) and len(ref[0]) > 0 else 1
            start_index = ref[0][1] if isinstance(ref[0], list) and len(ref[0]) > 1 else 0
            end_line = ref[1][0] if isinstance(ref[1], list) and len(ref[1]) > 0 else 1
            end_index = ref[1][1] if isinstance(ref[1], list) and len(ref[1]) > 1 else 0
            
            # 현재 ref 범위에서 고유한 traceInfo들을 수집
            unique_trace_infos = {}
            for i in range(start_line, end_line + 1):
                # traceMap 키가 문자열일 수 있으므로 정수와 문자열 모두 시도
                trace_info = trace_map.get(i) or trace_map.get(str(i))
                if not trace_info:
                    continue
                
                # traceKey 생성 (refs를 JSON 문자열로 변환)
                import json
                trace_key = json.dumps(trace_info.get('refs', []), sort_keys=True)
                if trace_key not in unique_trace_infos:
                    unique_trace_infos[trace_key] = {'traceInfo': trace_info, 'line': i}
            
            # 각 고유한 traceInfo에 대해 처리
            for trace_key, trace_data in unique_trace_infos.items():
                if trace_key in processed_trace_infos:
                    continue
                processed_trace_infos.add(trace_key)
                
                trace_info = trace_data['traceInfo']
                line = trace_data['line']
                
                if (trace_info.get('isDirectMatching') and 
                    trace_info.get('refs') and 
                    len(trace_info['refs']) == 1 and 
                    len(trace_info['refs'][0]) == 2):
                    
                    trace_ref = trace_info['refs'][0]
                    trace_start_line = trace_ref[0][0] if isinstance(trace_ref[0], list) and len(trace_ref[0]) > 0 else 1
                    trace_start_index = trace_ref[0][1] if isinstance(trace_ref[0], list) and len(trace_ref[0]) > 1 else 0
                    trace_end_line = trace_ref[1][0] if isinstance(trace_ref[1], list) and len(trace_ref[1]) > 0 else 1
                    trace_end_index = trace_ref[1][1] if isinstance(trace_ref[1], list) and len(trace_ref[1]) > 1 else 0
                    
                    if trace_start_line != trace_end_line:
                        if end_line - start_line == trace_end_line - trace_start_line:
                            # 여러 줄에 걸쳐 있고 줄 수가 같은 경우: 라인별 매핑
                            line_offset = line - start_line
                            corresponding_trace_line = trace_start_line + line_offset
                            
                            if line == start_line:
                                # 첫 번째 라인: 원본의 startIndex 사용
                                calculated_start_index = start_index
                                calculated_end_index = trace_end_index
                            elif line == end_line:
                                # 마지막 라인: 원본의 endIndex 사용
                                calculated_start_index = trace_start_index
                                calculated_end_index = end_index
                            else:
                                # 중간 라인: trace의 전체 범위 사용
                                calculated_start_index = trace_start_index
                                calculated_end_index = trace_end_index
                            
                            # 인덱스 유효성 검증
                            if calculated_start_index <= calculated_end_index:
                                original_refs.append([[corresponding_trace_line, calculated_start_index], 
                                                      [corresponding_trace_line, calculated_end_index]])
                            else:
                                original_refs.append([[corresponding_trace_line, trace_start_index], 
                                                      [corresponding_trace_line, trace_end_index]])
                        else:
                            # 줄 수가 다른 경우: 전체 trace refs 사용
                            original_refs.extend(trace_info['refs'])
                    elif start_line == end_line:
                        # 단일 라인 처리
                        if start_index > end_index or start_index < 1:
                            original_refs.append([[trace_start_line, trace_start_index], 
                                                  [trace_end_line, trace_end_index]])
                        else:
                            original_refs.append([[trace_start_line, start_index], 
                                                  [trace_end_line, end_index]])
                    else:
                        # 원본이 여러 줄, trace가 단일 줄인 경우
                        if line == start_line:
                            calculated_start_index = start_index
                            calculated_end_index = trace_end_index
                        elif line == end_line:
                            calculated_start_index = trace_start_index
                            calculated_end_index = end_index
                        else:
                            calculated_start_index = trace_start_index
                            calculated_end_index = trace_end_index
                        
                        if calculated_start_index <= calculated_end_index:
                            original_refs.append([[trace_start_line, calculated_start_index], 
                                                  [trace_end_line, calculated_end_index]])
                        else:
                            original_refs.append([[trace_start_line, trace_start_index], 
                                                  [trace_end_line, trace_end_index]])
                else:
                    # directMatching이 아니거나 복잡한 refs 구조인 경우
                    if trace_info.get('refs'):
                        original_refs.extend(trace_info['refs'])
        
        return original_refs if original_refs else refs
    
    @staticmethod
    def convert_refs_to_indexes(data: Any, line_numbered_requirements: str, is_use_xml_base: bool = True) -> Any:
        """프론트엔드 RefsTraceUtil.convertRefsToIndexes와 동일"""
        if not data or not line_numbered_requirements:
            return data
        
        lines = line_numbered_requirements.split('\n')
        min_line = RefsTraceUtil._get_min_line_number(line_numbered_requirements, is_use_xml_base)
        start_line_offset = min_line - 1
        
        return RefsTraceUtil._search_refs_array_recursively(
            data,
            lambda refs_array: RefsTraceUtil._convert_refs_array(refs_array, lines, start_line_offset, is_use_xml_base)
        )
    
    @staticmethod
    def _get_line_number_range(line_numbered_requirements: str, is_use_xml_base: bool) -> tuple:
        """최소/최대 라인 번호 찾기"""
        min_line = RefsTraceUtil._get_min_line_number(line_numbered_requirements, is_use_xml_base)
        max_line = RefsTraceUtil._get_max_line_number(line_numbered_requirements, is_use_xml_base)
        return (min_line, max_line)
    
    @staticmethod
    def _get_min_line_number(line_numbered_requirements: str, is_use_xml_base: bool) -> int:
        """최소 라인 번호 찾기"""
        lines = line_numbered_requirements.strip().split('\n')
        if not lines:
            return 1
        first_line = lines[0]
        min_line = RefsTraceUtil._extract_line_number(first_line, is_use_xml_base)
        return min_line if min_line is not None else 1
    
    @staticmethod
    def _get_max_line_number(line_numbered_requirements: str, is_use_xml_base: bool) -> int:
        """최대 라인 번호 찾기"""
        lines = line_numbered_requirements.strip().split('\n')
        if not lines:
            return 1
        last_line = lines[-1]
        max_line = RefsTraceUtil._extract_line_number(last_line, is_use_xml_base)
        return max_line if max_line is not None else len(lines)
    
    @staticmethod
    def _extract_line_number(line: str, is_use_xml_base: bool) -> int:
        """라인 번호 추출"""
        if is_use_xml_base:
            match = re.match(r'^<(\d+)>', line)
            return int(match.group(1)) if match else None
        else:
            match = re.match(r'^(\d+):', line)
            return int(match.group(1)) if match else None
    
    @staticmethod
    def _filter_invalid_refs(data: Any) -> Any:
        """refs의 구조에 맞지 않은 데이터 제거"""
        return RefsTraceUtil._search_refs_array_recursively(
            data,
            lambda refs_array: [
                refs for refs in refs_array
                if (isinstance(refs, list) and len(refs) == 2 and
                    isinstance(refs[0], list) and len(refs[0]) == 2 and
                    isinstance(refs[1], list) and len(refs[1]) == 2)
            ]
        )
    
    @staticmethod
    def _sanitize_refs_array(refs_array: List, lines: List[str], min_line: int, max_line: int, clamp) -> List:
        """프론트엔드 _sanitizeRefsArray와 동일"""
        result = []
        for mono in refs_array:
            if not isinstance(mono, list) or len(mono) != 2:
                result.append(mono)
                continue
            
            s, e = mono
            s_line = s[0] if isinstance(s[0], (int, float)) else min_line
            e_line = e[0] if isinstance(e[0], (int, float)) else s_line
            s_phrase = s[1]
            e_phrase = e[1]
            
            s_line = clamp(s_line, min_line, max_line)
            e_line = clamp(e_line, min_line, max_line)
            
            # tryRelocate 로직 (프론트엔드와 동일)
            # 가드: 짧은(< 4 자) phrase 는 흔한 토큰/조사일 수 있어 무관한 라인으로 잘못
            # 옮겨질 위험이 크므로 relocate 안 함. ±5 window 매칭이 모호(2 곳 이상)하면
            # 원래 라인 유지. (false relocation 방지)
            def try_relocate(line, phrase, is_end):
                if not isinstance(phrase, str) or not phrase.strip():
                    return line

                def has(ln):
                    idx = ln - min_line
                    content = lines[idx] if 0 <= idx < len(lines) else ''
                    return phrase in content

                if has(line):
                    return line

                if len(phrase.strip()) < 4:
                    return line

                candidates = []
                for d in range(1, 6):
                    if line - d >= min_line and has(line - d):
                        candidates.append(line - d)
                    if line + d <= max_line and has(line + d):
                        candidates.append(line + d)
                if len(candidates) == 1:
                    return candidates[0]
                return line  # 없거나 모호하면 원래 라인 유지
            
            s_line = try_relocate(s_line, s_phrase, False)
            e_line = try_relocate(e_line, e_phrase, True)
            
            if e_line < s_line:
                s_line, e_line = e_line, s_line
            
            result.append([[s_line, s_phrase], [e_line, e_phrase]])
        
        return result
    
    @staticmethod
    def _clamp_refs_array(refs_array: List, lines: List[str], min_line: int, max_line: int, clamp) -> List:
        """프론트엔드 _clampRefsArray와 동일"""
        result = []
        for mono in refs_array:
            if not isinstance(mono, list) or len(mono) != 2:
                result.append(mono)
                continue
            
            [[s_line, s_col], [e_line, e_col]] = mono
            s_line = clamp(s_line, min_line, max_line)
            e_line = clamp(e_line, min_line, max_line)
            
            def get_len(ln):
                idx = ln - min_line
                content = lines[idx] if 0 <= idx < len(lines) else ''
                return max(1, len(content))
            
            s_col = max(1, min(get_len(s_line), s_col if isinstance(s_col, (int, float)) else 1))
            e_col = max(1, min(get_len(e_line), e_col if isinstance(e_col, (int, float)) else e_col if isinstance(e_col, (int, float)) else get_len(e_line)))
            
            # 순서 정렬
            if e_line < s_line or (e_line == s_line and e_col < s_col):
                result.append([[e_line, e_col], [s_line, s_col]])
            else:
                result.append([[s_line, s_col], [e_line, e_col]])
        
        return result
    
    @staticmethod
    def _convert_refs_array(refs_array: List, lines: List[str], start_line_offset: int, is_use_xml_base: bool) -> List:
        """프론트엔드 _convertRefsArray와 동일"""
        return [RefsTraceUtil._convert_single_ref_range(ref_range, lines, start_line_offset, is_use_xml_base) 
                for ref_range in refs_array]
    
    @staticmethod
    def _convert_single_ref_range(ref_range: List, lines: List[str], start_line_offset: int, is_use_xml_base: bool) -> List:
        """프론트엔드 _convertSingleRefRange와 동일"""
        try:
            if not isinstance(ref_range, list) or (len(ref_range) != 1 and len(ref_range) != 2):
                return ref_range
            
            # 단일 ref 형식: [[lineNumber, phrase]]
            if len(ref_range) == 1:
                return RefsTraceUtil._convert_single_ref(ref_range[0], lines, start_line_offset, is_use_xml_base)
            
            # 이중 ref 형식: [[start], [end]]
            return RefsTraceUtil._convert_dual_ref(ref_range, lines, start_line_offset, is_use_xml_base)
        except (OSError, ValueError, TypeError, LookupError, AttributeError, RuntimeError, ImportError, ArithmeticError, AssertionError, StopIteration, StopAsyncIteration, BufferError):
            return ref_range
    
    @staticmethod
    def _convert_single_ref(single_ref: List, lines: List[str], start_line_offset: int, is_use_xml_base: bool) -> List:
        """프론트엔드 _convertSingleRef와 동일"""
        if not isinstance(single_ref, list) or len(single_ref) != 2:
            return [single_ref]
        
        line_number = single_ref[0]
        phrase = single_ref[1]
        
        try:
            line_num = int(line_number)
        except (ValueError, TypeError):
            return [single_ref]
        
        # 이미 변환된 경우 (phrase가 number인 경우)
        if isinstance(phrase, (int, float)):
            return [[line_num, int(phrase)]]
        
        # 라인 컨텐츠 가져오기
        adjusted_line_num = line_num - start_line_offset
        if adjusted_line_num < 1 or adjusted_line_num > len(lines):
            return [single_ref]
        
        line_content = lines[adjusted_line_num - 1]
        
        # XML 태그 제거
        if is_use_xml_base:
            match = re.match(rf'^<{line_num}>(.*)</{line_num}>$', line_content)
            if match:
                line_content = match.group(1)
        
        # phrase 찾기
        if isinstance(phrase, str) and phrase:
            phrase_index = line_content.find(phrase)
            if phrase_index == -1:
                # 대소문자 무시해서 다시 시도
                phrase_lower = phrase.lower()
                line_content_lower = line_content.lower()
                phrase_index_lower = line_content_lower.find(phrase_lower)
                if phrase_index_lower != -1:
                    phrase_index = phrase_index_lower
            
            if phrase_index != -1:
                start_index = phrase_index + 1  # 1-based
                end_index = phrase_index + len(phrase)
                return [[line_num, start_index], [line_num, end_index]]
        
        # phrase를 찾지 못하면 전체 라인
        line_end_index = len(line_content) if line_content else 1
        return [[line_num, 1], [line_num, line_end_index]]
    
    @staticmethod
    def _convert_dual_ref(ref_range: List, lines: List[str], start_line_offset: int, is_use_xml_base: bool) -> List:
        """프론트엔드 _convertDualRef와 동일"""
        start_ref = ref_range[0]
        end_ref = ref_range[1]
        
        if not isinstance(start_ref, list) or len(start_ref) < 2:
            return ref_range
        if not isinstance(end_ref, list) or len(end_ref) < 2:
            return ref_range
        
        start_line = start_ref[0]
        start_phrase = start_ref[1] if isinstance(start_ref[1], str) else ''
        end_line = end_ref[0]
        end_phrase = end_ref[1] if isinstance(end_ref[1], str) else ''
        
        try:
            start_line_num = int(start_line)
            end_line_num = int(end_line)
        except (ValueError, TypeError):
            return ref_range
        
        # 이미 변환된 경우
        if isinstance(start_phrase, (int, float)) and isinstance(end_phrase, (int, float)):
            return [[start_line_num, int(start_phrase)], [end_line_num, int(end_phrase)]]
        
        # 시작 라인 컨텐츠
        start_adjusted = start_line_num - start_line_offset
        if start_adjusted < 1 or start_adjusted > len(lines):
            return ref_range
        start_line_content = lines[start_adjusted - 1]
        if is_use_xml_base:
            start_match = re.match(rf'^<{start_line_num}>(.*)</{start_line_num}>$', start_line_content)
            if start_match:
                start_line_content = start_match.group(1)
        
        # 끝 라인 컨텐츠
        end_adjusted = end_line_num - start_line_offset
        if end_adjusted < 1 or end_adjusted > len(lines):
            return ref_range
        end_line_content = lines[end_adjusted - 1]
        if is_use_xml_base:
            end_match = re.match(rf'^<{end_line_num}>(.*)</{end_line_num}>$', end_line_content)
            if end_match:
                end_line_content = end_match.group(1)
        
        # phrase 위치 찾기
        start_index = RefsTraceUtil._find_column_for_words(start_line_content, start_phrase, False)
        end_index = RefsTraceUtil._find_column_for_words(end_line_content, end_phrase, True)
        
        return [[start_line_num, start_index], [end_line_num, end_index]]
    
    @staticmethod
    def _find_column_for_words(line_content: str, phrase: str, is_end_position: bool) -> int:
        """프론트엔드 _findColumnForWords와 동일"""
        if not line_content:
            return 1
        if not phrase:
            return len(line_content) if is_end_position else 1
        
        index = line_content.find(phrase)
        if index == -1:
            return len(line_content) if is_end_position else 1
        
        return (index + len(phrase)) if is_end_position else (index + 1)  # 1-based
    
    @staticmethod
    def _search_refs_array_recursively(data: Any, refs_handler) -> Any:
        """프론트엔드 searchRefsArrayRecursively와 동일"""
        if data is None:
            return None
        
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if key.lower().endswith('refs') and isinstance(value, list):
                    result[key] = refs_handler(value)
                else:
                    result[key] = RefsTraceUtil._search_refs_array_recursively(value, refs_handler)
            return result
        
        if isinstance(data, list):
            return [RefsTraceUtil._search_refs_array_recursively(item, refs_handler) for item in data]
        
        return data
    
    @staticmethod
    def remove_refs_attributes(data: Any) -> Any:
        """
        프론트엔드 RefsTraceUtil.removeRefsAttributes와 동일
        재귀적으로 "refs"로 끝나는 모든 키를 제거
        """
        if data is None:
            return None
        
        # 리스트인 경우
        if isinstance(data, list):
            return [RefsTraceUtil.remove_refs_attributes(item) for item in data]
        
        # 딕셔너리인 경우
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                # "refs"로 끝나는 키는 제거
                if key.lower().endswith('refs'):
                    continue
                # 재귀적으로 처리
                result[key] = RefsTraceUtil.remove_refs_attributes(value)
            return result
        
        # 원시 타입은 그대로 반환
        return data
    
    @staticmethod
    def validate_refs(data: Any, original_requirements: str, start_line_offset: int = 0) -> None:
        """
        프론트엔드 RefsTraceUtil.validateRefs와 동일
        refs의 유효성 검증 (라인/컬럼 인덱스 범위 체크)
        """
        if not data or not original_requirements:
            return
        
        lines = original_requirements.split('\n')
        invalid_refs = []
        
        def validate_refs_array(refs_array: List) -> None:
            try:
                for refs in refs_array:
                    if not isinstance(refs, list) or len(refs) < 2:
                        continue
                    
                    start_line = refs[0][0] if isinstance(refs[0], list) and len(refs[0]) > 0 else 1
                    start_col = refs[0][1] if isinstance(refs[0], list) and len(refs[0]) > 1 else 1
                    end_line = refs[1][0] if isinstance(refs[1], list) and len(refs[1]) > 0 else 1
                    end_col = refs[1][1] if isinstance(refs[1], list) and len(refs[1]) > 1 else 1
                    
                    start_line_index = start_line - 1 - start_line_offset
                    start_col_index = start_col - 1
                    end_line_index = end_line - 1 - start_line_offset
                    end_col_index = end_col - 1
                    
                    if start_line_index < 0 or start_line_index >= len(lines):
                        invalid_refs.append(refs)
                        continue
                    
                    if end_line_index < 0 or end_line_index >= len(lines):
                        invalid_refs.append(refs)
                        continue
                    
                    start_line_content = lines[start_line_index]
                    end_line_content = lines[end_line_index]
                    
                    if not start_line_content or not end_line_content:
                        invalid_refs.append(refs)
                        continue
                    
                    if (start_col_index < 0 or start_col_index > len(start_line_content) or 
                        end_col_index < 0 or end_col_index > len(end_line_content)):
                        invalid_refs.append(refs)
            except (OSError, ValueError, TypeError, LookupError, AttributeError, RuntimeError, ImportError, ArithmeticError, AssertionError, StopIteration, StopAsyncIteration, BufferError):
                invalid_refs.append(refs)
        
        RefsTraceUtil._search_refs_array_recursively(data, validate_refs_array)
        
        if invalid_refs:
            import logging
            logging.error(f"[RefsTraceUtil] Invalid refs found in validateRefs: {invalid_refs}")
            raise ValueError(f"Invalid refs found in validateRefs: {invalid_refs}")

