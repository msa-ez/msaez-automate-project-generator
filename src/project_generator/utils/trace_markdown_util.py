"""
TraceMarkdownUtil - 프론트엔드 TraceMarkdownUtil과 동일한 기능
BC의 requirements 배열을 기반으로 markdown과 traceMap 생성
"""
from typing import Dict, List, Optional


class TraceMarkdownUtil:
    @staticmethod
    def _is_valid_bounded_context(bounded_context: Dict) -> bool:
        """
        프론트엔드 __isValidBoundedContext와 동일한 validation
        """
        if not isinstance(bounded_context, dict):
            return False
        
        # name: required, string, minLength 1
        name = bounded_context.get('name')
        if not isinstance(name, str) or len(name.strip()) < 1:
            return False
        
        # alias: required, string, minLength 1
        alias = bounded_context.get('alias')
        if not isinstance(alias, str) or len(alias.strip()) < 1:
            return False
        
        # role: optional, string
        role = bounded_context.get('role')
        if role is not None and not isinstance(role, str):
            return False
        
        # events: optional, array of strings
        events = bounded_context.get('events')
        if events is not None:
            if not isinstance(events, list):
                return False
            for event in events:
                if not isinstance(event, str):
                    return False
        
        # requirements: optional, array of objects
        requirements = bounded_context.get('requirements')
        if requirements is not None:
            if not isinstance(requirements, list):
                return False
            for req in requirements:
                if not isinstance(req, dict):
                    return False
                # text: required, string
                text = req.get('text')
                if not isinstance(text, str):
                    return False
                # type: required, string
                req_type = req.get('type')
                if not isinstance(req_type, str):
                    return False
        
        return True
    
    @staticmethod
    def _is_valid_relations(relations: Optional[List[Dict]]) -> bool:
        """
        프론트엔드 __isValidRelations와 동일한 validation
        """
        # relations는 null, undefined 또는 빈 배열일 수 있음
        if relations is None:
            return True
        
        if not isinstance(relations, list):
            return False
        
        for rel in relations:
            if not isinstance(rel, dict):
                return False
            # name: required, string
            if not isinstance(rel.get('name'), str):
                return False
            # type: required, string
            if not isinstance(rel.get('type'), str):
                return False
            # upStream: required, object with name
            up_stream = rel.get('upStream')
            if not isinstance(up_stream, dict) or not isinstance(up_stream.get('name'), str):
                return False
            # downStream: required, object with name
            down_stream = rel.get('downStream')
            if not isinstance(down_stream, dict) or not isinstance(down_stream.get('name'), str):
                return False
        
        return True
    
    @staticmethod
    def _is_valid_explanations(explanations: Optional[List[Dict]]) -> bool:
        """
        프론트엔드 __isValidExplanations와 동일한 validation
        """
        # explanations는 null, undefined 또는 빈 배열일 수 있음
        if explanations is None:
            return True
        
        if not isinstance(explanations, list):
            return False
        
        for exp in explanations:
            if not isinstance(exp, dict):
                return False
            # sourceContext: required, string
            if not isinstance(exp.get('sourceContext'), str):
                return False
            # targetContext: required, string
            if not isinstance(exp.get('targetContext'), str):
                return False
            # reason: required, string
            if not isinstance(exp.get('reason'), str):
                return False
            # interactionPattern: required, string
            if not isinstance(exp.get('interactionPattern'), str):
                return False
        
        return True
    
    @staticmethod
    def _is_valid_events(events: Optional[List[Dict]]) -> bool:
        """
        프론트엔드 __isValidEvents와 동일한 validation
        """
        # events는 null, undefined 또는 빈 배열일 수 있음
        if events is None:
            return True
        
        if not isinstance(events, list):
            return False
        
        for event in events:
            if not isinstance(event, dict):
                return False
            # name: required, string
            if not isinstance(event.get('name'), str):
                return False
        
        return True
    @staticmethod
    def get_description_with_mapping_index(
        bounded_context: Dict,
        relations: Optional[List[Dict]] = None,
        explanations: Optional[List[Dict]] = None,
        events: Optional[List[Dict]] = None
    ) -> Dict:
        """
        BC 정보를 기반으로 markdown과 traceMap 생성 (프론트엔드와 동일)
        
        Args:
            bounded_context: BC 정보 (name, alias, role, events, requirements 등)
            relations: BC 간 관계 정보
            explanations: 관계 설명 정보
            events: 이벤트 정보
            
        Returns:
            {markdown: str, traceMap: Dict}
        """
        # 프론트엔드와 동일한 validation
        if not TraceMarkdownUtil._is_valid_bounded_context(bounded_context):
            raise ValueError('Invalid boundedContext')
        if not TraceMarkdownUtil._is_valid_relations(relations):
            raise ValueError('Invalid relations')
        if not TraceMarkdownUtil._is_valid_explanations(explanations):
            raise ValueError('Invalid explanations')
        if not TraceMarkdownUtil._is_valid_events(events):
            raise ValueError('Invalid events')
        
        if relations is None:
            relations = []
        if explanations is None:
            explanations = []
        if events is None:
            events = []
        
        current_line = 1
        markdown_lines = []
        trace_map = {}
        
        def add_lines(content: str, refs: Optional[List] = None, is_direct_matching: bool = False):
            """라인을 추가하고 traceMap에 refs 정보 저장 (프론트엔드와 동일)"""
            nonlocal current_line
            lines = str(content).split('\n')
            
            # isDirectMatching이 true이고 refs가 있는 경우, 라인별 refs를 계산
            if is_direct_matching and refs and len(refs) > 0:
                # 첫 번째 ref 범위에서 시작 라인 번호를 추출
                first_ref = refs[0]
                if isinstance(first_ref, list) and len(first_ref) >= 2:
                    start_ref = first_ref[0]
                    end_ref = first_ref[1]
                    
                    if (isinstance(start_ref, list) and len(start_ref) >= 2 and
                        isinstance(end_ref, list) and len(end_ref) >= 2):
                        start_line = start_ref[0]
                        end_line = end_ref[0]
                        
                        for i, line in enumerate(lines):
                            markdown_lines.append(line)
                            
                            if line.strip():
                                # 현재 라인에 해당하는 원본 라인 번호 계산
                                original_line_number = start_line + i
                                
                                # 원본 범위를 벗어나지 않도록 확인
                                if original_line_number <= end_line:
                                    # 프론트엔드와 동일: BC description 라인 길이 사용
                                    line_end_col = len(line) if len(line) > 0 else 1
                                    line_refs = [[[original_line_number, 1], [original_line_number, line_end_col]]]
                                    
                                    trace_map[current_line] = {
                                        'refs': line_refs,
                                        'isDirectMatching': True
                                    }
                            
                            current_line += 1
                        return
                else:
                    # refs 구조가 예상과 다른 경우 기존 로직 사용
                    for line in lines:
                        markdown_lines.append(line)
                        if refs and line.strip():
                            trace_map[current_line] = {
                                'refs': refs,
                                'isDirectMatching': is_direct_matching
                            }
                        current_line += 1
                    return
            
            # 기존 로직 (isDirectMatching이 false이거나 refs가 없는 경우)
            # 프론트엔드와 동일하게 원본 requirements의 refs를 그대로 사용
            for line in lines:
                markdown_lines.append(line)
                if refs and line.strip():
                    trace_map[current_line] = {
                        'refs': refs,
                        'isDirectMatching': is_direct_matching
                    }
                current_line += 1
        
        # BC Overview
        bc_name = bounded_context.get('name', 'Unknown')
        bc_alias = bounded_context.get('alias', '')
        add_lines(f'# Bounded Context Overview: {bc_name} ({bc_alias})')
        add_lines('')
        
        # Role
        if bounded_context.get('role') and bounded_context.get('roleRefs'):
            add_lines('## Role')
            add_lines(bounded_context['role'], bounded_context['roleRefs'], False)
            add_lines('')
        
        # Key Events
        bc_events = bounded_context.get('events', [])
        if bc_events:
            add_lines('## Key Events')
            
            for event_name in bc_events:
                matched_event = next((e for e in events if e.get('name') == event_name), None)
                if matched_event and matched_event.get('refs'):
                    add_lines(f'- {event_name}', matched_event['refs'], False)
            
            add_lines('')
        
        # Requirements
        requirements = bounded_context.get('requirements', [])
        if requirements:
            # 중복 제거 (text 기준)
            unique_requirements = {}
            for req in requirements:
                req_text = req.get('text', '')
                if req_text and req_text not in unique_requirements:
                    unique_requirements[req_text] = req
            
            # 타입별로 그룹화
            requirements_by_type = {}
            for req in unique_requirements.values():
                req_type = req.get('type', '')
                if req_type:
                    if req_type not in requirements_by_type:
                        requirements_by_type[req_type] = []
                    requirements_by_type[req_type].append(req)
            
            if requirements_by_type:
                add_lines('# Requirements')
                add_lines('')
                
                for req_type, reqs in requirements_by_type.items():
                    add_lines(f'## {req_type}')
                    add_lines('')
                    
                    for req in reqs:
                        req_refs = req.get('refs')
                        if not req_refs:
                            continue
                        
                        text_content = req.get('text', '')
                        req_type_lower = req_type.lower()
                        
                        if req_type_lower == 'ddl':
                            add_lines('```sql')
                            add_lines(text_content, req_refs, True)
                            add_lines('```')
                        elif req_type_lower == 'event':
                            # JSON 파싱 시도
                            try:
                                import json
                                parsed_event = json.loads(text_content)
                                # 프론트엔드 RefsTraceUtil.removeRefsAttributes와 동일하게 재귀적으로 refs 제거
                                event_without_refs = TraceMarkdownUtil._remove_refs_attributes(parsed_event)
                                formatted_json = json.dumps(event_without_refs, indent=2, ensure_ascii=False)
                                
                                add_lines('```json')
                                add_lines(formatted_json, req_refs, False)
                                add_lines('```')
                            except:
                                add_lines(text_content, req_refs, False)
                        else:
                            add_lines(text_content, req_refs, True)
                        
                        add_lines('')
        
        # Context Relations
        if relations:
            related_relations = [
                rel for rel in relations
                if (rel.get('upStream', {}).get('name') == bc_name or
                    rel.get('downStream', {}).get('name') == bc_name)
            ]
            
            if related_relations:
                add_lines('## Context Relations')
                add_lines('')
                
                for rel in related_relations:
                    rel_refs = rel.get('refs')
                    if not rel_refs:
                        continue
                    
                    is_upstream = rel.get('upStream', {}).get('name') == bc_name
                    target_context = rel.get('downStream', {}) if is_upstream else rel.get('upStream', {})
                    direction = "sends to" if is_upstream else "receives from"
                    
                    add_lines(f'### {rel.get("name", "")}')
                    add_lines(f'- **Type**: {rel.get("type", "")}', rel_refs, False)
                    add_lines(f'- **Direction**: {direction} {target_context.get("alias", "")} ({target_context.get("name", "")})', rel_refs, False)
                    
                    # Explanation 찾기
                    explanation = next((
                        exp for exp in explanations
                        if ((exp.get('sourceContext') == bc_alias and exp.get('targetContext') == target_context.get('alias')) or
                            (exp.get('targetContext') == bc_alias and exp.get('sourceContext') == target_context.get('alias')))
                    ), None)
                    
                    if explanation and explanation.get('refs'):
                        add_lines(f'- **Reason**: {explanation.get("reason", "")}', explanation['refs'], False)
                        add_lines(f'- **Interaction Pattern**: {explanation.get("interactionPattern", "")}', explanation['refs'], False)
                    
                    add_lines('')
        
        final_markdown = '\n'.join(markdown_lines).strip()
        
        # Firebase 호환: 숫자 키를 문자열로 변환 (프론트엔드 Map 객체와 동일한 효과)
        # Firebase Realtime Database는 숫자 문자열 키를 가진 객체를 배열로 변환할 수 있으므로,
        # 키에 prefix를 추가하여 배열 변환을 방지
        # 프론트엔드에서 traceMap[i]로 접근하므로, 키는 숫자 문자열이어야 함
        # 하지만 Firebase가 변환하지 않도록 prefix 추가 후, 프론트엔드에서 제거
        # 실제로는 프론트엔드 MessageDataRestoreUtil이 배열을 객체로 변환하므로,
        # 여기서는 문자열 키로만 변환 (프론트엔드 Map과 동일)
        firebase_compatible_trace_map = {}
        for key, value in trace_map.items():
            # 키를 문자열로 변환 (프론트엔드 Map 객체와 동일한 형식)
            # Firebase가 배열로 변환하지 않도록, 키를 그대로 문자열로 유지
            firebase_compatible_trace_map[str(key)] = value
        
        return {'markdown': final_markdown, 'traceMap': firebase_compatible_trace_map}
    
    @staticmethod
    def _remove_refs_attributes(data):
        """
        프론트엔드 RefsTraceUtil.removeRefsAttributes와 동일한 기능
        재귀적으로 "refs"로 끝나는 모든 키를 제거
        """
        if data is None:
            return None
        
        # 리스트인 경우
        if isinstance(data, list):
            return [TraceMarkdownUtil._remove_refs_attributes(item) for item in data]
        
        # 딕셔너리인 경우
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                # "refs"로 끝나는 키는 제거
                if key.lower().endswith('refs'):
                    continue
                # 재귀적으로 처리
                result[key] = TraceMarkdownUtil._remove_refs_attributes(value)
            return result
        
        # 원시 타입은 그대로 반환
        return data

