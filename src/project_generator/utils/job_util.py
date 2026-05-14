import re
import threading
import queue
import time
from dataclasses import dataclass
from typing import Dict, Optional
import atexit

from ..systems.storage_system_factory import StorageSystemFactory
from ..models import State
from ..utils import JsonUtil
from ..config import Config
from .logging_util import LoggingUtil

@dataclass
class UpdateRequest:
    """Firebase 업데이트 요청 데이터 클래스"""
    state: Dict[str, any]
    timestamp: float
    operation_type: str  # 'update', 'set', 'delete'
    path_suffix: Optional[str] = None  # 추가 경로가 필요한 경우

class JobUtil:
    # 클래스 레벨 변수로 큐와 스레드 관리
    _update_queues: Dict[str, queue.Queue] = {}
    _worker_threads: Dict[str, threading.Thread] = {}
    _shutdown_events: Dict[str, threading.Event] = {}
    _queue_lock = threading.Lock()
    _initialized = False
    
    _procssed_job_ids = set()

    # 설정값
    MAX_QUEUE_SIZE = 100  # 큐 최대 크기
    WORKER_TIMEOUT = 5.0  # 작업자 스레드 종료 대기 시간

    # 조건부 데이터 업데이트를 위한 메모리
    previous_data_job_id = None
    previous_data_state = None
 
    @classmethod
    def _initialize_cleanup(cls):
        """프로그램 종료시 자동 정리를 위한 초기화"""
        if not cls._initialized:
            # 프로그램 종료시 자동으로 모든 리소스 정리
            atexit.register(cls.cleanup_all_job_resources)
            cls._initialized = True
    
    @staticmethod
    def is_valid_job_id(job_id: str) -> bool:
        """
        JavaScript에서 생성된 JOB_ID 검증
        
        지원 형식:
        - UUID: 8-4-4-4-12 (예: 5f8c86d1-16be-2ca9-0720-8a51f44439ab)
        - UserStory: usgen-{timestamp}-{random} (예: usgen-1760429939640-pf7chz8d9)
        - Summarizer: summ-{timestamp}-{random} (예: summ-1760429939640-abc123xyz)
        
        Args:
            job_id (str): 검증할 JOB_ID
            
        Returns:
            bool: 유효한 JOB_ID 여부
        """
        if not job_id or not isinstance(job_id, str):
            return False
        
        # UserStory Job ID 형식 체크 (usgen-{timestamp}-{random})
        if job_id.startswith('usgen-'):
            usgen_pattern = re.compile(r'^usgen-\d{13,}-[a-z0-9]+$')
            return bool(usgen_pattern.match(job_id))
        
        # Summarizer Job ID 형식 체크 (summ-{timestamp}-{random})
        if job_id.startswith('summ-'):
            summ_pattern = re.compile(r'^summ-\d{13,}-[a-z0-9]+$')
            return bool(summ_pattern.match(job_id))
        
        # BoundedContext Job ID 형식 체크 (bcgen-{timestamp}-{random})
        if job_id.startswith('bcgen-'):
            bcgen_pattern = re.compile(r'^bcgen-\d{13,}-[a-z0-9]+$')
            return bool(bcgen_pattern.match(job_id))
        
        # Command/ReadModel Extractor Job ID 형식 체크 (cmrext-{timestamp}-{random})
        if job_id.startswith('cmrext-'):
            cmrext_pattern = re.compile(r'^cmrext-\d{13,}-[a-z0-9]+$')
            return bool(cmrext_pattern.match(job_id))
        
        # SiteMap Generator Job ID 형식 체크 (smapgen-{timestamp}-{random})
        if job_id.startswith('smapgen-'):
            smapgen_pattern = re.compile(r'^smapgen-\d{13,}-[a-z0-9]+$')
            return bool(smapgen_pattern.match(job_id))
        
        # Requirements Mapper Job ID 형식 체크 (reqmap-{timestamp}-{random})
        if job_id.startswith('reqmap-'):
            reqmap_pattern = re.compile(r'^reqmap-\d{13,}-[a-z0-9]+$')
            return bool(reqmap_pattern.match(job_id))
        
        # Aggregate Draft Generator Job ID 형식 체크 (aggr-draft-{timestamp}-{random})
        if job_id.startswith('aggr-draft-'):
            aggr_draft_pattern = re.compile(r'^aggr-draft-\d{13,}-[a-z0-9]+$')
            return bool(aggr_draft_pattern.match(job_id))
        
        # Preview Fields Generator Job ID 형식 체크 (preview-fields-{timestamp}-{random})
        if job_id.startswith('preview-fields-'):
            preview_fields_pattern = re.compile(r'^preview-fields-\d{13,}-[a-z0-9]+$')
            return bool(preview_fields_pattern.match(job_id))
        
        # DDL Fields Generator Job ID 형식 체크 (ddl-fields-{timestamp}-{random})
        if job_id.startswith('ddl-fields-'):
            ddl_fields_pattern = re.compile(r'^ddl-fields-\d{13,}-[a-z0-9]+$')
            return bool(ddl_fields_pattern.match(job_id))
        
        # Traceability Generator Job ID 형식 체크 (trace-add-{timestamp}-{random})
        if job_id.startswith('trace-add-'):
            trace_add_pattern = re.compile(r'^trace-add-\d{13,}-[a-z0-9]+$')
            return bool(trace_add_pattern.match(job_id))
        
        # Standard Transformer Job ID 형식 체크 (std-trans-{timestamp}-{random})
        if job_id.startswith('std-trans-'):
            std_trans_pattern = re.compile(r'^std-trans-\d{13,}-[a-z0-9]+$')
            return bool(std_trans_pattern.match(job_id))
        
        # DDL Extractor Job ID 형식 체크 (ddl-extract-{timestamp}-{random})
        if job_id.startswith('ddl-extract-'):
            ddl_extract_pattern = re.compile(r'^ddl-extract-\d{13,}-[a-z0-9]+$')
            return bool(ddl_extract_pattern.match(job_id))
        
        # Requirements Validator Job ID 형식 체크 (req-valid-{timestamp}-{random})
        if job_id.startswith('req-valid-'):
            req_valid_pattern = re.compile(r'^req-valid-\d{13,}-[a-z0-9]+$')
            return bool(req_valid_pattern.match(job_id))

        # Event Flow Stitcher Job ID 형식 체크 (flow-stitch-{timestamp}-{random})
        if job_id.startswith('flow-stitch-'):
            flow_stitch_pattern = re.compile(r'^flow-stitch-\d{13,}-[a-z0-9]+$')
            return bool(flow_stitch_pattern.match(job_id))
        
        # EventStorming UUID 형식 검증
        # 1. 기본 길이 검증 (정확히 36자: 32개 16진수 + 4개 하이픈)
        if len(job_id) != 36:
            return False
        
        # 2. UUID 형식 검증 (8-4-4-4-12 패턴의 16진수)
        uuid_pattern = re.compile(
            r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
        )
        
        if not uuid_pattern.match(job_id):
            return False
        
        # 3. 하이픈 위치 정확성 검증
        expected_hyphen_positions = [8, 13, 18, 23]
        for pos in expected_hyphen_positions:
            if job_id[pos] != '-':
                return False
        
        # 4. 16진수 부분만 추출하여 추가 검증
        hex_parts = job_id.split('-')
        expected_lengths = [8, 4, 4, 4, 12]
        
        if len(hex_parts) != 5:
            return False
            
        for i, part in enumerate(hex_parts):
            if len(part) != expected_lengths[i]:
                return False
            # 각 부분이 유효한 16진수인지 확인
            try:
                int(part, 16)
            except ValueError:
                return False
        
        # 5. 보안 검증: 경로 조작 문자 금지
        forbidden_chars = ['/', '\\', '..', '#', '$', '[', ']', '?', '&']
        if any(char in job_id for char in forbidden_chars):
            return False
        
        return True

    @staticmethod
    def _ensure_worker_for_job(job_id: str):
        """
        특정 Job ID에 대한 작업자 스레드가 존재하는지 확인하고 없으면 생성
        
        Args:
            job_id (str): Job ID
        """
        # 자동 정리 초기화
        JobUtil._initialize_cleanup()
        
        with JobUtil._queue_lock:
            if job_id not in JobUtil._update_queues:
                # 큐와 이벤트 생성 (크기 제한 적용)
                JobUtil._update_queues[job_id] = queue.Queue(maxsize=JobUtil.MAX_QUEUE_SIZE)
                JobUtil._shutdown_events[job_id] = threading.Event()
                
                # 작업자 스레드 시작
                worker_thread = threading.Thread(
                    target=JobUtil._update_worker,
                    args=(job_id,),
                    name=f"JobUpdateWorker-{job_id}",
                    daemon=True
                )
                worker_thread.start()
                JobUtil._worker_threads[job_id] = worker_thread
                
                LoggingUtil.debug("job_util", f"[Job Queue] Job ID {job_id}에 대한 업데이트 큐 및 작업자 스레드 생성됨")

    @staticmethod
    def _update_worker(job_id: str):
        """
        Job별 Firebase 업데이트 작업자 스레드
        
        Args:
            job_id (str): 처리할 Job ID
        """
        LoggingUtil.debug("job_util", f"[Job Worker] Job ID {job_id} 업데이트 작업자 스레드 시작")
        
        update_queue = JobUtil._update_queues[job_id]
        shutdown_event = JobUtil._shutdown_events[job_id]
        
        processed_count = 0
        
        try:
            while True:
                try:
                    # 큐에서 업데이트 요청 가져오기 (타임아웃 1초)
                    update_request = update_queue.get(timeout=1.0)
                    
                    if update_request is None:  # 종료 신호
                        LoggingUtil.debug("job_util", f"[Job Worker] Job ID {job_id} 종료 신호 수신")
                        break
                    
                    # Firebase 업데이트 실행
                    JobUtil._execute_firebase_update(update_request)
                    processed_count += 1
                    
                    update_queue.task_done()
                    
                except queue.Empty:
                    # 큐가 비어있음 - shutdown_event 확인
                    if shutdown_event.is_set():
                        # 종료 요청이 있고 큐가 비어있으면 한 번 더 확인 후 종료
                        try:
                            # 마지막으로 0.1초 더 기다려서 혹시 남은 요청이 있는지 확인
                            update_request = update_queue.get(timeout=0.1)
                            if update_request is None:
                                LoggingUtil.debug("job_util", f"[Job Worker] Job ID {job_id} 최종 종료 신호 수신")
                                break
                            
                            # 마지막 요청 처리
                            JobUtil._execute_firebase_update(update_request)
                            processed_count += 1
                            update_queue.task_done()
                            
                        except queue.Empty:
                            # 정말로 큐가 비어있음 - 종료
                            LoggingUtil.debug("job_util", f"[Job Worker] Job ID {job_id} 큐 비어있음 확인, 종료")
                            break
                    # shutdown_event가 설정되지 않았으면 계속 대기
                    continue
                    
                except Exception as e:
                    LoggingUtil.exception("job_util", f"[Job Worker Error] Job ID {job_id} 업데이트 처리 중 오류", e)
                    
        except Exception as e:
            LoggingUtil.exception("job_util", f"[Job Worker Fatal] Job ID {job_id} 작업자 스레드 치명적 오류", e)
        
        finally:
            LoggingUtil.debug("job_util", f"[Job Worker] Job ID {job_id} 업데이트 작업자 스레드 종료 (처리된 요청: {processed_count}개)")

    @staticmethod
    def _execute_firebase_update(update_request: UpdateRequest):
        """
        Firebase 업데이트 실행
        
        Args:
            update_request (UpdateRequest): 업데이트 요청
        """
        try:
            storage_system = StorageSystemFactory.instance()
            job_id = update_request.state["inputs"]["jobId"]
            
            # 기본 경로 구성
            base_path = Config.get_job_path(job_id)
            if update_request.path_suffix:
                path = f"{base_path}/{update_request.path_suffix}"
            else:
                path = base_path
            
            # 업데이트 데이터 준비
            data = {
                "state": update_request.state,
                "lastUpdated": update_request.timestamp
            }
            
            # 업데이트 타입에 따른 처리
            if update_request.operation_type == "set":
                storage_system.set_data(path, data)
            elif update_request.operation_type == "update":
                if JobUtil.previous_data_job_id == job_id:
                    storage_system.conditional_update_data(path, data, JobUtil.previous_data_state)
                    JobUtil.previous_data_job_id = job_id
                    JobUtil.previous_data_state = data
                else:
                    storage_system.update_data(path, data)
                    JobUtil.previous_data_job_id = job_id
                    JobUtil.previous_data_state = data
        
            elif update_request.operation_type == "delete":
                storage_system.delete_data(path)

        except Exception as e:
            LoggingUtil.exception("job_util", f"[Firebase Update Error] Job ID {update_request.state['inputs']['jobId']} 업데이트 실행 실패", e)

    @staticmethod
    def _add_update_to_queue(state: Dict[str, any], operation_type: str, path_suffix: Optional[str] = None):
        """
        업데이트 요청을 큐에 추가
        
        Args:
            state (Dict[str, any]): 상태 딕셔너리
            operation_type (str): 업데이트 타입 ('set', 'update', 'delete')
            path_suffix (str, optional): 추가 경로
        """
        job_id = state["inputs"]["jobId"]
        
        if not JobUtil.is_valid_job_id(job_id):
            LoggingUtil.warning("job_util", f"[Job Queue Error] 유효하지 않은 Job ID: {job_id}")
            return
        
        # 이미 종료 신호가 설정된 Job에 대해서는 업데이트 요청 거부
        with JobUtil._queue_lock:
            if job_id in JobUtil._shutdown_events and JobUtil._shutdown_events[job_id].is_set():
                LoggingUtil.warning("job_util", f"[Job Queue Warning] Job ID {job_id}는 이미 종료 중입니다 - 업데이트 요청 무시됨")
                return
        
        # 작업자가 없으면 생성
        JobUtil._ensure_worker_for_job(job_id)
        
        # 업데이트 요청 생성
        update_request = UpdateRequest(
            state=state,
            timestamp=time.time(),
            operation_type=operation_type,
            path_suffix=path_suffix
        )
        
        # 큐에 추가
        try:
            JobUtil._update_queues[job_id].put(update_request, timeout=2.0)
        except queue.Full:
            LoggingUtil.warning("job_util", f"[Job Queue Warning] Job ID {job_id} 큐가 가득참 - 업데이트 요청 무시됨")
        except KeyError:
            LoggingUtil.warning("job_util", f"[Job Queue Error] Job ID {job_id} 큐가 존재하지 않음")
        except Exception as e:
            LoggingUtil.warning("job_util", f"[Job Queue Error] 큐 추가 실패: {str(e)}")

    @staticmethod
    def cleanup_job_resources(job_id: str):
        """
        특정 Job의 리소스 정리 (큐, 스레드 등)
        큐에 있는 모든 업데이트가 완료된 후 안전하게 종료
        
        Args:
            job_id (str): 정리할 Job ID
        """
        with JobUtil._queue_lock:
            if job_id in JobUtil._shutdown_events:
                LoggingUtil.debug("job_util", f"[Job Cleanup] Job ID {job_id} 리소스 정리 시작")
                
                # 1단계: 새로운 요청 추가 방지 (아직 종료 신호는 보내지 않음)
                if job_id in JobUtil._update_queues:
                    queue_obj = JobUtil._update_queues[job_id]
                          
                    while not queue_obj.empty():
                        time.sleep(1)
                
                # 2단계: 종료 신호 설정
                JobUtil._shutdown_events[job_id].set()
                LoggingUtil.debug("job_util", f"[Job Cleanup] Job ID {job_id} 종료 신호 설정")
                
                # 3단계: 종료 신호(None)를 큐에 추가
                if job_id in JobUtil._update_queues:
                    try:
                        JobUtil._update_queues[job_id].put(None, timeout=1.0)
                        LoggingUtil.debug("job_util", f"[Job Cleanup] Job ID {job_id} 최종 종료 신호 전송")
                    except queue.Full:
                        LoggingUtil.warning("job_util", f"[Job Cleanup Warning] Job ID {job_id} 큐가 가득참 - 강제 종료")
                    except Exception as e:
                        LoggingUtil.exception("job_util", f"[Job Cleanup Error] Job ID {job_id} 종료 신호 전송 실패", e)
                
                # 4단계: 스레드 종료 대기
                if job_id in JobUtil._worker_threads:
                    worker_thread = JobUtil._worker_threads[job_id]
                    LoggingUtil.debug("job_util", f"[Job Cleanup] Job ID {job_id} 작업자 스레드 종료 대기...")
                    worker_thread.join(timeout=JobUtil.WORKER_TIMEOUT)
                    if worker_thread.is_alive():
                        LoggingUtil.warning("job_util", f"[Job Cleanup Warning] Job ID {job_id} 작업자 스레드가 {JobUtil.WORKER_TIMEOUT}초 내에 종료되지 않음")
                    else:
                        LoggingUtil.debug("job_util", f"[Job Cleanup] Job ID {job_id} 작업자 스레드 정상 종료")
                
                # 5단계: 리소스 정리
                JobUtil._update_queues.pop(job_id, None)
                JobUtil._worker_threads.pop(job_id, None)
                JobUtil._shutdown_events.pop(job_id, None)
                
                LoggingUtil.debug("job_util", f"[Job Cleanup] Job ID {job_id} 리소스 정리 완료")

    @staticmethod
    def cleanup_all_job_resources():
        """
        모든 Job의 리소스 정리
        """
        LoggingUtil.debug("job_util", "[Job Cleanup] 모든 Job 리소스 정리 시작")
        
        with JobUtil._queue_lock:
            job_ids = list(JobUtil._shutdown_events.keys())
        
        for job_id in job_ids:
            try:
                JobUtil.cleanup_job_resources(job_id)
            except Exception as e:
                LoggingUtil.exception("job_util", f"[Job Cleanup Error] Job ID {job_id} 정리 중 오류", e)
        
        LoggingUtil.debug("job_util", "[Job Cleanup] 모든 Job 리소스 정리 완료")


    @staticmethod
    def new_job_to_firebase_fire_and_forget(state: State):
        """
        새로운 작업을 Firebase에 큐 기반으로 안전하게 업로드
        
        Args:
            state (State): 업로드할 상태 객체
        """
        JobUtil._add_update_to_queue(JobUtil._convert_state_to_firebase_data(state), "set")

    @staticmethod
    def update_job_to_firebase_fire_and_forget(state: State):
        """
        작업 상태를 Firebase에 큐 기반으로 안전하게 업데이트
        
        Args:
            state (State): 업데이트할 상태 객체
        """
        JobUtil._add_update_to_queue(JobUtil._convert_state_to_firebase_data(state), "update")
    
    @staticmethod
    def _convert_state_to_firebase_data(state: State):
        """
        상태 객체를 Firebase에 업로드 가능한 객체 데이터로 변환 및 복제
        """
        update_state = JsonUtil.convert_to_dict(JsonUtil.convert_to_json(state))
        update_state = JobUtil._delete_element_ref_from_state(update_state)
        update_state = JobUtil._delete_unused_events(update_state)
        return update_state

    @staticmethod
    def _delete_element_ref_from_state(state):
        """
        state의 outputs.esValue.relations에서 중복 데이터인 sourceElement, targetElement를 제거
        
        Args:
            state (State): 처리할 상태 객체
        """
        try:

            if not state['outputs'] or \
               not state['outputs']['esValue'] or \
               not state['outputs']['esValue']['relations']:
                return state
            
            
            relations = state['outputs']['esValue']['relations']
            for _, relation in relations.items():
                if 'sourceElement' in relation:
                    del relation['sourceElement']
                if 'targetElement' in relation:
                    del relation['targetElement']

            return state
        
        except Exception as e:
            LoggingUtil.exception("job_util", f"[State Optimization Error] delete_element_ref_from_state 실행 중 오류", e)
            return state

    @staticmethod
    def _delete_unused_events(state):
        """
        Policy와 연관되지 않은 Event들과 해당 Event와 관련된 관계들을 삭제합니다.
        
        Args:
            state: Firebase 업로드용 상태 딕셔너리
            
        Returns:
            state: 최적화된 상태 딕셔너리
        """
        try:

            if not state['outputs'] or \
               not state['outputs']['esValue'] or \
               not state['outputs']['esValue']['elements'] or \
               not state['outputs']['esValue']['relations']:
                return state
            
            elements = state['outputs']['esValue']['elements']
            relations = state['outputs']['esValue']['relations']
            

            # 1단계: Policy와 연관된 Event ID 수집
            policy_related_event_ids = set()
            
            for relation_id, relation in relations.items():
                if not relation:
                    continue
                    
                from_id = relation.get('from')
                to_id = relation.get('to')
                
                # Policy -> Event 관계 확인
                if from_id and to_id:
                    from_element = elements.get(from_id)
                    to_element = elements.get(to_id)
                    
                    # Policy에서 Event로의 관계
                    if (from_element and from_element.get('_type') == 'org.uengine.modeling.model.Policy' and
                        to_element and to_element.get('_type') == 'org.uengine.modeling.model.Event'):
                        policy_related_event_ids.add(to_id)
                    
                    # Event에서 Policy로의 관계
                    elif (from_element and from_element.get('_type') == 'org.uengine.modeling.model.Event' and
                          to_element and to_element.get('_type') == 'org.uengine.modeling.model.Policy'):
                        policy_related_event_ids.add(from_id)
            

            # 2단계: 삭제할 Event ID 식별 (Policy와 연관되지 않은 Event들)
            events_to_delete = set()
            
            for element_id, element in elements.items():
                if (element and 
                    element.get('_type') == 'org.uengine.modeling.model.Event' and
                    element_id not in policy_related_event_ids):
                    events_to_delete.add(element_id)
            

            # 3단계: 삭제 대상 Event들을 elements에서 제거
            for event_id in events_to_delete:
                if event_id in elements:
                    del elements[event_id]
            

            # 4단계: 삭제된 Event와 관련된 관계들을 relations에서 제거
            relations_to_delete = []
            
            for relation_id, relation in relations.items():
                if not relation:
                    continue
                    
                from_id = relation.get('from')
                to_id = relation.get('to')
                
                # 삭제된 Event를 참조하는 관계 식별
                if (from_id in events_to_delete or to_id in events_to_delete):
                    relations_to_delete.append(relation_id)
            
            # 식별된 관계들 삭제
            for relation_id in relations_to_delete:
                if relation_id in relations:
                    del relations[relation_id]
            
            return state
            
        except Exception as e:
            LoggingUtil.exception("job_util", f"[Event Cleanup Error] delete_unused_events 실행 중 오류", e)
            return state


    @staticmethod
    def add_element_ref_to_state(state):
        """
        state의 outputs.esValue.relations에 sourceElement, targetElement를 복구
        from/to ID를 이용해 elements에서 해당 요소를 찾아 참조를 재구성
        
        Args:
            state (State): 처리할 상태 객체
        """
        try:

            # esValue가 존재하는지 확인
            if not hasattr(state.outputs, 'esValue') or not state.outputs.esValue:
                return state
            
            es_value = state.outputs.esValue
            
            # relations와 elements가 존재하는지 확인
            if not hasattr(es_value, 'relations') or not es_value.relations:
                return state
            if not hasattr(es_value, 'elements') or not es_value.elements:
                return state
            
            relations = es_value.relations
            elements = es_value.elements
            
            restored_count = 0
            
            # 각 relation에 대해 sourceElement, targetElement 복구
            for relation_id, relation in relations.items():
                # from ID로 sourceElement 찾기
                if relation['from']:
                    source_element = elements.get(relation['from'])
                    if source_element:
                        relation['sourceElement'] = source_element
                        restored_count += 1
                
                # to ID로 targetElement 찾기
                if relation['to']:
                    target_element = elements.get(relation['to'])
                    if target_element:
                        relation['targetElement'] = target_element
                        restored_count += 1

            return state
        
        except Exception as e:
            LoggingUtil.exception("job_util", f"[State Restoration Error] add_element_ref_to_state 실행 중 오류", e)
            return state