import os

class Config:
    @staticmethod
    def get_requested_job_root_path() -> str:
        return f"requestedJobs/{Config.get_namespace()}"
            
    @staticmethod
    def get_requested_job_path(job_id: str) -> str:
        return f"{Config.get_requested_job_root_path()}/{job_id}"


    @staticmethod
    def get_job_root_path() -> str:
        return f"jobs/{Config.get_namespace()}"

    @staticmethod
    def get_job_path(job_id: str) -> str:
        return f"{Config.get_job_root_path()}/{job_id}"


    @staticmethod
    def get_job_state_root_path() -> str:
        return f"jobStates/{Config.get_namespace()}"
    
    @staticmethod
    def get_job_state_path(job_id: str) -> str:
        return f"{Config.get_job_state_root_path()}/{job_id}"


    @staticmethod
    def get_namespace() -> str:
        return os.getenv('NAMESPACE')

    @staticmethod
    def get_pod_id() -> str:
        return os.getenv('POD_ID')


    @staticmethod
    def is_local_run() -> bool:
        return os.getenv('IS_LOCAL_RUN') == 'true'
    

    @staticmethod
    def autoscaler_namespace() -> str:
        return os.getenv('AUTO_SCALE_NAMESPACE', 'default')
    
    @staticmethod
    def autoscaler_deployment_name() -> str:
        return os.getenv('AUTO_SCALE_DEPLOYMENT_NAME', 'project-generator')
    
    @staticmethod
    def autoscaler_service_name() -> str:
        return os.getenv('AUTO_SCALE_SERVICE_NAME', 'project-generator-service')

    @staticmethod
    def autoscaler_min_replicas() -> int:
        return int(os.getenv('AUTO_SCALE_MIN_REPLICAS', '1'))

    @staticmethod
    def autoscaler_max_replicas() -> int:
        return int(os.getenv('AUTO_SCALE_MAX_REPLICAS', '3'))
    
    @staticmethod
    def autoscaler_target_jobs_per_pod() -> int:
        return int(os.getenv('AUTO_SCALE_TARGET_JOBS_PER_POD', '1'))
    
    @staticmethod
    def max_concurrent_jobs() -> int:
        """단일 인스턴스에서 동시에 처리할 수 있는 최대 작업 수"""
        return int(os.getenv('MAX_CONCURRENT_JOBS', '3'))
    
    @staticmethod
    def job_polling_interval() -> float:
        """작업 모니터링 폴링 간격 (초)"""
        return float(os.getenv('JOB_POLLING_INTERVAL', '2.0'))

    @staticmethod
    def job_heartbeat_interval() -> float:
        """활성 작업 heartbeat 전송 간격 (초)"""
        return float(os.getenv('JOB_HEARTBEAT_INTERVAL', '4.0'))

    @staticmethod
    def job_waiting_count_update_interval() -> float:
        """대기열 waitingJobCount 갱신 간격 (초)"""
        return float(os.getenv('JOB_WAITING_COUNT_UPDATE_INTERVAL', '6.0'))

    @staticmethod
    def job_recovery_check_interval() -> float:
        """실패 작업 복구 점검 간격 (초)"""
        return float(os.getenv('JOB_RECOVERY_CHECK_INTERVAL', '10.0'))

    @staticmethod
    def aggregate_draft_write_legacy_options() -> bool:
        """
        Aggregate Draft 결과 저장 시 기존 프론트 호환을 위해 outputs/options를 함께 저장할지 여부.
        성능 최적화(chunks) 적용 중 점진 배포를 위해 기본값은 true.
        """
        return os.getenv('AGGR_DRAFT_WRITE_LEGACY_OPTIONS', 'true').lower() == 'true'

    @staticmethod
    def get_log_level() -> str:
        """환경별 로그 레벨 반환 (DEBUG, INFO, WARNING, ERROR)"""
        if Config.is_local_run():
            return os.getenv('LOG_LEVEL', 'DEBUG')  # 로컬에서는 DEBUG 기본
        else:
            return os.getenv('LOG_LEVEL', 'INFO')   # Pod에서는 INFO 기본
    

    @staticmethod
    def get_ai_model() -> str:
        return os.getenv('AI_MODEL')
    
    @staticmethod
    def get_ai_model_vendor() -> str:
        return Config.get_ai_model().split(':')[0]
    
    @staticmethod
    def get_ai_model_name() -> str:
        return Config.get_ai_model().split(':')[1]
    
    @staticmethod
    def get_ai_model_max_input_limit() -> int:
        return int(os.getenv('AI_MODEL_MAX_INPUT_LIMIT'))
    
    @staticmethod
    def get_ai_model_max_batch_size() -> int:
        return int(os.getenv('AI_MODEL_MAX_BATCH_SIZE'))
    

    @staticmethod
    def get_ai_model_light() -> str:
        return os.getenv('AI_MODEL_LIGHT')
    
    @staticmethod
    def get_ai_model_light_vendor() -> str:
        return Config.get_ai_model_light().split(':')[0]
    
    @staticmethod
    def get_ai_model_light_name() -> str:
        return Config.get_ai_model_light().split(':')[1]

    @staticmethod
    def get_ai_model_light_max_input_limit() -> int:
        return int(os.getenv('AI_MODEL_LIGHT_MAX_INPUT_LIMIT'))
    
    @staticmethod
    def get_ai_model_light_max_batch_size() -> int:
        return int(os.getenv('AI_MODEL_LIGHT_MAX_BATCH_SIZE'))
    
    # Knowledge Base 경로 설정
    from pathlib import Path
    # __file__ = backend-generators/src/project_generator/config.py
    # parent.parent.parent = backend-generators/
    _project_root = Path(__file__).parent.parent.parent
    
    # 공유 스토리지 경로 (Kubernetes PersistentVolume 등)
    # 설정되지 않으면 기본 경로 사용
    _shared_storage = os.getenv('SHARED_STORAGE_PATH')
    if _shared_storage:
        _base_path = Path(_shared_storage)
    else:
        _base_path = _project_root
    
    # RAG 설정
    # VECTORSTORE_PATH는 SHARED_STORAGE_PATH가 있으면 그 경로를 사용, 없으면 기본 경로 사용
    _vectorstore_env = os.getenv('VECTORSTORE_PATH')
    if _vectorstore_env:
        VECTORSTORE_PATH = _vectorstore_env
    elif _shared_storage:
        # 배포 환경: PVC 경로 사용
        VECTORSTORE_PATH = str(_base_path / 'knowledge_base' / 'vectorstore')
    else:
        # 로컬 환경: 프로젝트 루트 기준 상대 경로
        VECTORSTORE_PATH = str(_project_root / 'knowledge_base' / 'vectorstore')
    
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
    
    # 표준 변환 시스템 설정
    # 임계값 0.8: 행별 청킹 + 핵심 키워드만 포함으로 유사도 향상
    # 필수 매핑은 전체 표준 원본을 직접 읽어서 global mapping 구성 (유사도 검색과 무관)
    # 각 행이 독립적으로 임베딩되고 검색 키워드(한글명, 영문명)만 포함
    # OpenAI 임베딩: "주문" vs "주문 Order" → 90%+ 예상
    STANDARD_TRANSFORMER_SCORE_THRESHOLD = float(os.getenv('STANDARD_TRANSFORMER_SCORE_THRESHOLD', '0.8'))
    
    COMPANY_STANDARDS_PATH = _base_path / 'knowledge_base' / 'company_standards'
    
    # LLM 설정 (OpenAI 실제 모델)
    DEFAULT_LLM_MODEL = os.getenv('DEFAULT_LLM_MODEL', 'gpt-4o')  # OpenAI 최신 모델
    DEFAULT_LLM_TEMPERATURE = float(os.getenv('DEFAULT_LLM_TEMPERATURE', '0.2'))  # Frontend와 동일

    # OpenAI 호환 LLM 게이트웨이 설정 (예: POSCO P-GPT)
    # 표준 openai SDK 환경변수 이름을 그대로 사용한다:
    #   OPENAI_BASE_URL — 설정 시 Chat/Responses/Models 호출이 해당 게이트웨이로 라우팅
    #   OPENAI_API_KEY  — LLM 인증 키 (게이트웨이 사용 시에는 P-GPT 발급 키)
    #   OPENAI_BASE_URL 예) http://taigpt.posco.net/gpgpta01-gpt/v1  (개발)
    #                       http://aigpt.posco.net/gpgpta01-gpt/v1   (운영)
    # 임베딩은 P-GPT 미지원이라 OpenAI 공용(api.openai.com)으로 자동 격리되며,
    # 이 때는 OPENAI_EMBEDDING_API_KEY(진짜 OpenAI 키)를 별도로 반드시 설정해야 한다.
    @staticmethod
    def get_llm_base_url() -> str:
        return os.getenv('OPENAI_BASE_URL', '').strip() or None

    @staticmethod
    def get_llm_api_key() -> str:
        return os.getenv('OPENAI_API_KEY', '').strip() or None

    @staticmethod
    def is_pgpt_enabled() -> bool:
        return bool(Config.get_llm_base_url() and Config.get_llm_api_key())

    # 임베딩 전용 OpenAI 키/URL (게이트웨이 사용 시에도 임베딩은 OpenAI 공용으로 유지)
    # OPENAI_BASE_URL이 설정된 경우 openai SDK가 그걸 기본값으로 읽어 임베딩까지
    # 엉뚱한 곳으로 보내므로, 임베딩은 반드시 명시적 base_url을 넘긴다.
    @staticmethod
    def get_embedding_api_key() -> str:
        # 게이트웨이가 켜져 있으면 OPENAI_API_KEY는 P-GPT 키이므로 쓰지 않는다.
        emb = os.getenv('OPENAI_EMBEDDING_API_KEY', '').strip()
        if emb:
            return emb
        if Config.get_llm_base_url():
            return None  # 게이트웨이 사용 중 → 전용 키 필수
        return os.getenv('OPENAI_API_KEY', '').strip() or None

    @staticmethod
    def get_embedding_base_url() -> str:
        explicit = os.getenv('OPENAI_EMBEDDING_BASE_URL', '').strip()
        if explicit:
            return explicit
        # 채팅용 게이트웨이가 켜져 있으면 임베딩은 OpenAI 공용으로 강제
        if Config.get_llm_base_url():
            return 'https://api.openai.com/v1'
        return None