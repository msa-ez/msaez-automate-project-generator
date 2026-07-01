import asyncio
import time
import os
from kubernetes import client, config

from .systems.storage_system_factory import StorageSystemFactory
from .config import Config
from .utils.logging_util import LoggingUtil
from project_generator.utils.catchable_exceptions import CATCHABLE_EXCEPTIONS

class SimpleAutoScaler:
    def __init__(self):
        self.namespace = Config.autoscaler_namespace()
        self.deployment_name = Config.autoscaler_deployment_name()
        self.service_name = Config.autoscaler_service_name()
        self.min_replicas = Config.autoscaler_min_replicas()
        self.max_replicas = Config.autoscaler_max_replicas()
        self.target_jobs_per_pod = Config.autoscaler_target_jobs_per_pod()  # 대기 작업 1개당 Pod 1개
        self.scale_check_interval = 60  # 60초마다 확인
        self.scale_up_cooldown = 120   # 스케일 업 후 2분 대기
        self.scale_down_cooldown = 1800  # 스케일 다운 후 30분 대기 (작업이 길어질 수 있으므로)
        self.scale_down_grace_period = 3600  # 스케일 다운 시 1시간 추가 유예 시간
        self.last_scale_time = 0
        self.last_scale_action = None  # 'up' or 'down'
        
        # 보수적 스케일 다운을 위한 추가 변수들
        self.scale_down_observation_count = 0  # 연속으로 스케일 다운 조건을 만족한 횟수
        self.required_scale_down_observations = 5  # 스케일 다운 실행 전 필요한 관찰 횟수
        self.last_processing_jobs_count = 0  # 이전 처리 중인 작업 수
        
        # Kubernetes 클라이언트 초기화
        try:
            # Pod 내부에서 실행되는 경우
            config.load_incluster_config()
        except CATCHABLE_EXCEPTIONS:
            try:
                # 로컬에서 테스트하는 경우
                config.load_kube_config()
            except CATCHABLE_EXCEPTIONS:
                # Docker 환경 등 Kubernetes가 없는 경우
                raise RuntimeError("Kubernetes config not available")
        
        self.apps_v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()
    
    def get_current_replicas(self) -> int:
        """현재 Deployment의 replicas 수 조회"""
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=self.deployment_name,
                namespace=self.namespace
            )
            return deployment.spec.replicas
        except CATCHABLE_EXCEPTIONS as e:
            LoggingUtil.exception("simple_autoscaler", f"현재 replicas 조회 실패: {e}", e)
            return 1
    
    def get_active_pods_count(self) -> int:
        """현재 실행 중인 Pod 수 조회 (Running 상태만)"""
        try:
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"app={self.deployment_name}"
            )
            
            active_count = 0
            for pod in pods.items:
                if pod.status.phase == 'Running':
                    active_count += 1
            
            return active_count
        except CATCHABLE_EXCEPTIONS as e:
            LoggingUtil.exception("simple_autoscaler", f"활성 Pod 수 조회 실패: {e}", e)
            return 1
    
    def set_replicas(self, target_replicas: int) -> bool:
        """Deployment의 replicas 수 변경"""
        try:
            # Deployment 조회
            deployment = self.apps_v1.read_namespaced_deployment(
                name=self.deployment_name,
                namespace=self.namespace
            )
            
            # replicas 수 변경
            deployment.spec.replicas = target_replicas
            
            # 업데이트 적용
            self.apps_v1.patch_namespaced_deployment(
                name=self.deployment_name,
                namespace=self.namespace,
                body=deployment
            )
            
            return True
            
        except CATCHABLE_EXCEPTIONS as e:
            LoggingUtil.exception("simple_autoscaler", f"replicas 변경 실패: {e}", e)
            return False
    
    def calculate_desired_replicas(self, waiting_jobs: int, processing_jobs: int) -> int:
        """대기 및 처리 중인 작업 수에 따른 목표 replicas 계산"""
        total_jobs = waiting_jobs + processing_jobs
        
        # 처리 중인 작업이 있으면 최소한 해당 수만큼은 유지
        min_required_for_processing = processing_jobs
        
        if total_jobs == 0:
            return max(self.min_replicas, min_required_for_processing)
        
        # 전체 작업 수 / 작업당 Pod 수로 계산
        desired = max(1, (total_jobs + self.target_jobs_per_pod - 1) // self.target_jobs_per_pod)
        
        # 처리 중인 작업을 위한 최소 replicas 보장
        desired = max(desired, min_required_for_processing)
        
        # min/max 범위 내로 제한
        return max(self.min_replicas, min(self.max_replicas, desired))
    
    def should_scale_up(self, current_replicas: int, desired_replicas: int) -> bool:
        """스케일 업이 필요한지 확인"""
        if desired_replicas <= current_replicas:
            return False
        
        current_time = time.time()
        time_since_last_scale = current_time - self.last_scale_time
        
        # 스케일 업 쿨다운 확인
        if self.last_scale_action == 'up' and time_since_last_scale < self.scale_up_cooldown:
            return False
        
        return True
    
    def should_scale_down(self, current_replicas: int, desired_replicas: int, processing_jobs: int) -> bool:
        """스케일 다운이 필요한지 확인 (매우 보수적 접근)"""
        if desired_replicas >= current_replicas:
            # 스케일 다운이 필요하지 않으면 관찰 카운터 리셋
            self.scale_down_observation_count = 0
            return False
        
        current_time = time.time()
        time_since_last_scale = current_time - self.last_scale_time
        
        # 기본 쿨다운 시간 확인
        if self.last_scale_action == 'down' and time_since_last_scale < self.scale_down_cooldown:
            return False
        
        # 처리 중인 작업이 있으면 스케일 다운 금지
        if processing_jobs > 0:
            self.scale_down_observation_count = 0
            return False
        
        # 연속 관찰 카운터 증가
        self.scale_down_observation_count += 1
        
        # 충분한 관찰 횟수를 만족했는지 확인
        if self.scale_down_observation_count < self.required_scale_down_observations:
            return False
        
        # 추가 유예 시간 확인 (마지막 스케일 작업 이후)
        if time_since_last_scale < self.scale_down_grace_period:
            remaining_grace = self.scale_down_grace_period - time_since_last_scale
            return False
        
        return True
    
    def is_leader_pod(self) -> bool:
        """현재 Pod가 리더인지 확인 (가장 먼저 생성된 Pod가 리더)"""
        try:
            pod_name = os.getenv('POD_ID') or os.getenv('HOSTNAME', 'unknown')
            
            # 같은 라벨을 가진 모든 Pod 조회
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"app={self.deployment_name}"
            )
            
            if not pods.items:
                return False
            
            # 생성 시간순으로 정렬하여 가장 오래된 Pod가 리더
            sorted_pods = sorted(pods.items, key=lambda p: p.metadata.creation_timestamp)
            leader_pod_name = sorted_pods[0].metadata.name
            
            is_leader = pod_name == leader_pod_name
            if is_leader:
                pass
            
            return is_leader
            
        except CATCHABLE_EXCEPTIONS as e:
            LoggingUtil.exception("simple_autoscaler", f"리더 확인 실패: {e}", e)
            return False
    
    async def run_autoscaling_loop(self):
        """자동 스케일링 메인 루프"""
        LoggingUtil.info("simple_autoscaler", "고급 자동 스케일링 시작 (처리 중인 작업 보호 기능 포함)")
        
        while True:
            try:
                # 리더 Pod만 스케일링 담당
                if not self.is_leader_pod():
                    await asyncio.sleep(self.scale_check_interval)
                    continue
                
                # 대기 및 처리 중인 작업 수 조회
                waiting_jobs = await self.get_waiting_jobs_count_async()
                processing_jobs = await self.get_processing_jobs_count_async()
                
                # 현재 replicas 및 활성 Pod 수 조회
                current_replicas = self.get_current_replicas()
                active_pods = self.get_active_pods_count()
                
                # 목표 replicas 계산
                desired_replicas = self.calculate_desired_replicas(waiting_jobs, processing_jobs)
                
                
                # 스케일링 결정
                if desired_replicas > current_replicas:
                    # 스케일 업 확인
                    if self.should_scale_up(current_replicas, desired_replicas):
                        success = self.set_replicas(desired_replicas)
                        if success:
                            self.last_scale_time = time.time()
                            self.last_scale_action = 'up'
                            self.scale_down_observation_count = 0  # 스케일 업 시 관찰 카운터 리셋
                        else:
                            LoggingUtil.warning("simple_autoscaler", f"스케일 업 실패")
                    
                elif desired_replicas < current_replicas:
                    # 스케일 다운 확인 (매우 보수적)
                    if self.should_scale_down(current_replicas, desired_replicas, processing_jobs):
                        success = self.set_replicas(desired_replicas)
                        if success:
                            self.last_scale_time = time.time()
                            self.last_scale_action = 'down'
                            self.scale_down_observation_count = 0  # 스케일 다운 후 카운터 리셋
                        else:
                            LoggingUtil.warning("simple_autoscaler", f"스케일 다운 실패")
                else:
                    # 스케일링이 필요하지 않은 경우
                    self.scale_down_observation_count = 0
                
                self.last_processing_jobs_count = processing_jobs
                await asyncio.sleep(self.scale_check_interval)
                
            except CATCHABLE_EXCEPTIONS as e:
                LoggingUtil.exception("simple_autoscaler", f"자동 스케일링 오류: {e}", e)
                await asyncio.sleep(self.scale_check_interval)

    async def get_waiting_jobs_count_async(self):
        """대기 중인 작업 수를 비동기로 계산"""
        try:
            # Firebase에서 현재 작업 데이터 조회
            requested_jobs = await StorageSystemFactory.instance().get_children_data_async(
                Config.get_requested_job_root_path()
            )
            
            if not requested_jobs:
                return 0
            
            # DecentralizedJobManager와 동일한 로직으로 대기 작업 계산
            waiting_count = 0
            for job_id, job_data in requested_jobs.items():
                # assignedPodId가 없는 작업들이 대기 중인 작업
                if job_data.get('assignedPodId') is None:
                    waiting_count += 1
            
            return waiting_count
            
        except CATCHABLE_EXCEPTIONS as e:
            LoggingUtil.exception("simple_autoscaler", "대기 작업 수 계산 오류", e)
            return 0

    async def get_processing_jobs_count_async(self):
        """처리 중인 작업 수를 비동기로 계산"""
        try:
            # Firebase에서 현재 작업 데이터 조회
            requested_jobs = await StorageSystemFactory.instance().get_children_data_async(
                Config.get_requested_job_root_path()
            )
            
            if not requested_jobs:
                return 0
            
            processing_count = 0
            current_time = time.time()
            
            for job_id, job_data in requested_jobs.items():
                assigned_pod = job_data.get('assignedPodId')
                last_heartbeat = job_data.get('lastHeartbeat', 0)
                status = job_data.get('status')
                
                # assignedPodId가 있고, 최근에 heartbeat가 있으며, processing 상태인 작업들
                if (assigned_pod and 
                    status == 'processing' and
                    current_time - last_heartbeat < 300):  # 5분 이내 heartbeat
                    processing_count += 1
            
            return processing_count
            
        except CATCHABLE_EXCEPTIONS as e:
            LoggingUtil.exception("simple_autoscaler", "처리 중인 작업 수 계산 오류", e)
            return 0

# 전역 AutoScaler 인스턴스
# Docker 환경 또는 로컬 환경에서는 autoscaler 초기화하지 않음
def _should_init_autoscaler():
    # Docker 환경 감지 (docker-compose.yml에서 POD_ID를 "docker-pod"로 설정)
    pod_id = os.getenv("POD_ID", "")
    if pod_id == "docker-pod":
        return False
    
    # 로컬 실행 환경
    if os.getenv("IS_LOCAL_RUN") == "true":
        return False
    
    # Development 환경
    if os.getenv("ENVIRONMENT") == "development":
        return False
    
    return True

if _should_init_autoscaler():
    try:
        autoscaler = SimpleAutoScaler()
    except CATCHABLE_EXCEPTIONS as e:
        LoggingUtil.warning("simple_autoscaler", f"AutoScaler 초기화 실패 (Kubernetes 없음): {e}")
        autoscaler = None
else:
    autoscaler = None

async def start_autoscaler():
    """자동 스케일러 시작"""
    if autoscaler is not None:
        await autoscaler.run_autoscaling_loop()
    else:
        # 로컬 환경에서는 autoscaler 사용 안 함
        while True:
            await asyncio.sleep(60) 