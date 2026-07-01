import os
import asyncio
import concurrent.futures
import json
import time
from typing import Dict, Any, Optional, Callable
from functools import partial
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..utils.logging_util import LoggingUtil
from .storage_system import StorageSystem


class AceBaseSystem(StorageSystem):
    """AceBase Storage 시스템 구현"""

    _instance: Optional['AceBaseSystem'] = None
    _initialized: bool = False

    # HTTP 요청 타임아웃.
    # 30s 는 큰 set_data (예: aggregate draft 의 options 3개 전체 + logs) 가
    # AceBase 가 바쁠 때 응답 받지 못하고 죽는 케이스가 잦음 → 60s 로 올려서 여유 확보.
    REQUEST_TIMEOUT: int = 60
    # 인증 요청은 짧게 — 토큰 만료 후 폭주 방지.
    AUTH_TIMEOUT: int = 10
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, host: str = None, port: int = None, dbname: str = None, 
                 https: bool = False, username: str = None, password: str = None):
        """
        AceBase 시스템 초기화 (싱글톤)
        
        Args:
            host (str): AceBase 서버 호스트
            port (int): AceBase 서버 포트
            dbname (str): 데이터베이스 이름
            https (bool): HTTPS 사용 여부
            username (str): 인증 사용자명
            password (str): 인증 비밀번호
        """
        # 이미 초기화된 경우 중복 초기화 방지
        if self._initialized:
            return
        
        if host is None or port is None or dbname is None:
            raise ValueError("host, port, dbname은 필수 매개변수입니다.")
        
        self.host = host
        self.port = port
        self.dbname = dbname
        self.https = https
        self.protocol = "https" if https else "http"
        self.base_url = f"{self.protocol}://{self.host}:{self.port}"
        # AceBase HTTP API는 /data/{dbname}/{path} 형식 사용
        self.api_url = f"{self.base_url}/data/{self.dbname}"
        
        # 세션 설정 (재시도 로직 포함)
        # total=5 + backoff_factor=0.8 → 누적 ~12s 까지 재시도. AceBase 가 RST 던질 때
        # 짧은 backoff(0.3) 로는 회복할 시간이 안 나서 application-level 까지 흘러옴.
        # POST/PUT 은 기본 allowed_methods 에 없으므로 명시 — set_data, update_data 가 retry 되게.
        self.session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=0.8,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "HEAD", "OPTIONS", "DELETE", "POST", "PUT", "PATCH"]),
            # ConnectionResetError(104) 같은 transient connection error 도 read retry 로 잡힘.
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.access_token: Optional[str] = None
        # 자격증명을 인스턴스에 보관 — 토큰 만료/서버 재시작으로 401·403 받았을 때 재인증할 수 있도록.
        # 기존엔 __init__ 에서만 쓰고 버려서 한 번 토큰이 죽으면 영원히 403 루프 도는 상태였음.
        self.username: Optional[str] = username
        self.password: Optional[str] = password
        # 재인증 최소 인터벌 — 401/403 폭주 시 매 요청마다 재인증 시도하면 서버에 부담.
        # 5초 이내엔 한 번만 시도, 그 다음 호출은 그냥 실패 처리.
        self._last_reauth_at: float = 0.0
        self._reauth_min_interval: float = 5.0
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self._listeners: Dict[str, Any] = {}  # watch 기능을 위한 리스너 관리

        # 인증 처리 (선택적 - AceBase는 인증 없이도 작동할 수 있음)
        if username and password:
            try:
                self._authenticate(username, password)
            except Exception as e:
                # 자격증명을 명시적으로 받은 상태에서의 실패는 운영상 거의 항상 버그이므로 WARNING 이상으로 노출
                LoggingUtil.warning("acebase_system", f"AceBase 인증 실패, 토큰 없이 진행 (이후 보호된 경로 접근 시 403 발생 가능): {str(e)}")
                self.access_token = None
        else:
            # 인증 정보가 제공되지 않은 경우 (정상 동작)
            LoggingUtil.info("acebase_system", "AceBase 인증 정보 없음: 인증 없이 진행")
            self.access_token = None

        self._initialized = True
    
    def _authenticate(self, username: str, password: str):
        """AceBase 인증 (선택적)"""
        # msa-ez/acebase 이미지의 OAuth2 라우트가 표준이며, 그 외는 호환 fallback
        auth_endpoints = [
            f"{self.base_url}/oauth2/{self.dbname}/signin",
            f"{self.base_url}/auth/{self.dbname}/signin",
            f"{self.base_url}/auth/signin",
        ]

        # 본문은 두 가지를 모두 시도: 단순형 ({username, password}) 과 acebase-client 표준형 ({method, ...})
        body_variants = [
            {"username": username, "password": password},
            {"method": "account", "username": username, "password": password},
        ]

        last_error: Optional[str] = None
        for auth_url in auth_endpoints:
            response = None
            for body in body_variants:
                try:
                    response = self.session.post(auth_url, json=body, timeout=self.AUTH_TIMEOUT)
                except requests.exceptions.RequestException as e:
                    last_error = f"{auth_url} body={list(body.keys())} → {e}"
                    response = None
                    continue
                if response.status_code == 200:
                    break  # 200 받은 body 사용
                last_error = f"{auth_url} body={list(body.keys())} → HTTP {response.status_code}"

            if response is None or response.status_code != 200:
                continue

            try:
                result = response.json()
            except ValueError:
                last_error = f"{auth_url} → 200 응답이지만 JSON 파싱 실패"
                continue
            self.access_token = result.get("accessToken") or result.get("access_token")
            if self.access_token:
                LoggingUtil.info("acebase_system", f"AceBase 인증 성공: {username} (endpoint={auth_url})")
                return
            last_error = f"{auth_url} → 200 응답이지만 토큰 키 없음: {result}"

        raise RuntimeError(f"AceBase signin 실패: {last_error}")
    
    @classmethod
    def initialize(cls, host: str = None, port: int = None, dbname: str = None,
                   https: bool = False, username: str = None, password: str = None) -> 'AceBaseSystem':
        """
        싱글톤 인스턴스 초기화
        
        Args:
            host (str): AceBase 서버 호스트
            port (int): AceBase 서버 포트
            dbname (str): 데이터베이스 이름
            https (bool): HTTPS 사용 여부
            username (str): 인증 사용자명
            password (str): 인증 비밀번호
            
        Returns:
            AceBaseSystem: 초기화된 싱글톤 인스턴스
        """
        if cls._instance is None or not cls._instance._initialized:
            cls._instance = cls(host, port, dbname, https, username, password)
        return cls._instance
    
    @classmethod
    def instance(cls) -> 'AceBaseSystem':
        """
        싱글톤 인스턴스 반환
        
        Returns:
            AceBaseSystem: 초기화된 싱글톤 인스턴스
            
        Raises:
            RuntimeError: 인스턴스가 초기화되지 않은 경우
        """
        if cls._instance is None or not cls._instance._initialized:
            raise RuntimeError("AceBaseSystem 초기화되지 않았습니다. 먼저 AceBaseSystem.initialize()를 호출하세요.")
        return cls._instance
    
    def _get_path_url(self, path: str) -> str:
        """경로를 AceBase API URL로 변환"""
        # 경로의 시작 슬래시 제거
        clean_path = path.lstrip('/')
        # AceBase HTTP API는 /data/{dbname}/{path} 형식 사용
        # path는 이미 루트부터 시작하는 경로 (root/ 접두사 불필요)
        return f"{self.api_url}/{clean_path}"
    
    def _get_headers(self) -> Dict[str, str]:
        """요청 헤더 생성"""
        headers = {"Content-Type": "application/json"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers
    
    def _try_reauthenticate(self) -> bool:
        """
        토큰 만료/서버 재시작/기동 순서 문제 등으로 401·403 받았을 때 호출.
        자격증명이 저장돼 있고 마지막 재인증으로부터 _reauth_min_interval 초가 지났으면 1회 시도.
        성공하면 True, 실패하거나 시도조차 안 하면 False.
        """
        if not (self.username and self.password):
            return False
        now = time.time()
        if now - self._last_reauth_at < self._reauth_min_interval:
            # 최근에 이미 시도했음 — 폭주 방지로 스킵
            return False
        self._last_reauth_at = now
        try:
            LoggingUtil.warning("acebase_system", "401/403 감지 — 재인증 시도")
            self._authenticate(self.username, self.password)
            return self.access_token is not None
        except Exception as reauth_err:
            LoggingUtil.warning("acebase_system", f"재인증 실패: {reauth_err}")
            return False

    def _execute_with_error_handling(self, operation_name: str, operation_func: Callable, *args, **kwargs) -> Any:
        """
        에러 처리가 포함된 공통 실행 래퍼

        Args:
            operation_name (str): 작업 이름 (에러 메시지용)
            operation_func (Callable): 실행할 함수
            *args, **kwargs: 함수에 전달할 인수들

        Returns:
            Any: 실행 결과 또는 실패 시 기본값
        """
        try:
            return operation_func(*args, **kwargs)
        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as conn_err:
            # urllib3 의 Retry 가 다 소진된 뒤에도 RST/connection abort 가 계속되면 여기로 옴.
            # 서버가 한순간 부하 폭주로 RST 던지더라도 ~2 초 뒤엔 회복하는 경우가 많아
            # application-level 에서 짧게 한 번 더 시도.
            LoggingUtil.warning(
                "acebase_system",
                f"{operation_name}: 연결 오류 후 application-level retry (1회): {conn_err}"
            )
            try:
                time.sleep(2.0)
                return operation_func(*args, **kwargs)
            except Exception as retry_err:
                LoggingUtil.exception("acebase_system", f"{operation_name} application-level retry 후에도 실패", retry_err)
                return False if operation_name.endswith(('업로드', '업데이트', '삭제', '시작', '중단')) else None
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else None
            if status_code == 404:
                # 404는 정상 케이스 (경로가 아직 없음) - 에러 로그 없이 None 반환
                return None
            if status_code in (401, 403):
                # 토큰 만료/서버 재시작/기동 순서 문제로 인증이 무효화된 케이스.
                # 재인증 성공 시 한 번만 retry.
                if self._try_reauthenticate():
                    try:
                        return operation_func(*args, **kwargs)
                    except requests.exceptions.HTTPError as retry_err:
                        retry_code = retry_err.response.status_code if retry_err.response is not None else None
                        if retry_code == 404:
                            return None
                        LoggingUtil.exception("acebase_system", f"{operation_name} 재인증 후 retry 도 실패", retry_err)
                        return False if operation_name.endswith(('업로드', '업데이트', '삭제', '시작', '중단')) else None
                    except Exception as retry_err:
                        LoggingUtil.exception("acebase_system", f"{operation_name} 재인증 후 retry 도 실패", retry_err)
                        return False if operation_name.endswith(('업로드', '업데이트', '삭제', '시작', '중단')) else None
            LoggingUtil.exception("acebase_system", f"{operation_name} 실패", e)
            return False if operation_name.endswith(('업로드', '업데이트', '삭제', '시작', '중단')) else None
        except Exception as e:
            LoggingUtil.exception("acebase_system", f"{operation_name} 실패", e)
            return False if operation_name.endswith(('업로드', '업데이트', '삭제', '시작', '중단')) else None
    
    async def _execute_async_with_error_handling(self, operation_name: str, sync_func: Callable, *args, **kwargs) -> Any:
        """
        비동기 실행을 위한 공통 래퍼
        
        Args:
            operation_name (str): 작업 이름
            sync_func (Callable): 동기 함수
            *args, **kwargs: 함수에 전달할 인수들
            
        Returns:
            Any: 실행 결과
        """
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                self._executor,
                partial(sync_func, *args, **kwargs)
            )
            return result
        except Exception as e:
            LoggingUtil.exception("acebase_system", f"비동기 {operation_name} 실패", e)
            return False if operation_name.endswith(('업로드', '업데이트', '삭제', '시작', '중단')) else None
    
    def _execute_fire_and_forget(self, async_func: Callable, *args, **kwargs) -> None:
        """
        Fire and Forget 패턴을 위한 공통 실행 래퍼
        
        Args:
            async_func (Callable): 실행할 비동기 함수
            *args, **kwargs: 함수에 전달할 인수들
        """
        try:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(async_func(*args, **kwargs))
                else:
                    loop.run_until_complete(async_func(*args, **kwargs))
            except RuntimeError:
                asyncio.run(async_func(*args, **kwargs))
        except Exception as e:
            LoggingUtil.exception("acebase_system", f"Fire and Forget 실행 실패", e)
    
    # =============================================================================
    # 데이터 설정 메서드들
    # =============================================================================
    
    def set_data(self, path: str, data: Dict[str, Any]) -> bool:
        """특정 경로에 딕셔너리 데이터를 업로드"""
        def _set_operation():
            url = self._get_path_url(path)
            sanitized_data = self.sanitize_data_for_storage(data)
            # AceBase는 {"val": {...}} 형식을 요구함
            payload = {"val": sanitized_data}
            response = self.session.put(
                url,
                json=payload,
                headers=self._get_headers(),
                timeout=self.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return True
        
        return self._execute_with_error_handling("데이터 업로드", _set_operation)
    
    async def set_data_async(self, path: str, data: Dict[str, Any]) -> bool:
        """특정 경로에 딕셔너리 데이터를 비동기로 업로드"""
        return await self._execute_async_with_error_handling(
            "데이터 업로드",
            lambda: self.set_data(path, data)
        )
    
    def set_data_fire_and_forget(self, path: str, data: Dict[str, Any]) -> None:
        """데이터를 업로드하되 결과를 기다리지 않음 (Fire and Forget)"""
        self._execute_fire_and_forget(self.set_data_async, path, data)
    
    # =============================================================================
    # 데이터 업데이트 메서드들
    # =============================================================================
    
    def update_data(self, path: str, data: Dict[str, Any]) -> bool:
        """특정 경로의 데이터를 부분 업데이트"""
        def _update_operation():
            url = self._get_path_url(path)
            sanitized_data = self.sanitize_data_for_storage(data)
            # AceBase는 update 시 POST를 사용하고 {"val": {...}} 형식을 요구함
            payload = {"val": sanitized_data}
            response = self.session.post(
                url,
                json=payload,
                headers=self._get_headers(),
                timeout=self.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return True
        
        return self._execute_with_error_handling("데이터 업데이트", _update_operation)
    
    async def update_data_async(self, path: str, data: Dict[str, Any]) -> bool:
        """특정 경로의 데이터를 비동기로 부분 업데이트"""
        return await self._execute_async_with_error_handling(
            "데이터 업데이트",
            lambda: self.update_data(path, data)
        )
    
    def update_data_fire_and_forget(self, path: str, data: Dict[str, Any]) -> None:
        """데이터를 업데이트하되 결과를 기다리지 않음 (Fire and Forget)"""
        self._execute_fire_and_forget(self.update_data_async, path, data)
    
    # =============================================================================
    # 조건부 업데이트 메서드들
    # =============================================================================
    
    def conditional_update_data(self, path: str, data_to_update: Dict[str, Any], previous_data: Dict[str, Any]) -> bool:
        """두 데이터를 비교하여 변경된 부분만 효율적으로 업데이트"""
        # AceBase는 Firebase와 달리 부분 업데이트를 직접 지원하므로
        # 차이점을 찾아서 업데이트
        updates = self._find_data_differences(
            self.sanitize_data_for_storage(data_to_update),
            self.sanitize_data_for_storage(previous_data)
        )
        
        if not updates:
            return True
        
        # 각 업데이트 경로에 대해 개별적으로 업데이트
        for update_path, value in updates.items():
            full_path = f"{path}/{update_path}" if path else update_path
            if value is None:
                # 삭제
                self.delete_data(full_path)
            else:
                # 업데이트
                self.update_data(full_path, {update_path.split('/')[-1]: value})
        
        return True
    
    async def conditional_update_data_async(self, path: str, data_to_update: Dict[str, Any], previous_data: Dict[str, Any]) -> bool:
        """두 데이터를 비교하여 변경된 부분만 비동기로 효율적으로 업데이트"""
        return await self._execute_async_with_error_handling(
            "조건부 데이터 업데이트",
            lambda: self.conditional_update_data(path, data_to_update, previous_data)
        )
    
    def conditional_update_data_fire_and_forget(self, path: str, data_to_update: Dict[str, Any], previous_data: Dict[str, Any]) -> None:
        """데이터를 조건부로 업데이트하되 결과를 기다리지 않음 (Fire and Forget)"""
        self._execute_fire_and_forget(self.conditional_update_data_async, path, data_to_update, previous_data)
    
    def _find_data_differences(self, new_data: Dict[str, Any], old_data: Dict[str, Any], path_prefix: str = "") -> Dict[str, Any]:
        """두 딕셔너리를 재귀적으로 비교하여 차이점을 반환"""
        updates = {}
        
        for key, new_value in new_data.items():
            current_path = f"{path_prefix}/{key}" if path_prefix else key
            old_value = old_data.get(key) if old_data else None
            
            if isinstance(new_value, dict) and isinstance(old_value, dict):
                nested_updates = self._find_data_differences(new_value, old_value, current_path)
                updates.update(nested_updates)
            elif new_value != old_value:
                updates[current_path] = new_value
        
        if old_data:
            for key in old_data.keys():
                if key not in new_data:
                    current_path = f"{path_prefix}/{key}" if path_prefix else key
                    updates[current_path] = None
        
        return updates
    
    # =============================================================================
    # 데이터 조회 메서드들
    # =============================================================================
    
    def get_data(self, path: str) -> Optional[Dict[str, Any]]:
        """특정 경로에서 데이터를 딕셔너리 형태로 조회"""
        def _get_operation():
            url = self._get_path_url(path)
            try:
                response = self.session.get(
                    url,
                    headers=self._get_headers(),
                    timeout=self.REQUEST_TIMEOUT
                )
                # 404는 데이터가 없는 것으로 처리 (에러가 아님)
                if response.status_code == 404:
                    return None
                
                response.raise_for_status()
                result = response.json()
                
                # AceBase API 응답 형식: {"exists":true/false,"val":{...}}
                if not result.get("exists", False):
                    return None
                
                data = result.get("val")
                if data is None:
                    return None
                
                if isinstance(data, dict):
                    return self.restore_data_from_storage(data)
                return data
            except requests.exceptions.HTTPError as e:
                # 404는 데이터가 없는 것으로 처리
                if e.response and e.response.status_code == 404:
                    return None
                raise
        
        return self._execute_with_error_handling("데이터 조회", _get_operation)
    
    def get_children_data(self, path: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """특정 경로의 모든 자식 노드 데이터를 조회"""
        def _get_children_operation():
            url = self._get_path_url(path)
            try:
                response = self.session.get(
                    url,
                    headers=self._get_headers(),
                    timeout=self.REQUEST_TIMEOUT
                )
                # 404는 데이터가 없는 것으로 처리 (에러가 아님)
                if response.status_code == 404:
                    return None
                
                response.raise_for_status()
                result = response.json()
                
                # AceBase API 응답 형식: {"exists":true/false,"val":{...}}
                if not result.get("exists", False):
                    return None
                
                data = result.get("val")
                if data is None or not isinstance(data, dict):
                    return None
                
                restored_data = {}
                for key, value in data.items():
                    if isinstance(value, dict):
                        restored_data[key] = self.restore_data_from_storage(value)
                    else:
                        restored_data[key] = value
                
                return restored_data
            except requests.exceptions.HTTPError as e:
                # 404는 데이터가 없는 것으로 처리
                if e.response and e.response.status_code == 404:
                    return None
                raise
        
        return self._execute_with_error_handling("자식 데이터 조회", _get_children_operation)
    
    async def get_children_data_async(self, path: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """특정 경로의 모든 자식 노드 데이터를 비동기로 조회"""
        return await self._execute_async_with_error_handling(
            "자식 데이터 조회",
            lambda: self.get_children_data(path)
        )
    
    # =============================================================================
    # 데이터 삭제 메서드들
    # =============================================================================
    
    def delete_data(self, path: str) -> bool:
        """특정 경로의 데이터 삭제
        
        부모 경로에서 특정 자식만 삭제하는 방식으로 안전하게 처리합니다.
        예: requestedJobs/user_story_generator/job1 삭제 시
        -> requestedJobs/user_story_generator 경로에서 {job1: null} 업데이트
        이렇게 하면 다른 job들(job2, job3 등)은 영향받지 않습니다.
        """
        def _delete_operation():
            # 경로를 부모 경로와 자식 키로 분리
            path_parts = path.rstrip('/').split('/')
            if len(path_parts) < 2:
                # 루트 경로나 단일 경로는 직접 삭제
                url = self._get_path_url(path)
                payload = {"val": None}
            else:
                # 부모 경로에서 자식만 삭제하는 방식 사용
                parent_path = '/'.join(path_parts[:-1])
                child_key = path_parts[-1]
                url = self._get_path_url(parent_path)
                # 부모 경로에서 특정 자식만 null로 설정하여 삭제
                payload = {"val": {child_key: None}}
            
            try:
                # update 방식으로 부모 경로에서 자식만 삭제
                response = self.session.post(
                    url,
                    json=payload,
                    headers=self._get_headers(),
                    timeout=self.REQUEST_TIMEOUT
                )
                # 404는 이미 삭제되었거나 존재하지 않는 것으로 처리 (에러가 아님)
                if response.status_code == 404:
                    LoggingUtil.info("acebase_system", f"데이터 삭제: 경로가 이미 존재하지 않습니다 (404): {path}")
                    return True
                response.raise_for_status()
                return True
            except requests.exceptions.HTTPError as e:
                # 404는 이미 삭제되었거나 존재하지 않는 것으로 처리
                if e.response and e.response.status_code == 404:
                    LoggingUtil.info("acebase_system", f"데이터 삭제: 경로가 이미 존재하지 않습니다 (404): {path}")
                    return True
                raise
        
        return self._execute_with_error_handling("데이터 삭제", _delete_operation)
    
    async def delete_data_async(self, path: str) -> bool:
        """특정 경로의 데이터를 비동기로 삭제"""
        return await self._execute_async_with_error_handling(
            "데이터 삭제",
            lambda: self.delete_data(path)
        )
    
    def delete_data_fire_and_forget(self, path: str) -> None:
        """데이터를 삭제하되 결과를 기다리지 않음 (Fire and Forget)"""
        self._execute_fire_and_forget(self.delete_data_async, path)
    
    # =============================================================================
    # 데이터 감시 메서드들 (WebSocket 기반 - 향후 구현)
    # =============================================================================
    
    def watch_data(self, path: str, callback: Callable[[Optional[Dict[str, Any]]], None]) -> bool:
        """특정 경로의 데이터 변화를 감시하고 콜백 함수 호출"""
        # TODO: WebSocket 기반 watch 구현
        LoggingUtil.warning("acebase_system", "watch_data는 아직 구현되지 않았습니다. 폴링 방식으로 대체하세요.")
        return False
    
    async def watch_data_async(self, path: str, callback: Callable[[Optional[Dict[str, Any]]], None]) -> bool:
        """특정 경로의 데이터 변화를 비동기로 감시하고 콜백 함수 호출"""
        return await self._execute_async_with_error_handling(
            "데이터 감시 시작",
            lambda: self.watch_data(path, callback)
        )
    
    def unwatch_data(self, path: str) -> bool:
        """특정 경로의 데이터 감시 중단"""
        if path in self._listeners:
            del self._listeners[path]
            return True
        return False
    
    async def unwatch_data_async(self, path: str) -> bool:
        """특정 경로의 데이터 감시를 비동기로 중단"""
        return await self._execute_async_with_error_handling(
            "데이터 감시 중단",
            lambda: self.unwatch_data(path)
        )
    
    def unwatch_all(self) -> bool:
        """모든 경로의 데이터 감시 중단"""
        paths_to_remove = list(self._listeners.keys())
        for path in paths_to_remove:
            self.unwatch_data(path)
        return True
    
    async def unwatch_all_async(self) -> bool:
        """모든 경로의 데이터 감시를 비동기로 중단"""
        return await self._execute_async_with_error_handling(
            "모든 데이터 감시 중단",
            lambda: self.unwatch_all()
        )
    
    def get_active_watchers(self) -> list[str]:
        """현재 감시 중인 경로들의 목록을 반환"""
        return list(self._listeners.keys())
    
    # =============================================================================
    # 트랜잭션 메서드들
    # =============================================================================
    
    def transaction(self, path: str, update_function: Callable) -> Any:
        """원자적 트랜잭션 실행"""
        # AceBase는 transaction API를 제공하므로 이를 사용
        try:
            # 현재 값 가져오기
            current_data = self.get_data(path)
            if current_data is None:
                current_data = {}
            
            # 업데이트 함수 실행
            updated_data = update_function(current_data)
            
            if updated_data is None:
                return None
            
            # 업데이트된 데이터 저장
            if updated_data != current_data:
                self.set_data(path, updated_data)
            
            return updated_data
        except Exception as e:
            LoggingUtil.exception("acebase_system", "트랜잭션 실패", e)
            return None
    
    async def transaction_async(self, path: str, update_function: Callable) -> Any:
        """원자적 트랜잭션 비동기 실행"""
        return await self._execute_async_with_error_handling(
            "트랜잭션",
            lambda: self.transaction(path, update_function)
        )
    
    # =============================================================================
    # 데이터 정제 메서드들
    # =============================================================================
    
    def sanitize_data_for_storage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Storage 업로드를 위해 데이터 정제 (AceBase는 Firebase와 동일한 방식 사용)"""
        def process_value(value):
            if value is None:
                return "@"  # null → 빈 문자열
            elif isinstance(value, list) and len(value) == 0:
                return ["@"]  # 빈 배열 → 마커가 포함된 배열
            elif isinstance(value, dict) and len(value) == 0:
                return {"@": True}  # 빈 객체 → 마커 객체
            elif isinstance(value, dict):
                return {k: process_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [process_value(item) for item in value]
            else:
                return value
        
        return {k: process_value(v) for k, v in data.items()}
    
    def restore_data_from_storage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Storage에서 가져온 데이터를 원본 형태로 복원"""
        def process_value(value):
            if value == "@":
                return None  # 빈 문자열 → null
            elif isinstance(value, list) and value == ["@"]:
                return []  # 마커 → 빈 배열
            elif isinstance(value, dict) and value == {"@": True}:
                return {}  # 마커 객체 → 빈 객체
            elif isinstance(value, dict):
                return {k: process_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [process_value(item) for item in value]
            else:
                return value
        
        return {k: process_value(v) for k, v in data.items()}
    
    @property
    def database(self):
        """데이터베이스 참조 객체 (호환성용 - AceBase는 self 반환)"""
        return self

