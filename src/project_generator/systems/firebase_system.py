import firebase_admin
from firebase_admin import credentials, db
from typing import Dict, Any, Optional, Callable
import os
import asyncio
import concurrent.futures
from functools import partial

from ..utils.logging_util import LoggingUtil
from .storage_system import StorageSystem

class FirebaseSystem(StorageSystem):
    _instance: Optional['FirebaseSystem'] = None
    _initialized: bool = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, service_account_path: str = None, database_url: str = None):
        """
        Firebase 시스템 초기화 (싱글톤)
        
        Args:
            service_account_path (str): Firebase 서비스 계정 JSON 키 파일 경로
            database_url (str): Firebase Realtime Database URL
        """
        # 이미 초기화된 경우 중복 초기화 방지
        if self._initialized:
            return
            
        if service_account_path is None or database_url is None:
            raise ValueError("service_account_path와 database_url은 필수 매개변수입니다.")
            
        try:
            # Firebase 앱이 이미 초기화되었는지 확인
            firebase_admin.get_app()
        except ValueError:
            # 앱이 초기화되지 않은 경우에만 초기화
            cred = credentials.Certificate(service_account_path)
            # storageBucket은 환경 변수에서 가져오거나 None (기본값 사용)
            storage_bucket = os.getenv('FIREBASE_STORAGE_BUCKET')
            init_options = {
                'databaseURL': database_url
            }
            if storage_bucket:
                # gs:// prefix 제거 (있는 경우)
                if storage_bucket.startswith('gs://'):
                    storage_bucket = storage_bucket[5:]
                init_options['storageBucket'] = storage_bucket
            firebase_admin.initialize_app(cred, init_options)
        
        self._database = db
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        # watch 기능을 위한 리스너 관리
        self._listeners: Dict[str, Any] = {}
        self._initialized = True
    

    @classmethod
    def initialize(cls, service_account_path: str, database_url: str) -> 'FirebaseSystem':
        """
        싱글톤 인스턴스 초기화
        
        Args:
            service_account_path (str): Firebase 서비스 계정 JSON 키 파일 경로
            database_url (str): Firebase Realtime Database URL
            
        Returns:
            FirebaseSystem: 초기화된 싱글톤 인스턴스
        """
        if cls._instance is None or not cls._instance._initialized:
            cls._instance = cls(service_account_path, database_url)
        return cls._instance
    
    @classmethod
    def instance(cls) -> 'FirebaseSystem':
        """
        싱글톤 인스턴스 반환
        
        Returns:
            FirebaseSystem: 초기화된 싱글톤 인스턴스
            
        Raises:
            RuntimeError: 인스턴스가 초기화되지 않은 경우
        """
        if cls._instance is None or not cls._instance._initialized:
            raise RuntimeError("FirebaseSystem 초기화되지 않았습니다. 먼저 FirebaseSystem.initialize()를 호출하세요.")
        return cls._instance
    

    # =============================================================================
    # 공통 헬퍼 메서드들
    # =============================================================================
    
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
        except firebase_admin.exceptions.NotFoundError as e:
            # 404는 정상 케이스 (경로가 아직 없음) - 에러 로그 없이 None 반환
            return None
        except Exception as e:
            LoggingUtil.exception("firebase_system", f"{operation_name} 실패", e)
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
            LoggingUtil.exception("firebase_system", f"비동기 {operation_name} 실패", e)
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
            LoggingUtil.exception("firebase_system", f"Fire and Forget 실행 실패", e)

    def _get_firebase_reference(self, path: str = None):
        """
        Firebase 참조 객체를 반환하는 공통 메서드
        
        Args:
            path (str): 데이터베이스 경로
            
        Returns:
            firebase_admin.db.Reference: Firebase 참조 객체
        """
        return self.database.reference(path) if path else self.database.reference()

    def _prepare_data_for_firebase(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Firebase 업로드용 데이터 준비 (정제 포함)
        
        Args:
            data (Dict[str, Any]): 원본 데이터
            
        Returns:
            Dict[str, Any]: Firebase용으로 정제된 데이터
        """
        def protect_trace_map(value, path=""):
            """
            traceMap 객체를 Firebase가 배열로 변환하지 않도록 보호
            traceMap의 키가 숫자 문자열인 경우, Firebase가 배열로 변환할 수 있으므로
            재귀적으로 모든 traceMap을 찾아서 보호
            """
            if isinstance(value, dict):
                result = {}
                for k, v in value.items():
                    current_path = f"{path}.{k}" if path else k
                    # traceMap 필드를 찾으면 보호
                    if k == 'traceMap' and isinstance(v, dict):
                        # traceMap의 키가 숫자 문자열인 경우, Firebase가 배열로 변환할 수 있음
                        # 하지만 비연속적인 키는 객체로 유지되어야 함
                        # Firebase가 변환하지 않도록, 키를 그대로 유지 (문자열 키는 배열로 변환되지 않음)
                        protected_trace_map = {}
                        for map_key, map_value in v.items():
                            protected_trace_map[str(map_key)] = protect_trace_map(map_value, f"{current_path}.{map_key}")
                        result[k] = protected_trace_map
                    else:
                        result[k] = protect_trace_map(v, current_path)
                return result
            elif isinstance(value, list):
                return [protect_trace_map(item, f"{path}[{i}]") for i, item in enumerate(value)]
            else:
                return value
        
        # traceMap 보호 후 sanitize
        protected_data = protect_trace_map(data)
        return self.sanitize_data_for_firebase(protected_data)

    # =============================================================================
    # 데이터 설정 메서드들
    # =============================================================================

    def set_data(self, path: str, data: Dict[str, Any]) -> bool:
        """
        특정 경로에 딕셔너리 데이터를 업로드
        
        Args:
            path (str): Firebase 데이터베이스 경로 (예: 'users/user1')
            data (Dict[str, Any]): 업로드할 딕셔너리 데이터
            
        Returns:
            bool: 성공 여부
        """
        def _set_operation():
            ref = self._get_firebase_reference(path)
            sanitized_data = self._prepare_data_for_firebase(data)
            ref.set(sanitized_data)
            return True

        return self._execute_with_error_handling("데이터 업로드", _set_operation)

    async def set_data_async(self, path: str, data: Dict[str, Any]) -> bool:
        """
        특정 경로에 딕셔너리 데이터를 비동기로 업로드
        
        Args:
            path (str): Firebase 데이터베이스 경로 (예: 'users/user1')
            data (Dict[str, Any]): 업로드할 딕셔너리 데이터
            
        Returns:
            bool: 성공 여부
        """
        return await self._execute_async_with_error_handling(
            "데이터 업로드", 
            lambda: self.set_data(path, data)
        )

    def set_data_fire_and_forget(self, path: str, data: Dict[str, Any]) -> None:
        """
        Firebase에 데이터를 업로드하되 결과를 기다리지 않음 (Fire and Forget)
        
        Args:
            path (str): Firebase 데이터베이스 경로
            data (Dict[str, Any]): 업로드할 딕셔너리 데이터
        """
        self._execute_fire_and_forget(self.set_data_async, path, data)

    # =============================================================================
    # 데이터 업데이트 메서드들
    # =============================================================================
    
    def update_data(self, path: str, data: Dict[str, Any]) -> bool:
        """
        특정 경로의 데이터를 부분 업데이트
        
        Args:
            path (str): Firebase 데이터베이스 경로
            data (Dict[str, Any]): 업데이트할 딕셔너리 데이터
            
        Returns:
            bool: 성공 여부
        """
        def _update_operation():
            ref = self._get_firebase_reference(path)
            sanitized_data = self._prepare_data_for_firebase(data)
            ref.update(sanitized_data)
            return True

        return self._execute_with_error_handling("데이터 업데이트", _update_operation)

    async def update_data_async(self, path: str, data: Dict[str, Any]) -> bool:
        """
        특정 경로의 데이터를 비동기로 부분 업데이트
        
        Args:
            path (str): Firebase 데이터베이스 경로
            data (Dict[str, Any]): 업데이트할 딕셔너리 데이터
            
        Returns:
            bool: 성공 여부
        """
        return await self._execute_async_with_error_handling(
            "데이터 업데이트",
            lambda: self.update_data(path, data)
        )

    def update_data_fire_and_forget(self, path: str, data: Dict[str, Any]) -> None:
        """
        Firebase 데이터를 업데이트하되 결과를 기다리지 않음 (Fire and Forget)
        
        Args:
            path (str): Firebase 데이터베이스 경로
            data (Dict[str, Any]): 업데이트할 딕셔너리 데이터
        """
        self._execute_fire_and_forget(self.update_data_async, path, data)

    # =============================================================================
    # 조건부 업데이트 메서드들
    # =============================================================================

    def conditional_update_data(self, path: str, data_to_update: Dict[str, Any], previous_data: Dict[str, Any]) -> bool:
        """
        두 데이터를 비교하여 변경된 부분만 효율적으로 업데이트
        
        Args:
            path (str): Firebase 데이터베이스 기본 경로
            data_to_update (Dict[str, Any]): 업데이트할 새로운 데이터
            previous_data (Dict[str, Any]): 기존 데이터
            
        Returns:
            bool: 성공 여부
        """
        def _conditional_update_operation():
            # 데이터 차이점 찾기
            sanitized_updates = self._find_data_differences(
                self.sanitize_data_for_firebase(data_to_update), 
                self.sanitize_data_for_firebase(previous_data)
            )
            
            # 변경사항이 없으면 업데이트하지 않음
            if not sanitized_updates:
                return True
            
            # 기본 경로를 기준으로 업데이트 경로 조정
            final_updates = {}
            for update_path, value in sanitized_updates.items():
                full_path = f"{path}/{update_path}" if path else update_path
                final_updates[full_path] = value
            
            # Firebase 루트에서 다중 경로 업데이트 실행
            root_ref = self._get_firebase_reference()
            root_ref.update(final_updates)
            return True

        return self._execute_with_error_handling("조건부 데이터 업데이트", _conditional_update_operation)

    async def conditional_update_data_async(self, path: str, data_to_update: Dict[str, Any], previous_data: Dict[str, Any]) -> bool:
        """
        두 데이터를 비교하여 변경된 부분만 비동기로 효율적으로 업데이트
        
        Args:
            path (str): Firebase 데이터베이스 기본 경로
            data_to_update (Dict[str, Any]): 업데이트할 새로운 데이터
            previous_data (Dict[str, Any]): 기존 데이터
            
        Returns:
            bool: 성공 여부
        """
        return await self._execute_async_with_error_handling(
            "조건부 데이터 업데이트",
            lambda: self.conditional_update_data(path, data_to_update, previous_data)
        )

    def conditional_update_data_fire_and_forget(self, path: str, data_to_update: Dict[str, Any], previous_data: Dict[str, Any]) -> None:
        """
        Firebase 데이터를 조건부로 업데이트하되 결과를 기다리지 않음 (Fire and Forget)
        
        Args:
            path (str): Firebase 데이터베이스 경로
            data_to_update (Dict[str, Any]): 업데이트할 새로운 데이터
            previous_data (Dict[str, Any]): 기존 데이터
        """
        self._execute_fire_and_forget(self.conditional_update_data_async, path, data_to_update, previous_data)

    def _find_data_differences(self, new_data: Dict[str, Any], old_data: Dict[str, Any], path_prefix: str = "") -> Dict[str, Any]:
        """
        두 딕셔너리를 재귀적으로 비교하여 차이점을 Firebase 업데이트 형태로 반환
        
        Args:
            new_data (Dict[str, Any]): 새로운 데이터
            old_data (Dict[str, Any]): 기존 데이터
            path_prefix (str): 현재 경로 접두사
            
        Returns:
            Dict[str, Any]: Firebase 업데이트용 경로-값 딕셔너리
        """
        updates = {}
        
        # 새 데이터의 모든 키를 확인
        for key, new_value in new_data.items():
            current_path = f"{path_prefix}/{key}" if path_prefix else key
            old_value = old_data.get(key) if old_data else None
            
            # 값이 딕셔너리인 경우 재귀적으로 비교
            if isinstance(new_value, dict) and isinstance(old_value, dict):
                nested_updates = self._find_data_differences(new_value, old_value, current_path)
                updates.update(nested_updates)
            # 값이 다른 경우 업데이트 필요
            elif new_value != old_value:
                updates[current_path] = new_value
        
        # 기존 데이터에만 있고 새 데이터에 없는 키들은 삭제 처리
        if old_data:
            for key in old_data.keys():
                if key not in new_data:
                    current_path = f"{path_prefix}/{key}" if path_prefix else key
                    updates[current_path] = None  # Firebase에서 삭제를 위해 None 사용
        
        return updates

    # =============================================================================
    # 데이터 조회 메서드들
    # =============================================================================

    def get_data(self, path: str) -> Optional[Dict[str, Any]]:
        """
        특정 경로에서 데이터를 딕셔너리 형태로 조회
        
        Args:
            path (str): Firebase 데이터베이스 경로 (예: 'users/user1')
            
        Returns:
            Optional[Dict[str, Any]]: 조회된 데이터 (딕셔너리 형태) 또는 None
        """
        def _get_operation():
            ref = self._get_firebase_reference(path)
            data = ref.get()
            
            if data is None:
                return None
            
            if isinstance(data, dict):
                return self.restore_data_from_firebase(data)
            return data

        return self._execute_with_error_handling("데이터 조회", _get_operation)

    def get_children_data(self, path: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        특정 경로의 모든 자식 노드 데이터를 조회
        
        Args:
            path (str): Firebase 데이터베이스 경로 (예: 'jobs/project_generator')
            
        Returns:
            Optional[Dict[str, Dict[str, Any]]]: 자식 노드들의 데이터 (key: 자식 노드명, value: 데이터)
        """
        def _get_children_operation():
            ref = self._get_firebase_reference(path)
            data = ref.get()
            
            if data is None or not isinstance(data, dict):
                return None
            
            # 각 자식 노드의 데이터를 복원
            restored_data = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    restored_data[key] = self.restore_data_from_firebase(value)
                else:
                    restored_data[key] = value
            
            return restored_data

        return self._execute_with_error_handling("자식 데이터 조회", _get_children_operation)

    async def get_children_data_async(self, path: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        특정 경로의 모든 자식 노드 데이터를 비동기로 조회
        
        Args:
            path (str): Firebase 데이터베이스 경로
            
        Returns:
            Optional[Dict[str, Dict[str, Any]]]: 자식 노드들의 데이터
        """
        return await self._execute_async_with_error_handling(
            "자식 데이터 조회",
            lambda: self.get_children_data(path)
        )

    # =============================================================================
    # 데이터 삭제 메서드들
    # =============================================================================

    def delete_data(self, path: str) -> bool:
        """
        특정 경로의 데이터 삭제
        
        Args:
            path (str): Firebase 데이터베이스 경로
            
        Returns:
            bool: 성공 여부
        """
        def _delete_operation():
            ref = self._get_firebase_reference(path)
            ref.delete()
            return True

        return self._execute_with_error_handling("데이터 삭제", _delete_operation)

    async def delete_data_async(self, path: str) -> bool:
        """
        특정 경로의 데이터를 비동기로 삭제
        
        Args:
            path (str): Firebase 데이터베이스 경로
            
        Returns:
            bool: 성공 여부
        """
        return await self._execute_async_with_error_handling(
            "데이터 삭제",
            lambda: self.delete_data(path)
        )
    
    def delete_data_fire_and_forget(self, path: str) -> None:
        """
        Firebase 데이터를 삭제하되 결과를 기다리지 않음 (Fire and Forget)
        
        Args:
            path (str): Firebase 데이터베이스 경로
        """
        self._execute_fire_and_forget(self.delete_data_async, path)

    # =============================================================================
    # 데이터 감시 메서드들
    # =============================================================================

    def watch_data(self, path: str, callback: Callable[[Optional[Dict[str, Any]]], None]) -> bool:
        """
        특정 경로의 데이터 변화를 감시하고 콜백 함수 호출
        
        Args:
            path (str): Firebase 데이터베이스 경로
            callback (Callable): 데이터 변화 시 호출할 콜백 함수
            
        Returns:
            bool: 감시 시작 성공 여부
        """
        def _watch_operation():
            # 이미 해당 경로를 감시 중인 경우 기존 리스너 제거
            if path in self._listeners:
                self.unwatch_data(path)
            
            ref = self._get_firebase_reference(path)
            
            def listener(snapshot):
                try:
                    data = snapshot.val()
                    if data is not None and isinstance(data, dict):
                        restored_data = self.restore_data_from_firebase(data)
                        callback(restored_data)
                    else:
                        callback(data)
                except Exception as e:
                    LoggingUtil.exception("firebase_system", f"콜백 함수 실행 실패", e)
            
            # 리스너 등록
            ref.listen(listener)
            self._listeners[path] = ref
            return True

        return self._execute_with_error_handling("데이터 감시 시작", _watch_operation)

    async def watch_data_async(self, path: str, callback: Callable[[Optional[Dict[str, Any]]], None]) -> bool:
        """
        특정 경로의 데이터 변화를 비동기로 감시하고 콜백 함수 호출
        
        Args:
            path (str): Firebase 데이터베이스 경로
            callback (Callable): 데이터 변화 시 호출할 콜백 함수
            
        Returns:
            bool: 감시 시작 성공 여부
        """
        return await self._execute_async_with_error_handling(
            "데이터 감시 시작",
            lambda: self.watch_data(path, callback)
        )

    def unwatch_data(self, path: str) -> bool:
        """
        특정 경로의 데이터 감시 중단
        
        Args:
            path (str): Firebase 데이터베이스 경로
            
        Returns:
            bool: 감시 중단 성공 여부
        """
        def _unwatch_operation():
            if path in self._listeners:
                ref = self._listeners[path]
                # 모든 리스너 제거
                ref.listen(None)
                del self._listeners[path]
                return True
            else:
                LoggingUtil.warning("firebase_system", f"경로 '{path}'에 대한 활성 리스너가 없습니다.")
                return False

        return self._execute_with_error_handling("데이터 감시 중단", _unwatch_operation)

    async def unwatch_data_async(self, path: str) -> bool:
        """
        특정 경로의 데이터 감시를 비동기로 중단
        
        Args:
            path (str): Firebase 데이터베이스 경로
            
        Returns:
            bool: 감시 중단 성공 여부
        """
        return await self._execute_async_with_error_handling(
            "데이터 감시 중단",
            lambda: self.unwatch_data(path)
        )

    def unwatch_all(self) -> bool:
        """
        모든 경로의 데이터 감시 중단
        
        Returns:
            bool: 모든 감시 중단 성공 여부
        """
        def _unwatch_all_operation():
            paths_to_remove = list(self._listeners.keys())
            success_count = 0
            
            for path in paths_to_remove:
                if self.unwatch_data(path):
                    success_count += 1
            
            return success_count == len(paths_to_remove)

        return self._execute_with_error_handling("모든 데이터 감시 중단", _unwatch_all_operation)

    async def unwatch_all_async(self) -> bool:
        """
        모든 경로의 데이터 감시를 비동기로 중단
        
        Returns:
            bool: 모든 감시 중단 성공 여부
        """
        return await self._execute_async_with_error_handling(
            "모든 데이터 감시 중단",
            lambda: self.unwatch_all()
        )

    def get_active_watchers(self) -> list[str]:
        """
        현재 감시 중인 경로들의 목록을 반환
        
        Returns:
            list[str]: 감시 중인 경로들의 리스트
        """
        return list(self._listeners.keys())

    # =============================================================================
    # 데이터 정제 메서드들
    # =============================================================================

    def sanitize_data_for_storage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Storage 업로드를 위해 데이터 정제 (Firebase 호환)"""
        return self.sanitize_data_for_firebase(data)

    def sanitize_data_for_firebase(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Firebase 업로드를 위해 null/빈 배열/빈 객체를 기본값으로 변환
        
        Args:
            data (Dict[str, Any]): 원본 데이터
            
        Returns:
            Dict[str, Any]: 변환된 데이터
        """
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
        """Storage에서 가져온 데이터를 원본 형태로 복원 (Firebase 호환)"""
        return self.restore_data_from_firebase(data)

    def restore_data_from_firebase(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Firebase에서 가져온 데이터를 원본 형태로 복원
        
        Args:
            data (Dict[str, Any]): Firebase에서 가져온 데이터
            
        Returns:
            Dict[str, Any]: 복원된 데이터
        """
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

    # =============================================================================
    # 트랜잭션 메서드들
    # =============================================================================
    
    def transaction(self, path: str, update_function: Callable) -> Any:
        """원자적 트랜잭션 실행"""
        ref = self._get_firebase_reference(path)
        return ref.transaction(update_function)
    
    async def transaction_async(self, path: str, update_function: Callable) -> Any:
        """원자적 트랜잭션 비동기 실행"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.transaction(path, update_function)
        )
    
    @property
    def database(self):
        """데이터베이스 참조 객체 (호환성용)"""
        return self._database