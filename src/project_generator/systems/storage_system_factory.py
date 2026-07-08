import os
from typing import Optional

from .storage_system import StorageSystem
from .acebase_system import AceBaseSystem
from .postgres_storage_system import PostgresStorageSystem
from ..utils.logging_util import LoggingUtil


class StorageSystemFactory:
    """Storage 시스템 팩토리 (환경에 따라 AceBase 또는 PostgreSQL 선택)"""

    _storage_system: Optional[StorageSystem] = None

    @staticmethod
    def get_storage_type() -> str:
        """
        환경 변수에서 스토리지 타입 반환

        Returns:
            str: 'postgres' 또는 'acebase'
        """
        storage_type = os.getenv('STORAGE_TYPE', 'postgres').lower()
        if storage_type not in ['acebase', 'postgres']:
            LoggingUtil.warning("storage_system_factory", f"알 수 없는 STORAGE_TYPE: {storage_type}, 기본값 'postgres' 사용")
            return 'postgres'
        return storage_type
    
    @staticmethod
    def initialize() -> StorageSystem:
        """
        환경에 따라 적절한 Storage 시스템 초기화
        
        Returns:
            StorageSystem: 초기화된 Storage 시스템 인스턴스
        """
        storage_type = StorageSystemFactory.get_storage_type()
        
        if storage_type == 'acebase':
            # AceBase 초기화
            host = os.getenv('ACEBASE_HOST', '127.0.0.1')
            port = int(os.getenv('ACEBASE_PORT', '5757'))
            dbname = os.getenv('ACEBASE_DB_NAME', 'mydb')
            https = os.getenv('ACEBASE_HTTPS', 'false').lower() == 'true'
            # 인증은 선택적: 환경 변수가 설정된 경우에만 인증 시도
            username = os.getenv('ACEBASE_USERNAME', None)
            password = os.getenv('ACEBASE_PASSWORD', None)
            
            LoggingUtil.info("storage_system_factory", f"AceBase 시스템 초기화: {host}:{port}/{dbname}")
            if username and password:
                LoggingUtil.info("storage_system_factory", f"AceBase 인증 정보 제공됨: {username}")
            else:
                LoggingUtil.info("storage_system_factory", "AceBase 인증 정보 없음: 인증 없이 진행")
            
            StorageSystemFactory._storage_system = AceBaseSystem.initialize(
                host=host,
                port=port,
                dbname=dbname,
                https=https,
                username=username,
                password=password
            )
        elif storage_type == 'postgres':
            # PostgreSQL 초기화 (v1.0.30 — DB-migration-plan.md §5)
            host = os.getenv('POSTGRES_HOST', '127.0.0.1')
            port = int(os.getenv('POSTGRES_PORT', '5432'))
            dbname = os.getenv('POSTGRES_DB', 'msaez')
            user = os.getenv('POSTGRES_USER', 'msaez')
            password = os.getenv('POSTGRES_PASSWORD', '')

            LoggingUtil.info("storage_system_factory", f"PostgreSQL 시스템 초기화: {host}:{port}/{dbname}")
            StorageSystemFactory._storage_system = PostgresStorageSystem.initialize(
                host=host,
                port=port,
                dbname=dbname,
                user=user,
                password=password
            )
        else:
            raise ValueError(f"지원하지 않는 STORAGE_TYPE: {storage_type} (지원: acebase, postgres)")

        return StorageSystemFactory._storage_system
    
    @staticmethod
    def instance() -> StorageSystem:
        """
        현재 초기화된 Storage 시스템 인스턴스 반환
        
        Returns:
            StorageSystem: Storage 시스템 인스턴스
            
        Raises:
            RuntimeError: 시스템이 초기화되지 않은 경우
        """
        if StorageSystemFactory._storage_system is None:
            # 자동 초기화 시도
            StorageSystemFactory.initialize()
        
        if StorageSystemFactory._storage_system is None:
            raise RuntimeError("Storage 시스템이 초기화되지 않았습니다.")
        
        return StorageSystemFactory._storage_system

