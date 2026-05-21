import os
from typing import Optional

from .storage_system import StorageSystem
from .firebase_system import FirebaseSystem
from .acebase_system import AceBaseSystem
from .postgres_storage_system import PostgresStorageSystem
from ..utils.logging_util import LoggingUtil


class StorageSystemFactory:
    """Storage 시스템 팩토리 (환경에 따라 Firebase 또는 AceBase 선택)"""
    
    _storage_system: Optional[StorageSystem] = None
    
    @staticmethod
    def get_storage_type() -> str:
        """
        환경 변수에서 스토리지 타입 반환
        
        Returns:
            str: 'firebase' 또는 'acebase'
        """
        storage_type = os.getenv('STORAGE_TYPE', 'firebase').lower()
        if storage_type not in ['firebase', 'acebase', 'postgres']:
            LoggingUtil.warning("storage_system_factory", f"알 수 없는 STORAGE_TYPE: {storage_type}, 기본값 'firebase' 사용")
            return 'firebase'
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
            # Firebase 초기화 (기본값)
            service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
            database_url = os.getenv("FIREBASE_DATABASE_URL")
            
            if not service_account_path or not database_url:
                raise ValueError("Firebase 초기화를 위해 FIREBASE_SERVICE_ACCOUNT_PATH와 FIREBASE_DATABASE_URL이 필요합니다.")
            
            LoggingUtil.info("storage_system_factory", f"Firebase 시스템 초기화: {database_url}")
            StorageSystemFactory._storage_system = FirebaseSystem.initialize(
                service_account_path=service_account_path,
                database_url=database_url
            )
        
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

