"""
RAG Retriever - 공통 RAG 검색 모듈
모든 워크플로우에서 재사용 가능한 RAG 검색 기능 제공
"""
from typing import List, Dict, Optional
from pathlib import Path
import json
import sys
import threading
import os
import time
from contextlib import contextmanager
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    # Windows 환경에서는 fcntl이 없을 수 있음
    HAS_FCNTL = False

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    # 새로운 패키지로 import 시도 (deprecation warning 해결)
    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        # fallback: 기존 패키지 사용
        from langchain.vectorstores import Chroma
        from langchain.embeddings import OpenAIEmbeddings
    # Document는 langchain_core에서 import
    try:
        from langchain_core.documents import Document
    except ImportError:
        from langchain.schema import Document
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
    print("⚠️  chromadb not installed. RAG features will be disabled.")

from src.project_generator.config import Config
from src.project_generator.utils.llm_factory import create_embeddings

# 기본 유사도 임계값 (0.0~1.0)
# 자연어 + 도메인 텍스트에서 코사인 기반으로 0.3~0.4 이하를 컷으로 쓰는 경우가 많음
# 0.7은 거의 "거의 같은 문장 수준"이라 너무 높음
DEFAULT_SIM_THRESHOLD = 0.3

# 경로별 Lock 관리 (동시 접근 방지)
# RLock을 사용하여 재진입 가능하도록 함 (같은 스레드에서 중첩 호출 가능)
_path_locks: Dict[str, threading.RLock] = {}
_locks_lock = threading.Lock()  # _path_locks 자체를 보호하는 Lock

# 파일 시스템 레벨 Lock 관리 (프로세스 간 동기화)
# 여러 Pod가 동시에 같은 경로에 접근할 때 사용
_file_locks: Dict[str, any] = {}  # {path: file_handle}
_file_locks_lock = threading.Lock()


def _get_path_lock(path: str) -> threading.RLock:
    """경로별 Lock 반환 (동시 접근 방지, 재진입 가능)"""
    with _locks_lock:
        if path not in _path_locks:
            _path_locks[path] = threading.RLock()
        return _path_locks[path]


@contextmanager
def _get_file_lock(path: str):
    """
    파일 시스템 레벨 Lock 획득 (프로세스 간 동기화)
    
    여러 Pod가 동시에 같은 경로에 접근할 때 사용합니다.
    fcntl.flock()을 사용하여 파일 시스템 레벨에서 Lock을 획득합니다.
    
    Returns:
        context manager: with 문으로 사용 가능한 Lock 객체
    """
    if not HAS_FCNTL:
        # fcntl이 없으면 Lock 없이 진행 (Windows 환경 등)
        yield
        return
    
    import os
    
    lock_file_path = Path(path) / '.chromadb.lock'
    lock_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    lock_file = None
    try:
        # Lock 파일 열기 (없으면 생성)
        lock_file = open(lock_file_path, 'w')
        
        # 비차단 모드로 Lock 획득 시도
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            print(f"🔒 File lock acquired: {lock_file_path}")
            try:
                yield
            finally:
                # Lock 해제
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                print(f"🔓 File lock released: {lock_file_path}")
        except BlockingIOError:
            # Lock이 이미 사용 중이면 대기 후 재시도
            print(f"⏳ File lock is in use, waiting...: {lock_file_path}")
            lock_file.close()
            lock_file = None
            
            # 차단 모드로 Lock 획득 (대기)
            lock_file = open(lock_file_path, 'w')
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            print(f"🔒 File lock acquired after waiting: {lock_file_path}")
            try:
                yield
            finally:
                # Lock 해제
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                print(f"🔓 File lock released: {lock_file_path}")
    except Exception as e:
        print(f"⚠️  File lock error: {e}")
        # Lock 실패해도 계속 진행 (fallback)
        yield
    finally:
        if lock_file:
            try:
                lock_file.close()
            except:
                pass


class RAGRetriever:
    """
    RAG 검색 공통 클래스
    
    Knowledge Base에서 관련 정보를 검색하여 AI 프롬프트에 컨텍스트를 추가
    """
    
    def __init__(self, vectorstore_path: Optional[str] = None):
        """
        Args:
            vectorstore_path: Vector Store 경로 (None이면 Config에서 가져옴)
        """
        self.vectorstore_path = vectorstore_path or Config.VECTORSTORE_PATH
        self.vectorstore = None
        self._initialized = False
        
        # 프로세스 시작 시 umask를 0으로 설정하여 모든 새 파일이 쓰기 가능하도록 함
        # 이는 ChromaDB가 SQLite 파일을 생성할 때 readonly로 생성되는 문제를 방지
        try:
            os.umask(0)
        except:
            pass
        
        # 초기화 전에 이전 인스턴스의 캐시를 클리어 (프로세스 레벨 싱글톤 캐시 문제 방지)
        if HAS_CHROMA:
            self._clear_existing_cache()
            self._initialize_vectorstore()
    
    def _clear_existing_cache(self):
        """
        프로세스 시작 시 1회만 호출되는 캐시 클리어
        
        피드백: 런타임에 캐시를 지우지 말고, 프로세스 시작 시 1회만 초기화
        여러 Pod에서 동시에 캐시를 지우면 레이스 컨디션이 발생할 수 있음
        """
        try:
            import chromadb
            from chromadb.api.shared_system_client import SharedSystemClient
            identifier = str(self.vectorstore_path)
            
            # 해당 경로의 캐시 인스턴스만 제거 (전체 클리어는 하지 않음)
            if hasattr(SharedSystemClient, '_instances'):
                if identifier in SharedSystemClient._instances:
                    print(f"🗑️  Clearing ChromaDB cache for this path only: {identifier}")
                    try:
                        instance = SharedSystemClient._instances[identifier]
                        if hasattr(instance, 'close'):
                            instance.close()
                    except:
                        pass
                    del SharedSystemClient._instances[identifier]
        except Exception as e:
            # 캐시 클리어 실패해도 계속 진행 (초기화에서 처리)
            pass
    
    def _fix_sqlite_permissions(self, path_obj: Path):
        """SQLite 파일 및 디렉토리 권한 수정 (readonly database 오류 방지)"""
        try:
            import stat
            
            # 디렉토리 권한 확인 및 수정
            if path_obj.exists():
                current_mode = os.stat(path_obj).st_mode
                if not (current_mode & stat.S_IWRITE):
                    os.chmod(path_obj, 0o777)
            
            # 모든 하위 파일 및 디렉토리 권한 수정
            for root, dirs, files in os.walk(path_obj):
                try:
                    # 디렉토리 권한
                    os.chmod(root, 0o777)
                    for d in dirs:
                        dir_path = os.path.join(root, d)
                        os.chmod(dir_path, 0o777)
                    
                    # 파일 권한 (특히 SQLite 파일)
                    for f in files:
                        file_path = os.path.join(root, f)
                        # SQLite 파일은 쓰기 가능해야 함 (.sqlite3, .wal, .shm 포함)
                        if f.endswith('.sqlite') or f.endswith('.db') or f.endswith('.sqlite3') or f.endswith('.wal') or f.endswith('.shm'):
                            try:
                                os.chmod(file_path, 0o666)
                                print(f"✅ Fixed permissions for SQLite file: {file_path}")
                            except Exception as e:
                                print(f"⚠️  Failed to fix permissions for {file_path}: {e}")
                        else:
                            try:
                                os.chmod(file_path, 0o666)
                            except:
                                pass
                except Exception as e:
                    # 개별 파일/디렉토리 권한 수정 실패는 무시
                    pass
        except Exception as e:
            print(f"⚠️  Failed to fix SQLite permissions: {e}")
    
    def _initialize_vectorstore(self):
        """Vector Store 초기화 (동시 접근 방지를 위해 Lock 사용)"""
        # 파일 시스템 레벨 Lock 획득 (프로세스 간 동기화)
        file_lock = _get_file_lock(self.vectorstore_path)
        with file_lock:
            # 스레드 레벨 Lock 획득 (같은 프로세스 내 동기화)
            path_lock = _get_path_lock(self.vectorstore_path)
            with path_lock:
                return self._initialize_vectorstore_internal()
    
    def _initialize_vectorstore_internal(self):
        """_initialize_vectorstore의 내부 구현 (Lock 보호됨)"""
        try:
            vectorstore_path_obj = Path(self.vectorstore_path)
            if vectorstore_path_obj.exists():
                try:
                    # ChromaDB 싱글톤 캐시 확인 및 정리 (기존 인스턴스가 있으면 제거)
                    try:
                        from chromadb.api.shared_system_client import SharedSystemClient
                        identifier = str(self.vectorstore_path)
                        if hasattr(SharedSystemClient, '_instances') and identifier in SharedSystemClient._instances:
                            # 기존 인스턴스가 있으면 제거 (설정 충돌 방지)
                            del SharedSystemClient._instances[identifier]
                    except:
                        pass
                    
                    # ChromaDB 1.4.0에서는 tenant를 명시적으로 지정해야 tenants 테이블 문제를 방지할 수 있음
                    import chromadb
                    chroma_client = chromadb.PersistentClient(
                        path=str(self.vectorstore_path),
                        tenant="default_tenant",
                        database="default_database",
                        settings=chromadb.Settings(
                            anonymized_telemetry=False,
                            allow_reset=True,
                            is_persistent=True
                        )
                    )
                    # ChromaDB 초기화 전에 디렉토리 권한 강제 설정
                    try:
                        import stat
                        print(f"🔧 Setting directory permissions before loading ChromaDB...")
                        os.chmod(vectorstore_path_obj, 0o777)
                        # 부모 디렉토리들도 권한 설정
                        current_path = vectorstore_path_obj
                        while current_path != current_path.parent:
                            try:
                                if current_path.exists():
                                    os.chmod(current_path, 0o777)
                                current_path = current_path.parent
                            except:
                                break
                        time.sleep(0.5)
                        print(f"✅ Directory permissions set: {oct(os.stat(vectorstore_path_obj).st_mode)}")
                    except Exception as perm_error:
                        print(f"⚠️  Permission setting failed: {perm_error}")
                    
                    # 기존 Vector Store 로드 시 collection_name은 자동으로 찾음
                    # 하지만 tenants 테이블 문제가 있으면 복구 로직으로 넘어감
                    # 중요: umask를 0으로 설정하고 ChromaDB가 SQLite 파일을 생성할 때까지 유지
                    max_chroma_retries = 3
                    original_umask = os.umask(0)  # ChromaDB 초기화 전에 umask 설정
                    try:
                        for chroma_retry in range(max_chroma_retries):
                            try:
                                print(f"🔧 Attempting ChromaDB loading (attempt {chroma_retry + 1}/{max_chroma_retries})...")
                                self.vectorstore = Chroma(
                                    client=chroma_client,
                                    embedding_function=create_embeddings(
                                        model=Config.EMBEDDING_MODEL
                                    )
                                )
                                # ChromaDB가 SQLite 파일을 생성했을 수 있으므로 즉시 권한 확인 및 수정
                                time.sleep(0.5)  # 파일 생성 대기
                                self._fix_sqlite_permissions(vectorstore_path_obj)
                                print(f"✅ ChromaDB loaded successfully")
                                break  # 성공하면 루프 탈출
                            except Exception as chroma_init_error:
                                error_msg = str(chroma_init_error).lower()
                                if "readonly" in error_msg and chroma_retry < max_chroma_retries - 1:
                                    print(f"⚠️  ChromaDB loading failed (attempt {chroma_retry + 1}/{max_chroma_retries}): {chroma_init_error}")
                                    # SQLite 파일 권한 확인 및 수정 (이미 생성된 파일이 있을 수 있음)
                                    self._fix_sqlite_permissions(vectorstore_path_obj)
                                    # 디렉토리 권한 다시 설정
                                    try:
                                        os.chmod(vectorstore_path_obj, 0o777)
                                        # 부모 디렉토리들도 권한 설정
                                        current_path = vectorstore_path_obj
                                        while current_path != current_path.parent:
                                            try:
                                                if current_path.exists():
                                                    os.chmod(current_path, 0o777)
                                                current_path = current_path.parent
                                            except:
                                                break
                                        time.sleep(1.0)
                                    except Exception as chmod_error:
                                        print(f"⚠️  Failed to fix directory permissions: {chmod_error}")
                                else:
                                    # 마지막 시도이거나 readonly가 아닌 오류인 경우
                                    raise
                    finally:
                        # ChromaDB 초기화 완료 후 umask 복원
                        os.umask(original_umask)
                    
                    # 초기화 검증: 간단한 작업으로 데이터베이스가 정상인지 확인
                    try:
                        # 컬렉션 목록 조회로 데이터베이스 접근 테스트
                        _ = self.vectorstore._collection
                        # collection이 실제로 존재하는지 확인 (검색 테스트)
                        try:
                            # 간단한 검색으로 collection 존재 여부 확인
                            test_results = self.vectorstore.similarity_search_with_score("test", k=1)
                            self._initialized = True
                            print(f"✅ Vector Store loaded from {self.vectorstore_path}")
                        except Exception as search_test_error:
                            # 검색 실패 시 collection이 없거나 손상된 것으로 간주
                            error_msg = str(search_test_error).lower()
                            if "no such table" in error_msg or "collections" in error_msg or "database" in error_msg:
                                print(f"⚠️  Vector Store collection missing or corrupted: {search_test_error}")
                                print(f"   Attempting to repair by recreating the database...")
                                # 손상된 데이터베이스 복구 시도
                                if self._repair_vectorstore():
                                    print(f"✅ Vector Store repaired and reinitialized")
                                else:
                                    raise search_test_error
                            else:
                                # 다른 오류는 무시하고 계속 진행 (빈 collection일 수 있음)
                                self._initialized = True
                                print(f"✅ Vector Store loaded from {self.vectorstore_path} (collection may be empty)")
                    except Exception as verify_error:
                        # 데이터베이스 손상 감지 (예: tenants 테이블 없음)
                        error_msg = str(verify_error).lower()
                        if "tenants" in error_msg or "no such table" in error_msg or "database" in error_msg or "collections" in error_msg:
                            print(f"⚠️  Vector Store database corrupted: {verify_error}")
                            print(f"   Attempting to repair by recreating the database...")
                            # 손상된 데이터베이스 복구 시도
                            if self._repair_vectorstore():
                                print(f"✅ Vector Store repaired and reinitialized")
                            else:
                                raise verify_error
                        else:
                            raise verify_error
                except Exception as init_error:
                    # 초기화 실패 시 복구 시도
                    error_msg = str(init_error).lower()
                    if "tenants" in error_msg or "no such table" in error_msg or "database" in error_msg:
                        print(f"⚠️  Vector Store initialization failed (database error): {init_error}")
                        print(f"   Attempting to repair by recreating the database...")
                        if self._repair_vectorstore():
                            print(f"✅ Vector Store repaired and reinitialized")
                        else:
                            raise init_error
                    else:
                        raise init_error
            else:
                # Vector Store가 없으면 생성
                vectorstore_path_obj.mkdir(parents=True, exist_ok=True)
                
                # 파일 권한 강제 설정 (Kubernetes PVC에서 중요)
                try:
                    import stat
                    # umask를 0으로 설정하고 디렉토리 생성
                    original_umask = os.umask(0)
                    try:
                        os.chmod(vectorstore_path_obj, 0o777)
                        # 부모 디렉토리들도 권한 설정
                        current_path = vectorstore_path_obj
                        while current_path != current_path.parent:
                            try:
                                if current_path.exists():
                                    os.chmod(current_path, 0o777)
                                current_path = current_path.parent
                            except:
                                break
                    finally:
                        os.umask(original_umask)
                    time.sleep(0.5)
                    print(f"✅ Directory permissions set: {oct(os.stat(vectorstore_path_obj).st_mode)}")
                except Exception as perm_error:
                    print(f"⚠️  Permission setting failed: {perm_error}")
                
                # ChromaDB 싱글톤 캐시 확인 및 정리
                try:
                    from chromadb.api.shared_system_client import SharedSystemClient
                    identifier = str(self.vectorstore_path)
                    if hasattr(SharedSystemClient, '_instances') and identifier in SharedSystemClient._instances:
                        del SharedSystemClient._instances[identifier]
                except:
                    pass
                
                # ChromaDB 1.4.0에서는 tenant를 명시적으로 지정
                import chromadb
                chroma_client = chromadb.PersistentClient(
                    path=str(self.vectorstore_path),
                    tenant="default_tenant",
                    database="default_database",
                    settings=chromadb.Settings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                        is_persistent=True
                    )
                )
                
                # ChromaDB 초기화 (재시도 로직 포함)
                # 중요: umask를 0으로 설정하고 ChromaDB가 SQLite 파일을 생성할 때까지 유지
                max_chroma_retries = 3
                original_umask = os.umask(0)  # ChromaDB 초기화 전에 umask 설정
                try:
                    for chroma_retry in range(max_chroma_retries):
                        try:
                            print(f"🔧 Attempting ChromaDB initialization (attempt {chroma_retry + 1}/{max_chroma_retries})...")
                            self.vectorstore = Chroma(
                                client=chroma_client,
                                embedding_function=create_embeddings(
                                    model=Config.EMBEDDING_MODEL
                                )
                            )
                            # ChromaDB가 SQLite 파일을 생성했을 수 있으므로 즉시 권한 확인 및 수정
                            time.sleep(0.5)  # 파일 생성 대기
                            self._fix_sqlite_permissions(vectorstore_path_obj)
                            self._initialized = True
                            print(f"✅ Vector Store created at {self.vectorstore_path}")
                            break  # 성공하면 루프 탈출
                        except Exception as chroma_init_error:
                            error_msg = str(chroma_init_error).lower()
                            if "readonly" in error_msg and chroma_retry < max_chroma_retries - 1:
                                print(f"⚠️  ChromaDB initialization failed (attempt {chroma_retry + 1}/{max_chroma_retries}): {chroma_init_error}")
                                # SQLite 파일 권한 확인 및 수정 (이미 생성된 파일이 있을 수 있음)
                                self._fix_sqlite_permissions(vectorstore_path_obj)
                                # 디렉토리 권한 다시 설정
                                try:
                                    os.chmod(vectorstore_path_obj, 0o777)
                                    # 부모 디렉토리들도 권한 설정
                                    current_path = vectorstore_path_obj
                                    while current_path != current_path.parent:
                                        try:
                                            if current_path.exists():
                                                os.chmod(current_path, 0o777)
                                            current_path = current_path.parent
                                        except:
                                            break
                                    time.sleep(1.0)
                                except Exception as chmod_error:
                                    print(f"⚠️  Failed to fix directory permissions: {chmod_error}")
                            else:
                                # 마지막 시도이거나 readonly가 아닌 오류인 경우
                                print(f"⚠️  ChromaDB initialization failed: {chroma_init_error}")
                                raise
                finally:
                    # ChromaDB 초기화 완료 후 umask 복원
                    os.umask(original_umask)
            
            # 초기화 후 검증
            if not self._initialized:
                print(f"⚠️  Vector Store initialization incomplete: _initialized={self._initialized}")
                self.vectorstore = None
            elif not self.vectorstore:
                print(f"⚠️  Vector Store initialization incomplete: vectorstore is None")
                self._initialized = False
        except Exception as e:
            print(f"⚠️  Failed to initialize Vector Store: {e}")
            print("   RAG features will work with fallback mode.")
            self._initialized = False
            self.vectorstore = None
    
    def _repair_vectorstore(self) -> bool:
        """
        손상된 Vector Store 복구 시도
        디렉토리를 완전히 삭제하고 재생성
        
        동시 접근 방지를 위해 Lock을 사용합니다.
        
        Returns:
            복구 성공 여부
        """
        # 파일 시스템 레벨 Lock 획득 (프로세스 간 동기화)
        file_lock = _get_file_lock(self.vectorstore_path)
        with file_lock:
            # 스레드 레벨 Lock 획득 (같은 프로세스 내 동기화)
            path_lock = _get_path_lock(self.vectorstore_path)
            with path_lock:
                return self._repair_vectorstore_internal()
    
    def _repair_vectorstore_internal(self) -> bool:
        """_repair_vectorstore의 내부 구현 (Lock 보호됨)"""
        import shutil
        import time
        import os
        try:
            vectorstore_path_obj = Path(self.vectorstore_path)
            
            # 기존 vectorstore 인스턴스 정리 (먼저 정리)
            if self.vectorstore is not None:
                try:
                    # ChromaDB 클라이언트 명시적 종료
                    if hasattr(self.vectorstore, '_client'):
                        try:
                            client = self.vectorstore._client
                            if hasattr(client, 'close'):
                                client.close()
                            if hasattr(client, '_server'):
                                server = client._server
                                if hasattr(server, 'close'):
                                    server.close()
                        except:
                            pass
                    self.vectorstore = None
                except:
                    pass
            
            # ChromaDB 싱글톤 인스턴스 완전 정리 (디렉토리 삭제 전에)
            try:
                import chromadb
                import gc
                # ChromaDB의 내부 싱글톤 캐시 완전 클리어
                # SharedSystemClient._instances 딕셔너리에서 해당 경로 제거
                try:
                    from chromadb.api.shared_system_client import SharedSystemClient
                    # _instances 딕셔너리에서 현재 경로 제거
                    if hasattr(SharedSystemClient, '_instances'):
                        # 현재 경로에 대한 인스턴스 제거 및 명시적 종료
                        identifier = str(self.vectorstore_path)
                        if identifier in SharedSystemClient._instances:
                            print(f"🗑️  Removing ChromaDB instance from cache before deletion: {identifier}")
                            try:
                                instance = SharedSystemClient._instances[identifier]
                                if hasattr(instance, 'close'):
                                    instance.close()
                            except:
                                pass
                            del SharedSystemClient._instances[identifier]
                        
                        # 해당 경로의 인스턴스만 제거 (전체 캐시를 지우지 않음)
                        # 피드백: 여러 Pod에서 동시에 전체 캐시를 지우면 레이스 컨디션 발생
                        print(f"🗑️  Removing ChromaDB instance for this path only: {identifier}")
                except Exception as clear_error:
                    print(f"⚠️  Failed to clear SharedSystemClient cache: {clear_error}")
                
                # chromadb 모듈 레벨 캐시는 건드리지 않음
                # 피드백: 런타임에 캐시를 지우지 말 것
            except Exception as cleanup_error:
                print(f"⚠️  ChromaDB cleanup warning: {cleanup_error}")
            
            # 피드백: 디렉토리 삭제는 복구 시에만 수행 (다른 Pod가 읽을 수 있으므로 신중하게)
            # Lock을 확실히 획득한 상태에서만 삭제 수행
            if vectorstore_path_obj.exists():
                # 복구 시에만 디렉토리 삭제 (손상된 DB 복구를 위해 필요)
                print(f"🗑️  Repair: Removing corrupted Vector Store directory: {self.vectorstore_path}")
                print(f"⚠️  WARNING: This will delete the entire directory. Ensure no other Pod is accessing it.")
                
                try:
                    shutil.rmtree(vectorstore_path_obj)
                    print(f"✅ Corrupted directory removed")
                    
                    # 디렉토리 삭제 후 파일 시스템 동기화를 위한 대기
                    # Kubernetes PVC에서 파일 시스템 동기화가 더 오래 걸릴 수 있음
                    time.sleep(3.0)
                    
                    # 파일 시스템 동기화 (디렉토리 삭제가 완전히 반영되도록)
                    try:
                        os.sync()
                    except AttributeError:
                        # os.sync()는 Linux에서만 사용 가능, macOS/Windows에서는 무시
                        pass
                    
                    # 디렉토리가 완전히 삭제되었는지 확인
                    max_retries = 10
                    for retry in range(max_retries):
                        if not vectorstore_path_obj.exists():
                            break
                        time.sleep(0.5)
                    else:
                        print(f"⚠️  Directory still exists after deletion attempts: {self.vectorstore_path}")
                except Exception as delete_error:
                    print(f"⚠️  Failed to delete directory: {delete_error}")
                    # 삭제 실패해도 계속 진행 (재생성 시도)
            
            # 디렉토리 재생성 (권한을 즉시 설정)
            # Kubernetes PVC에서 umask가 다를 수 있으므로 명시적으로 권한 설정
            import stat
            original_umask = os.umask(0)  # umask를 0으로 설정하여 모든 권한 허용
            try:
                vectorstore_path_obj.mkdir(parents=True, exist_ok=True)
                # 생성 직후 즉시 권한 설정
                os.chmod(vectorstore_path_obj, 0o777)
                # 부모 디렉토리들도 권한 설정
                current_path = vectorstore_path_obj
                while current_path != current_path.parent:
                    try:
                        if current_path.exists():
                            os.chmod(current_path, 0o777)
                        current_path = current_path.parent
                    except:
                        break
            finally:
                os.umask(original_umask)  # 원래 umask 복원
            
            # 파일 시스템 동기화 대기
            time.sleep(2.0)
            
            # 권한 재확인 및 설정 (ChromaDB가 파일 생성하기 전에)
            try:
                # 디렉토리 권한 재확인
                current_mode = os.stat(vectorstore_path_obj).st_mode
                if not (current_mode & stat.S_IWRITE):
                    print(f"⚠️  Directory not writable, fixing permissions...")
                    os.chmod(vectorstore_path_obj, 0o777)
                
                # 하위 디렉토리와 파일도 쓰기 가능하도록 설정
                for root, dirs, files in os.walk(vectorstore_path_obj):
                    try:
                        os.chmod(root, 0o777)
                        for d in dirs:
                            os.chmod(os.path.join(root, d), 0o777)
                        for f in files:
                            os.chmod(os.path.join(root, f), 0o666)
                    except:
                        pass
                time.sleep(1.0)
            except Exception as perm_error:
                print(f"⚠️  Failed to set permissions: {perm_error}")
                # 권한 설정 실패해도 계속 진행 (PVC에서 자동으로 권한이 설정될 수 있음)
            
            # ChromaDB 클라이언트 설정: tenants 테이블 문제 방지
            # ChromaDB 1.4+ 버전에서 발생할 수 있는 문제를 방지하기 위해
            # 클라이언트 설정을 명시적으로 지정하고 싱글톤 문제 회피
            # tenant를 명시적으로 지정하여 tenants 테이블 검증 우회
            chroma_client = None
            
            # 클라이언트 생성 전에 싱글톤 캐시를 다시 한 번 완전히 클리어
            try:
                import chromadb
                import gc
                from chromadb.api.shared_system_client import SharedSystemClient
                identifier = str(self.vectorstore_path)
                
                # 모든 인스턴스 클리어 및 명시적 종료 (안전하게)
                if hasattr(SharedSystemClient, '_instances'):
                    if identifier in SharedSystemClient._instances:
                        print(f"🗑️  Removing ChromaDB instance from cache before client creation: {identifier}")
                        try:
                            instance = SharedSystemClient._instances[identifier]
                            if hasattr(instance, 'close'):
                                instance.close()
                        except:
                            pass
                        del SharedSystemClient._instances[identifier]
                    
                    # 모든 인스턴스 클리어 및 명시적 종료 (더 안전한 방법)
                    for key, instance in list(SharedSystemClient._instances.items()):
                        try:
                            if hasattr(instance, 'close'):
                                instance.close()
                        except:
                            pass
                    # 해당 경로의 인스턴스만 제거 (전체 캐시를 지우지 않음)
                    identifier = str(self.vectorstore_path)
                    if identifier in SharedSystemClient._instances:
                        try:
                            instance = SharedSystemClient._instances[identifier]
                            if hasattr(instance, 'close'):
                                instance.close()
                        except:
                            pass
                        del SharedSystemClient._instances[identifier]
                    print(f"✅ ChromaDB instance removed from cache for this path only")
                
                # chromadb 모듈 레벨 캐시는 건드리지 않음
                # 피드백: 런타임에 캐시를 지우지 말 것
            except Exception as cache_clear_error:
                print(f"⚠️  Failed to clear SharedSystemClient cache before client creation: {cache_clear_error}")
            
            try:
                import chromadb
                # ChromaDB 클라이언트 설정
                # tenant를 명시적으로 지정하여 tenants 테이블 검증 문제 방지
                # ChromaDB 1.4.0에서는 기본 tenant가 "default_tenant"이지만,
                # 명시적으로 지정하면 tenants 테이블 검증을 우회할 수 있음
                print(f"🔧 Creating ChromaDB client for: {self.vectorstore_path}")
                chroma_client = chromadb.PersistentClient(
                    path=str(self.vectorstore_path),
                    tenant="default_tenant",
                    database="default_database",
                    settings=chromadb.Settings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                        is_persistent=True
                    )
                )
                print(f"✅ ChromaDB client created successfully")
            except Exception as client_error:
                # ChromaDB 클라이언트 설정 실패 시 기본 설정 사용
                error_msg = str(client_error)
                print(f"⚠️  ChromaDB client configuration failed: {error_msg}")
                
                # "already exists" 오류인 경우, 캐시를 더 강력하게 클리어하고 재시도
                if "already exists" in error_msg.lower():
                    print(f"   Detected existing instance conflict. Clearing cache more aggressively...")
                    try:
                        from chromadb.api.shared_system_client import SharedSystemClient
                        # 모든 인스턴스 클리어 및 명시적 종료
                        if hasattr(SharedSystemClient, '_instances'):
                            for key, instance in list(SharedSystemClient._instances.items()):
                                try:
                                    if hasattr(instance, 'close'):
                                        instance.close()
                                except:
                                    pass
                            # 해당 경로의 인스턴스만 제거 (전체 캐시를 지우지 않음)
                            identifier = str(self.vectorstore_path)
                            if identifier in SharedSystemClient._instances:
                                try:
                                    instance = SharedSystemClient._instances[identifier]
                                    if hasattr(instance, 'close'):
                                        instance.close()
                                except:
                                    pass
                                del SharedSystemClient._instances[identifier]
                        time.sleep(1.0)  # 짧은 대기 시간
                        
                        # 재시도
                        chroma_client = chromadb.PersistentClient(
                            path=str(self.vectorstore_path),
                            tenant="default_tenant",
                            database="default_database",
                            settings=chromadb.Settings(
                                anonymized_telemetry=False,
                                allow_reset=True,
                                is_persistent=True
                            )
                        )
                    except Exception as retry_error:
                        print(f"   Aggressive retry also failed: {retry_error}")
                        chroma_client = None
                # tenants 테이블 오류인 경우, tenant 없이 재시도
                elif "tenants" in error_msg.lower() or "no such table" in error_msg.lower():
                    print(f"   Attempting to create client without tenant validation...")
                    try:
                        # tenant 없이 클라이언트 생성 시도 (ChromaDB가 자동으로 처리)
                        chroma_client = chromadb.PersistentClient(
                            path=str(self.vectorstore_path),
                            settings=chromadb.Settings(
                                anonymized_telemetry=False,
                                allow_reset=True,
                                is_persistent=True
                            )
                        )
                    except Exception as retry_error:
                        print(f"   Retry also failed: {retry_error}")
                        chroma_client = None
                else:
                    chroma_client = None
            
            # Chroma 인스턴스 생성 전에 디렉토리 권한 최종 확인 및 강제 설정
            try:
                import stat
                # 디렉토리 권한 강제 설정 (ChromaDB가 SQLite 파일을 생성하기 전에)
                print(f"🔧 Setting directory permissions before ChromaDB initialization...")
                os.chmod(vectorstore_path_obj, 0o777)
                # 부모 디렉토리들도 권한 설정
                current_path = vectorstore_path_obj
                while current_path != current_path.parent:
                    try:
                        if current_path.exists():
                            os.chmod(current_path, 0o777)
                        current_path = current_path.parent
                    except:
                        break
                
                # 현재 권한 확인
                current_mode = os.stat(vectorstore_path_obj).st_mode
                if not (current_mode & stat.S_IWRITE):
                    print(f"⚠️  Directory still not writable after chmod, attempting umask fix...")
                    # umask를 0으로 설정하고 다시 시도
                    original_umask = os.umask(0)
                    try:
                        os.chmod(vectorstore_path_obj, 0o777)
                    finally:
                        os.umask(original_umask)
                
                time.sleep(1.0)  # 권한 설정이 완전히 반영되도록 대기
                print(f"✅ Directory permissions set: {oct(os.stat(vectorstore_path_obj).st_mode)}")
            except Exception as perm_check_error:
                print(f"⚠️  Permission check failed: {perm_check_error}")
                import traceback
                traceback.print_exc()
            
            # 디렉토리가 없으면 생성
            if not vectorstore_path_obj.exists():
                print(f"🔧 Creating directory: {vectorstore_path_obj}")
                vectorstore_path_obj.mkdir(parents=True, exist_ok=True)
            
            # 새로운 Vector Store 생성 (재시도 로직 포함)
            # 중요: umask를 0으로 설정하고 ChromaDB가 SQLite 파일을 생성할 때까지 유지
            max_chroma_retries = 3
            original_umask = os.umask(0)  # ChromaDB 초기화 전에 umask 설정
            try:
                # ChromaDB 초기화 전에 미리 SQLite 파일을 생성하고 권한 설정
                # ChromaDB의 Rust 바인딩이 파일을 생성할 때 readonly로 생성되는 문제를 방지하기 위해
                # Python에서 미리 파일을 생성하고 권한을 설정
                print(f"🔧 Pre-creating SQLite files with proper permissions...")
                potential_sqlite_files = [
                    vectorstore_path_obj / "chroma.sqlite3",
                    vectorstore_path_obj / "chroma.sqlite3-wal",
                    vectorstore_path_obj / "chroma.sqlite3-shm",
                ]
                
                # 디렉토리와 부모 디렉토리 권한 강제 설정
                for sqlite_path in potential_sqlite_files:
                    if sqlite_path.parent.exists():
                        os.chmod(sqlite_path.parent, 0o777)
                    # 부모 디렉토리들도 권한 설정
                    current_path = sqlite_path.parent
                    while current_path != current_path.parent:
                        try:
                            if current_path.exists():
                                os.chmod(current_path, 0o777)
                            current_path = current_path.parent
                        except:
                            break
                    
                    # SQLite 파일을 미리 생성하고 권한 설정 (ChromaDB가 덮어쓸 수 있지만 권한은 유지됨)
                    if not sqlite_path.exists():
                        try:
                            sqlite_path.touch()
                            os.chmod(sqlite_path, 0o666)
                            print(f"✅ Pre-created SQLite file: {sqlite_path}")
                        except Exception as create_error:
                            print(f"⚠️  Failed to pre-create {sqlite_path}: {create_error}")
                    else:
                        # 이미 존재하는 파일도 권한 확인 및 수정
                        try:
                            os.chmod(sqlite_path, 0o666)
                            print(f"✅ Fixed permissions for existing SQLite file: {sqlite_path}")
                        except Exception as chmod_error:
                            print(f"⚠️  Failed to fix permissions for {sqlite_path}: {chmod_error}")
                
                time.sleep(0.5)  # 파일 생성 및 권한 설정이 완전히 반영되도록 대기
                
                for chroma_retry in range(max_chroma_retries):
                    try:
                        print(f"🔧 Attempting ChromaDB initialization (attempt {chroma_retry + 1}/{max_chroma_retries})...")
                        if chroma_client:
                            # 명시적 클라이언트 사용
                            self.vectorstore = Chroma(
                                client=chroma_client,
                                embedding_function=create_embeddings(
                                    model=Config.EMBEDDING_MODEL
                                )
                            )
                        else:
                            # 기본 설정 사용
                            self.vectorstore = Chroma(
                                persist_directory=str(self.vectorstore_path),
                                embedding_function=create_embeddings(
                                    model=Config.EMBEDDING_MODEL
                                )
                            )
                        # ChromaDB가 SQLite 파일을 생성/덮어썼을 수 있으므로 즉시 권한 확인 및 수정
                        time.sleep(0.5)  # 파일 생성 대기
                        # ChromaDB 초기화 후 생성된 모든 SQLite 파일의 권한을 즉시 수정
                        for sqlite_path in potential_sqlite_files:
                            if sqlite_path.exists():
                                try:
                                    os.chmod(sqlite_path, 0o666)
                                    print(f"✅ Fixed permissions after ChromaDB init: {sqlite_path}")
                                except Exception as post_chmod_error:
                                    print(f"⚠️  Failed to fix permissions after init for {sqlite_path}: {post_chmod_error}")
                        self._fix_sqlite_permissions(vectorstore_path_obj)
                        print(f"✅ ChromaDB initialized successfully")
                        break  # 성공하면 루프 탈출
                    except Exception as chroma_init_error:
                        error_msg = str(chroma_init_error).lower()
                        if "readonly" in error_msg and chroma_retry < max_chroma_retries - 1:
                            print(f"⚠️  ChromaDB initialization failed (attempt {chroma_retry + 1}/{max_chroma_retries}): {chroma_init_error}")
                            # SQLite 파일 권한 확인 및 수정 (이미 생성된 파일이 있을 수 있음)
                            self._fix_sqlite_permissions(vectorstore_path_obj)
                            # 디렉토리 권한 다시 설정
                            try:
                                os.chmod(vectorstore_path_obj, 0o777)
                                # 부모 디렉토리들도 권한 설정
                                current_path = vectorstore_path_obj
                                while current_path != current_path.parent:
                                    try:
                                        if current_path.exists():
                                            os.chmod(current_path, 0o777)
                                        current_path = current_path.parent
                                    except:
                                        break
                                time.sleep(1.0)
                            except Exception as chmod_error:
                                print(f"⚠️  Failed to fix directory permissions: {chmod_error}")
                        else:
                            # 마지막 시도이거나 readonly가 아닌 오류인 경우
                            print(f"⚠️  ChromaDB initialization failed: {chroma_init_error}")
                            raise
            finally:
                # ChromaDB 초기화 완료 후 umask 복원
                os.umask(original_umask)
            
            # Chroma 인스턴스 생성 후 생성된 파일들의 권한 확인 및 수정
            try:
                time.sleep(0.5)  # ChromaDB가 파일을 생성할 시간을 줌
                self._fix_sqlite_permissions(vectorstore_path_obj)
            except Exception as post_perm_error:
                print(f"⚠️  Failed to set post-creation permissions: {post_perm_error}")
            
            # 초기화 검증: 컬렉션 접근 테스트
            try:
                _ = self.vectorstore._collection
                self._initialized = True
                print(f"✅ Vector Store repaired and reinitialized")
                return True
            except Exception as verify_error:
                print(f"⚠️  Vector Store repair verification failed: {verify_error}")
                self._initialized = False
                self.vectorstore = None
                return False
        except Exception as e:
            print(f"⚠️  Failed to repair Vector Store: {e}")
            import traceback
            traceback.print_exc()
            self._initialized = False
            self.vectorstore = None
            return False
    
    def clear_vectorstore(self) -> bool:
        """
        Vector Store의 모든 문서를 삭제 (컬렉션 클리어)
        
        ChromaDB 1.4.0에서는 delete_collection()만으로는 SQLite 데이터베이스가 완전히 초기화되지 않을 수 있으므로,
        디렉토리를 완전히 삭제하고 재생성하는 것이 안전합니다.
        
        동시 접근 방지를 위해 Lock을 사용합니다.
        
        Returns:
            성공 여부
        """
        if not self._initialized or not self.vectorstore:
            print("⚠️  Vector Store not initialized. Cannot clear.")
            return False
        
        print(f"🔒 Acquiring lock for clear_vectorstore: {self.vectorstore_path}")
        # 파일 시스템 레벨 Lock 획득 (프로세스 간 동기화)
        file_lock = _get_file_lock(self.vectorstore_path)
        with file_lock:
            # 스레드 레벨 Lock 획득 (같은 프로세스 내 동기화)
            path_lock = _get_path_lock(self.vectorstore_path)
            print(f"🔒 Lock acquired, starting clear operation...")
            with path_lock:
                print(f"🔒 Lock acquired, calling _clear_vectorstore_internal...")
                result = self._clear_vectorstore_internal()
                print(f"🔒 clear_vectorstore completed, result: {result}")
                return result
    
    def _clear_vectorstore_internal(self) -> bool:
        """clear_vectorstore의 내부 구현 (Lock 보호됨)"""
        try:
            print(f"🔧 Starting _clear_vectorstore_internal for: {self.vectorstore_path}")
            # ChromaDB 컬렉션 삭제 시도
            try:
                print(f"🔧 Attempting to delete collection...")
                self.vectorstore.delete_collection()
                print(f"🗑️  Vector Store collection deleted: {self.vectorstore_path}")
            except Exception as delete_error:
                print(f"⚠️  Failed to delete collection (will use directory deletion): {delete_error}")
            
            # ChromaDB 싱글톤 캐시 완전 클리어 (디렉토리 삭제 전에)
            try:
                import chromadb
                import gc
                from chromadb.api.shared_system_client import SharedSystemClient
                identifier = str(self.vectorstore_path)
                
                # 모든 인스턴스 클리어
                if hasattr(SharedSystemClient, '_instances'):
                    if identifier in SharedSystemClient._instances:
                        print(f"🗑️  Removing ChromaDB instance from cache before clear: {identifier}")
                        del SharedSystemClient._instances[identifier]
                    # 해당 경로의 인스턴스만 제거 (전체 캐시를 지우지 않음)
                    # 피드백: 여러 Pod에서 동시에 전체 캐시를 지우면 레이스 컨디션 발생
                    print(f"✅ ChromaDB instance removed from cache for this path only")
                
                # chromadb 모듈 레벨 캐시는 건드리지 않음
                # 피드백: 런타임에 캐시를 지우지 말 것
            except Exception as cache_clear_error:
                print(f"⚠️  Failed to clear SharedSystemClient cache: {cache_clear_error}")
            
            # 기존 vectorstore 인스턴스 정리
            self.vectorstore = None
            self._initialized = False
            
            # 디렉토리 완전 삭제 및 재생성 (SQLite 초기화 문제 방지)
            import shutil
            import os
            import time
            vectorstore_path_obj = Path(self.vectorstore_path)
            
            # 피드백: 디렉토리 삭제는 복구 시에만 수행
            # clear는 컬렉션만 삭제하고 디렉토리는 유지 (다른 Pod가 읽을 수 있음)
            # 디렉토리 삭제는 _repair_vectorstore_internal에서만 수행
            print(f"ℹ️  Clear operation: collection deleted, directory preserved for safety")
            
            # 디렉토리 재생성 (권한을 즉시 설정)
            # Kubernetes PVC에서 umask가 다를 수 있으므로 명시적으로 권한 설정
            import stat
            original_umask = os.umask(0)  # umask를 0으로 설정하여 모든 권한 허용
            try:
                vectorstore_path_obj.mkdir(parents=True, exist_ok=True)
                # 생성 직후 즉시 권한 설정
                os.chmod(vectorstore_path_obj, 0o777)
                # 부모 디렉토리들도 권한 설정
                current_path = vectorstore_path_obj
                while current_path != current_path.parent:
                    try:
                        if current_path.exists():
                            os.chmod(current_path, 0o777)
                        current_path = current_path.parent
                    except:
                        break
            finally:
                os.umask(original_umask)  # 원래 umask 복원
            
            # 파일 시스템 동기화 대기
            time.sleep(2.0)
            
            # 권한 재확인 및 설정 (ChromaDB가 파일 생성하기 전에)
            try:
                # 디렉토리 권한 재확인
                current_mode = os.stat(vectorstore_path_obj).st_mode
                if not (current_mode & stat.S_IWRITE):
                    print(f"⚠️  Directory not writable, fixing permissions...")
                    os.chmod(vectorstore_path_obj, 0o777)
                
                # 하위 디렉토리와 파일도 쓰기 가능하도록 설정
                for root, dirs, files in os.walk(vectorstore_path_obj):
                    try:
                        os.chmod(root, 0o777)
                        for d in dirs:
                            os.chmod(os.path.join(root, d), 0o777)
                        for f in files:
                            os.chmod(os.path.join(root, f), 0o666)
                    except:
                        pass
                time.sleep(1.0)
            except Exception as perm_error:
                print(f"⚠️  Failed to set permissions: {perm_error}")
            
            # 클라이언트 생성 전에 캐시를 완전히 클리어 (중요!)
            import chromadb
            import gc
            try:
                from chromadb.api.shared_system_client import SharedSystemClient
                identifier = str(self.vectorstore_path)
                
                # 모든 인스턴스 클리어 및 명시적 종료
                if hasattr(SharedSystemClient, '_instances'):
                    if identifier in SharedSystemClient._instances:
                        print(f"🗑️  Removing ChromaDB instance from cache before client creation: {identifier}")
                        try:
                            instance = SharedSystemClient._instances[identifier]
                            if hasattr(instance, 'close'):
                                instance.close()
                        except:
                            pass
                        del SharedSystemClient._instances[identifier]
                    
                    # 해당 경로의 인스턴스만 제거 (전체 캐시를 지우지 않음)
                    # 피드백: 여러 Pod에서 동시에 전체 캐시를 지우면 레이스 컨디션 발생
                    if identifier in SharedSystemClient._instances:
                        try:
                            instance = SharedSystemClient._instances[identifier]
                            if hasattr(instance, 'close'):
                                instance.close()
                        except:
                            pass
                        del SharedSystemClient._instances[identifier]
                    print(f"✅ ChromaDB instance removed from cache for this path only")
                
                # chromadb 모듈 레벨 캐시는 건드리지 않음
                # 피드백: 런타임에 캐시를 지우지 말 것
            except Exception as cache_clear_error:
                print(f"⚠️  Failed to clear SharedSystemClient cache before client creation: {cache_clear_error}")
            
            # ChromaDB 클라이언트 생성 (재시도 로직 포함)
            max_retries = 3
            for retry in range(max_retries):
                try:
                    chroma_client = chromadb.PersistentClient(
                        path=str(self.vectorstore_path),
                        tenant="default_tenant",
                        database="default_database",
                        settings=chromadb.Settings(
                            anonymized_telemetry=False,
                            allow_reset=True,
                            is_persistent=True
                        )
                    )
                    print(f"✅ ChromaDB client created successfully (attempt {retry + 1}/{max_retries})")
                    break
                except Exception as client_error:
                    error_msg = str(client_error).lower()
                    if "already exists" in error_msg:
                        # 캐시를 다시 클리어하고 재시도
                        print(f"⚠️  ChromaDB client creation failed (attempt {retry + 1}/{max_retries}): {client_error}")
                        try:
                            from chromadb.api.shared_system_client import SharedSystemClient
                            if hasattr(SharedSystemClient, '_instances'):
                                # 해당 경로의 인스턴스만 제거 (전체 캐시를 지우지 않음)
                                identifier = str(self.vectorstore_path)
                                if identifier in SharedSystemClient._instances:
                                    try:
                                        instance = SharedSystemClient._instances[identifier]
                                        if hasattr(instance, 'close'):
                                            instance.close()
                                    except:
                                        pass
                                    del SharedSystemClient._instances[identifier]
                            # chromadb 모듈 레벨 캐시는 건드리지 않음
                            time.sleep(1.0)  # 짧은 대기 시간
                        except:
                            pass
                        if retry == max_retries - 1:
                            raise
                    else:
                        raise
            
            # ChromaDB 초기화 전에 디렉토리 명시적 생성 및 권한 설정
            import stat
            vectorstore_path_obj = Path(self.vectorstore_path)
            
            # 디렉토리가 없으면 생성
            if not vectorstore_path_obj.exists():
                print(f"🔧 Creating directory: {vectorstore_path_obj}")
                vectorstore_path_obj.mkdir(parents=True, exist_ok=True)
            
            # 디렉토리 및 부모 디렉토리 권한 강제 설정
            try:
                print(f"🔧 Setting directory permissions before ChromaDB initialization...")
                # umask를 0으로 설정하고 디렉토리 권한 설정
                original_umask = os.umask(0)
                try:
                    os.chmod(vectorstore_path_obj, 0o777)
                    # 부모 디렉토리들도 권한 설정
                    current_path = vectorstore_path_obj
                    while current_path != current_path.parent:
                        try:
                            if current_path.exists():
                                os.chmod(current_path, 0o777)
                            current_path = current_path.parent
                        except:
                            break
                finally:
                    os.umask(original_umask)
                time.sleep(1.0)  # 권한 설정이 완전히 반영되도록 대기
                print(f"✅ Directory permissions set: {oct(os.stat(vectorstore_path_obj).st_mode)}")
            except Exception as perm_error:
                print(f"⚠️  Permission setting failed: {perm_error}")
                import traceback
                traceback.print_exc()
            
            # ChromaDB 초기화 (재시도 로직 포함)
            # 중요: umask를 0으로 설정하고 ChromaDB가 SQLite 파일을 생성할 때까지 유지
            # 환경 변수도 설정하여 Rust 바인딩에 영향을 줌
            max_chroma_retries = 3
            original_umask = os.umask(0)  # ChromaDB 초기화 전에 umask 설정
            original_umask_env = os.environ.get('UMASK')
            try:
                # 환경 변수로 umask 설정 (Rust 바인딩에 영향을 줄 수 있음)
                os.environ['UMASK'] = '0000'
                
                # ChromaDB 초기화 전에 미리 SQLite 파일을 생성하고 권한 설정
                # ChromaDB의 Rust 바인딩이 파일을 생성할 때 readonly로 생성되는 문제를 방지하기 위해
                # Python에서 미리 파일을 생성하고 권한을 설정
                print(f"🔧 Pre-creating SQLite files with proper permissions...")
                potential_sqlite_files = [
                    vectorstore_path_obj / "chroma.sqlite3",
                    vectorstore_path_obj / "chroma.sqlite3-wal",
                    vectorstore_path_obj / "chroma.sqlite3-shm",
                ]
                
                # 디렉토리와 부모 디렉토리 권한 강제 설정
                for sqlite_path in potential_sqlite_files:
                    if sqlite_path.parent.exists():
                        os.chmod(sqlite_path.parent, 0o777)
                    # 부모 디렉토리들도 권한 설정
                    current_path = sqlite_path.parent
                    while current_path != current_path.parent:
                        try:
                            if current_path.exists():
                                os.chmod(current_path, 0o777)
                            current_path = current_path.parent
                        except:
                            break
                    
                    # SQLite 파일을 미리 생성하고 권한 설정 (ChromaDB가 덮어쓸 수 있지만 권한은 유지됨)
                    if not sqlite_path.exists():
                        try:
                            sqlite_path.touch()
                            os.chmod(sqlite_path, 0o666)
                            print(f"✅ Pre-created SQLite file: {sqlite_path}")
                        except Exception as create_error:
                            print(f"⚠️  Failed to pre-create {sqlite_path}: {create_error}")
                    else:
                        # 이미 존재하는 파일도 권한 확인 및 수정
                        try:
                            os.chmod(sqlite_path, 0o666)
                            print(f"✅ Fixed permissions for existing SQLite file: {sqlite_path}")
                        except Exception as chmod_error:
                            print(f"⚠️  Failed to fix permissions for {sqlite_path}: {chmod_error}")
                
                time.sleep(0.5)  # 파일 생성 및 권한 설정이 완전히 반영되도록 대기
                
                for chroma_retry in range(max_chroma_retries):
                    try:
                        print(f"🔧 Attempting ChromaDB initialization (attempt {chroma_retry + 1}/{max_chroma_retries})...")
                        self.vectorstore = Chroma(
                            client=chroma_client,
                            embedding_function=create_embeddings(
                                model=Config.EMBEDDING_MODEL
                            )
                        )
                        # ChromaDB가 SQLite 파일을 생성/덮어썼을 수 있으므로 즉시 권한 확인 및 수정
                        time.sleep(0.5)  # 파일 생성 대기
                        # ChromaDB 초기화 후 생성된 모든 SQLite 파일의 권한을 즉시 수정
                        for sqlite_path in potential_sqlite_files:
                            if sqlite_path.exists():
                                try:
                                    os.chmod(sqlite_path, 0o666)
                                    print(f"✅ Fixed permissions after ChromaDB init: {sqlite_path}")
                                except Exception as post_chmod_error:
                                    print(f"⚠️  Failed to fix permissions after init for {sqlite_path}: {post_chmod_error}")
                        self._fix_sqlite_permissions(vectorstore_path_obj)
                        print(f"✅ ChromaDB initialized successfully")
                        break  # 성공하면 루프 탈출
                    except Exception as chroma_init_error:
                        error_msg = str(chroma_init_error).lower()
                        if "readonly" in error_msg and chroma_retry < max_chroma_retries - 1:
                            print(f"⚠️  ChromaDB initialization failed (attempt {chroma_retry + 1}/{max_chroma_retries}): {chroma_init_error}")
                            # SQLite 파일 권한 확인 및 수정 (이미 생성된 파일이 있을 수 있음)
                            self._fix_sqlite_permissions(vectorstore_path_obj)
                            # 디렉토리 권한 다시 설정
                            try:
                                os.chmod(vectorstore_path_obj, 0o777)
                                # 부모 디렉토리들도 권한 설정
                                current_path = vectorstore_path_obj
                                while current_path != current_path.parent:
                                    try:
                                        if current_path.exists():
                                            os.chmod(current_path, 0o777)
                                        current_path = current_path.parent
                                    except:
                                        break
                                time.sleep(1.0)
                            except Exception as chmod_error:
                                print(f"⚠️  Failed to fix directory permissions: {chmod_error}")
                        else:
                            # 마지막 시도이거나 readonly가 아닌 오류인 경우
                            print(f"⚠️  ChromaDB initialization failed: {chroma_init_error}")
                            raise
            finally:
                # ChromaDB 초기화 완료 후 umask 및 환경 변수 복원
                os.umask(original_umask)
                if original_umask_env is not None:
                    os.environ['UMASK'] = original_umask_env
                elif 'UMASK' in os.environ:
                    del os.environ['UMASK']
            
            self._initialized = True
            print(f"✅ Vector Store reinitialized at {self.vectorstore_path}")
            return True
        except Exception as e:
            error_msg = str(e).lower()
            # 데이터베이스 손상 오류 감지
            if "tenants" in error_msg or "no such table" in error_msg or "database" in error_msg or "already exists" in error_msg:
                print(f"⚠️  Vector Store clear failed (database error): {e}")
                print(f"   Attempting to repair by recreating the database...")
                # 복구 시도
                if self._repair_vectorstore():
                    print(f"✅ Vector Store repaired and reinitialized")
                    return True
                else:
                    return False
            else:
                print(f"⚠️  Failed to clear Vector Store: {e}")
                return False
    
    def add_documents(self, documents: List[Document], check_duplicates: bool = True) -> bool:
        """
        Vector Store에 문서를 동적으로 추가
        
        Args:
            documents: 추가할 Document 리스트
            check_duplicates: 중복 체크 여부 (기본값: True)
            
        Returns:
            성공 여부
        """
        if not self._initialized or not self.vectorstore:
            # Vector Store가 초기화되지 않았을 때 재초기화 시도
            print("⚠️  Vector Store not initialized. Attempting to reinitialize...")
            self._initialize_vectorstore()
            if not self._initialized or not self.vectorstore:
                print("⚠️  Vector Store reinitialization failed. Cannot add documents.")
            return False
        
        try:
            if check_duplicates:
                # 중복 체크: source + sheet + has_draft_context 조합으로 고유 키 생성
                # ChromaDB에서 기존 문서 확인
                documents_to_add = []
                skipped_count = 0
                
                for doc in documents:
                    metadata = doc.metadata
                    source = metadata.get("source", "")
                    sheet = metadata.get("sheet", "")
                    has_draft_context = metadata.get("has_draft_context", False)
                    
                    # 고유 ID 생성: source + sheet + has_draft_context
                    # 같은 source+sheet에 대해 초안 정보 포함 버전은 별도 문서로 취급
                    unique_id = f"{source}::{sheet}::{has_draft_context}"
                    metadata["_unique_id"] = unique_id
                    
                    # 기존 문서 확인: ChromaDB의 get 메서드로 메타데이터 필터링
                    try:
                        # ChromaDB에서 같은 source, sheet, has_draft_context를 가진 문서 검색
                        existing_docs = self.vectorstore.get(
                            where={
                                "source": source,
                                "sheet": sheet,
                                "has_draft_context": has_draft_context
                            }
                        )
                        
                        if existing_docs and len(existing_docs.get("ids", [])) > 0:
                            # 이미 존재하는 문서는 스킵
                            skipped_count += 1
                            continue
                    except Exception as e:
                        # 필터 검색 실패 시 일단 추가 (안전한 방식)
                        # ChromaDB 버전에 따라 get 메서드가 다를 수 있음
                        pass
                    
                    documents_to_add.append(doc)
                
                if documents_to_add:
                    self.vectorstore.add_documents(documents_to_add)
                    if skipped_count > 0:
                        print(f"✅ Added {len(documents_to_add)}/{len(documents)} documents to Vector Store ({skipped_count} duplicates skipped)")
                    else:
                        print(f"✅ Added {len(documents_to_add)} documents to Vector Store")
                else:
                    print(f"⚠️  All {len(documents)} documents are duplicates, skipping...")
            else:
                self.vectorstore.add_documents(documents)
                print(f"✅ Added {len(documents)} documents to Vector Store")
            
            return True
        except Exception as e:
            error_msg = str(e).lower()
            # 데이터베이스 손상 오류 또는 readonly 오류 감지
            if "tenants" in error_msg or "no such table" in error_msg or "database" in error_msg or "readonly" in error_msg:
                print(f"⚠️  Failed to add documents (database corrupted or readonly): {e}")
                print(f"   Attempting to repair by recreating the database...")
                # SQLite 파일 권한 수정 시도
                vectorstore_path_obj = Path(self.vectorstore_path)
                self._fix_sqlite_permissions(vectorstore_path_obj)
                # 복구 시도 후 재시도
                if self._repair_vectorstore():
                    print(f"✅ Vector Store repaired. Retrying to add documents...")
                    try:
                        # 복구 후 재시도
                        if check_duplicates:
                            # 중복 체크 없이 간단하게 추가
                            self.vectorstore.add_documents(documents)
                            print(f"✅ Added {len(documents)} documents to Vector Store (after repair)")
                        else:
                            self.vectorstore.add_documents(documents)
                            print(f"✅ Added {len(documents)} documents to Vector Store (after repair)")
                        return True
                    except Exception as retry_error:
                        print(f"⚠️  Failed to add documents after repair: {retry_error}")
                        return False
                else:
                    print(f"⚠️  Vector Store repair failed. Cannot add documents.")
                    return False
            else:
                print(f"⚠️  Failed to add documents to Vector Store: {e}")
                return False
    
    def search_ddd_patterns(self, query: str, k: int = 10) -> List[Dict]:
        """
        DDD 패턴 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            
        Returns:
            검색 결과 리스트
        """
        if not self._initialized or not self.vectorstore:
            return self._fallback_search_ddd_patterns(query, k)
        
        try:
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter={"type": "ddd_pattern"}
            )
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results
            ]
        except Exception as e:
            print(f"⚠️  DDD pattern search failed: {e}")
            return self._fallback_search_ddd_patterns(query, k)
    
    def search_project_templates(self, query: str, k: int = 5) -> List[Dict]:
        """
        유사 프로젝트 사례 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            
        Returns:
            검색 결과 리스트
        """
        if not self._initialized or not self.vectorstore:
            return self._fallback_search_project_templates(query, k)
        
        try:
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter={"type": "project_template"}
            )
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results
            ]
        except Exception as e:
            print(f"⚠️  Project template search failed: {e}")
            return self._fallback_search_project_templates(query, k)
    
    def search_vocabulary(self, query: str, k: int = 20) -> List[Dict]:
        """
        도메인 용어 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            
        Returns:
            검색 결과 리스트
        """
        if not self._initialized or not self.vectorstore:
            return self._fallback_search_vocabulary(query, k)
        
        try:
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter={"type": "vocabulary"}
            )
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results
            ]
        except Exception as e:
            print(f"⚠️  Vocabulary search failed: {e}")
            return self._fallback_search_vocabulary(query, k)
    
    def search_ui_patterns(self, query: str, k: int = 10) -> List[Dict]:
        """
        UI 패턴 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            
        Returns:
            검색 결과 리스트
        """
        if not self._initialized or not self.vectorstore:
            return self._fallback_search_ui_patterns(query, k)
        
        try:
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter={"type": "ui_pattern"}
            )
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results
            ]
        except Exception as e:
            print(f"⚠️  UI pattern search failed: {e}")
            return self._fallback_search_ui_patterns(query, k)
    
    # Fallback methods (Vector Store가 없을 때 JSON 파일에서 직접 검색)
    
    def _fallback_search_ddd_patterns(self, query: str, k: int) -> List[Dict]:
        """DDD 패턴 Fallback 검색 (JSON 파일에서 직접)"""
        try:
            pattern_files = list(Config.DOMAIN_PATTERNS_PATH.glob("*.json"))
            results = []
            
            for file_path in pattern_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append({
                        "content": json.dumps(data, ensure_ascii=False),
                        "metadata": {
                            "source": str(file_path),
                            "type": "ddd_pattern"
                        }
                    })
            
            return results[:k]
        except Exception as e:
            print(f"⚠️  Fallback DDD search failed: {e}")
            return []
    
    def _fallback_search_project_templates(self, query: str, k: int) -> List[Dict]:
        """프로젝트 템플릿 Fallback 검색"""
        try:
            template_files = list(Config.PROJECT_TEMPLATES_PATH.glob("*.json"))
            results = []
            
            for file_path in template_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append({
                        "content": json.dumps(data, ensure_ascii=False),
                        "metadata": {
                            "source": str(file_path),
                            "type": "project_template"
                        }
                    })
            
            return results[:k]
        except Exception as e:
            print(f"⚠️  Fallback project search failed: {e}")
            return []
    
    def _fallback_search_vocabulary(self, query: str, k: int) -> List[Dict]:
        """용어 Fallback 검색"""
        try:
            vocab_files = list(Config.VOCABULARY_PATH.glob("*.json"))
            results = []
            
            for file_path in vocab_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append({
                        "content": json.dumps(data, ensure_ascii=False),
                        "metadata": {
                            "source": str(file_path),
                            "type": "vocabulary"
                        }
                    })
            
            return results[:k]
        except Exception as e:
            print(f"⚠️  Fallback vocabulary search failed: {e}")
            return []
    
    def _fallback_search_ui_patterns(self, query: str, k: int) -> List[Dict]:
        """UI 패턴 Fallback 검색"""
        try:
            ui_files = list(Config.UI_PATTERNS_PATH.glob("*.json"))
            results = []
            
            for file_path in ui_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append({
                        "content": json.dumps(data, ensure_ascii=False),
                        "metadata": {
                            "source": str(file_path),
                            "type": "ui_pattern"
                        }
                    })
            
            return results[:k]
        except Exception as e:
            print(f"⚠️  Fallback UI search failed: {e}")
            return []
    
    def search_company_standards(self, query: str, k: int = 5, score_threshold: float = DEFAULT_SIM_THRESHOLD) -> List[Dict]:
        """
        회사 표준 검색 (데이터베이스, API, 용어 등 모든 표준)
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            score_threshold: 유사도 점수 임계값 (0.0~1.0, 기본값 0.3)
            
        Returns:
            검색 결과 리스트 (점수 포함)
        """
        if not self._initialized or not self.vectorstore:
            return self._fallback_search_company_standards(query, k)
        
        try:
            # similarity_search_with_score 사용하여 점수 포함
            # 필터 사용 시 오류가 발생할 수 있으므로 try-except로 감싸기
            try:
                results_with_scores = self.vectorstore.similarity_search_with_score(
                    query,
                    k=k * 3,  # 필터링을 위해 더 많이 가져옴
                    filter={"type": {"$in": ["database_standard", "api_standard", "terminology_standard"]}}
                )
            except Exception as filter_error:
                # 필터 오류 시 필터 없이 검색 후 수동 필터링
                # ChromaDB 동시성 문제("Failed to get segments")는 일시적이므로 조용히 처리
                error_msg = str(filter_error).lower()
                if "Failed to get segments" not in error_msg:
                    print(f"⚠️  Search failed with filter: {filter_error}")
                # "no such table: collections" 오류 감지 시 자동 복구 시도
                if "no such table" in error_msg or "collections" in error_msg or "database" in error_msg:
                    print(f"⚠️  Vector Store database corrupted during search. Attempting to repair...")
                    if self._repair_vectorstore():
                        print(f"✅ Vector Store repaired. Retrying search...")
                        # 복구 후 재시도
                        try:
                            all_results = self.vectorstore.similarity_search_with_score(
                                query,
                                k=k * 5  # 더 많이 가져와서 필터링
                            )
                            # 수동 필터링
                            results_with_scores = []
                            for doc, score in all_results:
                                doc_type = doc.metadata.get("type", "")
                                if doc_type in ["database_standard", "api_standard", "terminology_standard"]:
                                    results_with_scores.append((doc, score))
                        except Exception as retry_error:
                            print(f"⚠️  Search still failed after repair: {retry_error}")
                            return self._fallback_search_company_standards(query, k)
                    else:
                        print(f"⚠️  Vector Store repair failed. Using fallback search.")
                        return self._fallback_search_company_standards(query, k)
                else:
                    try:
                        all_results = self.vectorstore.similarity_search_with_score(
                            query,
                            k=k * 5  # 더 많이 가져와서 필터링
                        )
                        # 수동 필터링
                        results_with_scores = []
                        for doc, score in all_results:
                            doc_type = doc.metadata.get("type", "")
                            if doc_type in ["database_standard", "api_standard", "terminology_standard"]:
                                results_with_scores.append((doc, score))
                    except Exception as search_error:
                        error_msg2 = str(search_error).lower()
                        if "no such table" in error_msg2 or "collections" in error_msg2 or "database" in error_msg2:
                            print(f"⚠️  Vector Store database corrupted during search. Attempting to repair...")
                            if self._repair_vectorstore():
                                print(f"✅ Vector Store repaired. Retrying search...")
                                # 복구 후 재시도
                                try:
                                    all_results = self.vectorstore.similarity_search_with_score(
                                        query,
                                        k=k * 5
                                    )
                                    results_with_scores = []
                                    for doc, score in all_results:
                                        doc_type = doc.metadata.get("type", "")
                                        if doc_type in ["database_standard", "api_standard", "terminology_standard"]:
                                            results_with_scores.append((doc, score))
                                except Exception as retry_error2:
                                    print(f"⚠️  Search still failed after repair: {retry_error2}")
                                    return self._fallback_search_company_standards(query, k)
                            else:
                                return self._fallback_search_company_standards(query, k)
                        else:
                            print(f"⚠️  Search failed: {search_error}")
                            return self._fallback_search_company_standards(query, k)
            # 점수 필터링
            # ChromaDB의 similarity_search_with_score는 거리(distance)를 반환
            # ChromaDB는 기본적으로 코사인 거리(cosine distance)를 사용
            # 
            # 거리 범위는 상황에 따라 다를 수 있음:
            # - 정규화된 벡터: 0~1 범위 (distance = 1 - cosine_similarity)
            # - 일반 코사인 거리: 0~2 범위 (distance = 1 - cos(θ), cos(θ) = -1~1)
            # 
            # 실제 거리 값의 범위를 동적으로 감지하여 변환
            filtered_results = []
            all_scores = []  # 디버깅용
            
            # 먼저 모든 거리 값을 수집하여 범위 확인
            distances = [abs(float(score_value)) for _, score_value in results_with_scores]
            if distances:
                dist_min, dist_max = min(distances), max(distances)
                # 거리 범위에 따라 변환 방식 결정
                # 대부분의 거리가 1.0을 넘으면 0~2 범위로 가정, 아니면 0~1 범위로 가정
                if dist_max > 1.0:
                    # 0~2 범위: similarity = 1 - (distance / 2)
                    distance_range = 2.0
                else:
                    # 0~1 범위: similarity = 1 - distance
                    distance_range = 1.0
            else:
                # 기본값: 0~2 범위로 가정 (안전한 선택)
                distance_range = 2.0
            
            for doc, score_value in results_with_scores:
                # 원본 값을 확인
                raw_score = float(score_value)
                distance = abs(raw_score)
                
                # 거리 범위에 따라 유사도 변환
                if distance_range == 2.0:
                    # 0~2 범위: similarity = 1 - (distance / 2)
                    similarity = max(0.0, 1.0 - (distance / 2.0))
                else:
                    # 0~1 범위: similarity = 1 - distance
                    similarity = max(0.0, 1.0 - distance)
                
                all_scores.append((raw_score, distance, similarity))
                
                # 점수 필터링: 유사도가 임계값 이상인 것만 포함
                if similarity >= score_threshold:
                    filtered_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": similarity,
                        "distance": distance,
                        "raw_score": raw_score  # 원본 값도 저장
                    })
                # 상위 k개를 가져오되, 임계값 이상인 것만 포함
                # k개를 채우지 못해도 임계값 이상인 것들은 모두 포함
            
            # 점수 순으로 정렬 (높은 점수부터)
            filtered_results.sort(key=lambda x: x["score"], reverse=True)
            
            # 디버깅: 전체 점수 분포 출력 (처음 10개)
            if all_scores:
                print(f"  [DEBUG] 검색 결과 점수 분포 (처음 10개, 총 {len(all_scores)}개):")
                for i, (raw, dist, sim) in enumerate(all_scores[:10]):
                    print(f"    [{i+1}] 원본값: {raw:.6f}, 거리: {dist:.6f}, 유사도: {sim:.6f}")
                print(f"  [DEBUG] 필터링 후 결과: {len(filtered_results)}/{len(all_scores)}")
                if len(all_scores) > 0:
                    raw_min, raw_max = min(s[0] for s in all_scores), max(s[0] for s in all_scores)
                    dist_min, dist_max = min(s[1] for s in all_scores), max(s[1] for s in all_scores)
                    sim_min, sim_max = min(s[2] for s in all_scores), max(s[2] for s in all_scores)
                    print(f"  [DEBUG] 원본값 범위: {raw_min:.6f} ~ {raw_max:.6f}")
                    print(f"  [DEBUG] 거리 범위: {dist_min:.6f} ~ {dist_max:.6f}")
                    print(f"  [DEBUG] 유사도 범위: {sim_min:.6f} ~ {sim_max:.6f}")
                    
                    # 필터링 전후 비교
                    print(f"  [DEBUG] 필터링 전 결과: {len(results_with_scores)}개")
                    print(f"  [DEBUG] 필터링 후 결과: {len(filtered_results)}개 (임계값: {score_threshold:.3f} 이상)")
                    
                    # 만약 유사도가 모두 0이면 경고 및 거리 범위 분석
                    if sim_max == 0.0:
                        print(f"  [WARNING] ⚠️  모든 유사도가 0입니다! 거리 범위를 확인하세요.")
                        print(f"  [WARNING] 거리 범위: {dist_min:.6f} ~ {dist_max:.6f}")
                        # 실제 변환 로직과 일치하도록 distance_range 사용
                        if distance_range == 2.0:
                            sim_est = max(0.0, 1.0 - (dist_max / 2.0))
                        else:
                            sim_est = max(0.0, 1.0 - dist_max)
                        print(f"  [WARNING] 거리 {dist_max:.3f}는 유사도 {sim_est:.3f}로 변환됩니다.")
                        print(f"  [WARNING] 임계값 {score_threshold:.3f}보다 낮아서 필터링되었습니다.")
            
            # 최종적으로 상위 k개만 반환 (일관성 유지)
            return filtered_results[:k]
        except Exception as e:
            # similarity_search_with_score가 지원되지 않는 경우 기본 검색 사용 (점수 필터링 없음)
            print(f"⚠️  similarity_search_with_score 실패, 기본 검색 사용: {e}")
            try:
                results = self.vectorstore.similarity_search(
                    query,
                    k=k,
                    filter={"type": {"$in": ["database_standard", "api_standard", "terminology_standard"]}}
                )
                return [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": None  # 점수 없음 (필터링 안 함)
                    }
                    for doc in results
                ]
            except Exception as e2:
                # 필터가 지원되지 않는 경우 필터 없이 검색
                try:
                    results = self.vectorstore.similarity_search(query, k=k)
                    return [
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "score": None
                        }
                        for doc in results
                        if doc.metadata.get("type") in ["database_standard", "api_standard", "terminology_standard"]
                    ]
                except Exception as e3:
                    print(f"⚠️  Company standards search failed: {e3}")
                    return self._fallback_search_company_standards(query, k)
    
    def search_api_standards(self, query: str, k: int = 5, score_threshold: float = DEFAULT_SIM_THRESHOLD) -> List[Dict]:
        """
        API 표준 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            score_threshold: 유사도 점수 임계값 (0.0~1.0, 기본값 0.3)
            
        Returns:
            검색 결과 리스트 (점수 포함)
        """
        if not self._initialized or not self.vectorstore:
            return self._fallback_search_api_standards(query, k)
        
        try:
            # similarity_search_with_score 사용하여 점수 포함
            # 필터 사용 시 오류가 발생할 수 있으므로 try-except로 감싸기
            try:
                results_with_scores = self.vectorstore.similarity_search_with_score(
                    query,
                    k=k * 3,
                    filter={"type": "api_standard"}
                )
            except Exception as filter_error:
                # 필터 오류 시 필터 없이 검색 후 수동 필터링
                # ChromaDB 동시성 문제("Failed to get segments")는 일시적이므로 조용히 처리
                error_msg = str(filter_error)
                if "Failed to get segments" not in error_msg:
                    print(f"⚠️  Search failed with filter: {filter_error}")
                all_results = self.vectorstore.similarity_search_with_score(
                    query,
                    k=k * 5  # 더 많이 가져와서 필터링
                )
                # 수동 필터링
                results_with_scores = []
                for doc, score in all_results:
                    doc_type = doc.metadata.get("type", "")
                    if doc_type == "api_standard":
                        results_with_scores.append((doc, score))
            # 점수 필터링 (코사인 거리 기반)
            # 거리 범위를 동적으로 감지하여 변환
            filtered_results = []
            
            # 먼저 모든 거리 값을 수집하여 범위 확인
            distances = [abs(float(score_value)) for _, score_value in results_with_scores]
            if distances:
                dist_max = max(distances)
                distance_range = 2.0 if dist_max > 1.0 else 1.0
            else:
                distance_range = 2.0  # 기본값
            
            for doc, score_value in results_with_scores:
                raw_score = float(score_value)
                distance = abs(raw_score)
                
                # 거리 범위에 따라 유사도 변환
                if distance_range == 2.0:
                    similarity = max(0.0, 1.0 - (distance / 2.0))
                else:
                    similarity = max(0.0, 1.0 - distance)
                
                if similarity >= score_threshold:
                    filtered_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": similarity,
                        "distance": distance,
                        "raw_score": raw_score  # 원본 값도 저장
                    })
            
            # 점수 순으로 정렬 (높은 점수부터)
            filtered_results.sort(key=lambda x: x["score"], reverse=True)
            
            # 최종적으로 상위 k개만 반환 (일관성 유지)
            return filtered_results[:k]
        except Exception as e:
            # fallback: 기본 검색 사용
            try:
                results = self.vectorstore.similarity_search(
                    query,
                    k=k,
                    filter={"type": "api_standard"}
                )
                return [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": None
                    }
                    for doc in results
                ]
            except Exception as e2:
                print(f"⚠️  API standards search failed: {e2}")
                return self._fallback_search_api_standards(query, k)
    
    def search_terminology_standards(self, query: str, k: int = 5, score_threshold: float = DEFAULT_SIM_THRESHOLD) -> List[Dict]:
        """
        용어 표준 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            score_threshold: 유사도 점수 임계값 (0.0~1.0, 기본값 0.3)
            
        Returns:
            검색 결과 리스트 (점수 포함)
        """
        if not self._initialized or not self.vectorstore:
            return self._fallback_search_terminology_standards(query, k)
        
        try:
            # similarity_search_with_score 사용하여 점수 포함
            # 필터 사용 시 오류가 발생할 수 있으므로 try-except로 감싸기
            try:
                results_with_scores = self.vectorstore.similarity_search_with_score(
                    query,
                    k=k * 3,
                    filter={"type": "terminology_standard"}
                )
            except Exception as filter_error:
                # 필터 오류 시 필터 없이 검색 후 수동 필터링
                # ChromaDB 동시성 문제("Failed to get segments")는 일시적이므로 조용히 처리
                error_msg = str(filter_error)
                if "Failed to get segments" not in error_msg:
                    print(f"⚠️  Search failed with filter: {filter_error}")
                all_results = self.vectorstore.similarity_search_with_score(
                    query,
                    k=k * 5  # 더 많이 가져와서 필터링
                )
                # 수동 필터링
                results_with_scores = []
                for doc, score in all_results:
                    doc_type = doc.metadata.get("type", "")
                    if doc_type == "terminology_standard":
                        results_with_scores.append((doc, score))
            # 점수 필터링 (코사인 거리 기반)
            # 거리 범위를 동적으로 감지하여 변환
            filtered_results = []
            
            # 먼저 모든 거리 값을 수집하여 범위 확인
            distances = [abs(float(score_value)) for _, score_value in results_with_scores]
            if distances:
                dist_max = max(distances)
                distance_range = 2.0 if dist_max > 1.0 else 1.0
            else:
                distance_range = 2.0  # 기본값
            
            for doc, score_value in results_with_scores:
                distance = abs(float(score_value))
                
                # 거리 범위에 따라 유사도 변환
                if distance_range == 2.0:
                    similarity = max(0.0, 1.0 - (distance / 2.0))
                else:
                    similarity = max(0.0, 1.0 - distance)
                if similarity >= score_threshold:
                    filtered_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": similarity,
                        "distance": distance,
                        "raw_score": float(score_value)  # 원본 값도 저장
                    })
            
            # 점수 순으로 정렬 (높은 점수부터)
            filtered_results.sort(key=lambda x: x["score"], reverse=True)
            
            # 최종적으로 상위 k개만 반환 (일관성 유지)
            return filtered_results[:k]
        except Exception as e:
            # fallback: 기본 검색 사용
            try:
                results = self.vectorstore.similarity_search(
                    query,
                    k=k,
                    filter={"type": "terminology_standard"}
                )
                return [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": None
                    }
                    for doc in results
                ]
            except Exception as e2:
                print(f"⚠️  Terminology standards search failed: {e2}")
                return self._fallback_search_terminology_standards(query, k)
    
    def _fallback_search_company_standards(self, query: str, k: int) -> List[Dict]:
        """회사 표준 Fallback 검색"""
        try:
            standards_path = Config.COMPANY_STANDARDS_PATH
            if not standards_path.exists():
                return []
            
            results = []
            # 표준 문서 파일 찾기
            for file_path in standards_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in ['.xlsx', '.xls', '.pptx', '.txt', '.md']:
                    # 간단한 텍스트 매칭 (실제로는 Vector Store가 필요)
                    results.append({
                        "content": f"Standard document: {file_path.name}",
                        "metadata": {
                            "source": str(file_path),
                            "type": "database_standard"
                        }
                    })
            
            return results[:k]
        except Exception as e:
            print(f"⚠️  Fallback company standards search failed: {e}")
            return []
    
    def _fallback_search_api_standards(self, query: str, k: int) -> List[Dict]:
        """API 표준 Fallback 검색"""
        return self._fallback_search_company_standards(query, k)
    
    def _fallback_search_terminology_standards(self, query: str, k: int) -> List[Dict]:
        """용어 표준 Fallback 검색"""
        return self._fallback_search_company_standards(query, k)

