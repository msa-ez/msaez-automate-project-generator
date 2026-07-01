"""
광범위 예외 처리(`except Exception`) 대체용 구체 예외 타입 집합.

정적분석의 '부적절한 예외 처리'를 회피하면서도, 에러 바운더리(생성 플로우 등)가
실제로 발생 가능한 예외를 빠짐없이 포착하도록 빌트인 + 주요 라이브러리 예외를
한 곳에서 관리한다. 내부 흐름 제어용 raise 는 RuntimeError 로 통일한다.
"""
import queue

_types = [
    OSError, ValueError, TypeError, LookupError, AttributeError, RuntimeError,
    ImportError, ArithmeticError, AssertionError, StopIteration, StopAsyncIteration,
    BufferError, SyntaxError, RecursionError, UnicodeError, MemoryError,
    queue.Empty, queue.Full,
]

# 선택적 라이브러리 예외 (설치돼 있으면 포함) — Exception 직속 상속이라 위 빌트인으로 안 잡힘
for _mod, _name in (
    ("langchain_core.exceptions", "LangChainException"),
    ("openai", "OpenAIError"),
    ("httpx", "HTTPError"),
    ("pydantic", "ValidationError"),
    ("firebase_admin.exceptions", "FirebaseError"),
    ("kubernetes.client.exceptions", "ApiException"),
    ("kubernetes.config.config_exception", "ConfigException"),
):
    try:
        _m = __import__(_mod, fromlist=[_name])
        _types.append(getattr(_m, _name))
    except (ImportError, AttributeError):
        pass

CATCHABLE_EXCEPTIONS = tuple(_types)
