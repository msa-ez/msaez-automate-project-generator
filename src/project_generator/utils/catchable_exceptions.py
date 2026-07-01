"""
광범위 예외 처리(`except Exception`) 대체용 구체 예외 타입 집합 (leaf 핸들러 전용).

한정된 연산을 감싸는 try(파일 I/O·파싱·라이브러리 호출 등)에서 실제 발생 가능한
예외를 빠짐없이 포착하도록, 빌트인 + 이 코드베이스가 사용하는 주요 라이브러리의
예외(Exception 직속 상속이라 빌트인으로 안 잡히는 것들)를 한 곳에서 관리한다.

주의: langgraph 의 제어 흐름 예외(GraphInterrupt/GraphBubbleUp 등)는 의도적으로
제외한다(그래프 런타임으로 전파돼야 함). 그래서 이 집합은 그래프를 invoke 하는
오케스트레이션 경계가 아니라, 한정 연산 leaf 핸들러에만 사용한다.
"""
import queue
import xml.etree.ElementTree as _ET

_types = [
    OSError, ValueError, TypeError, LookupError, AttributeError, RuntimeError,
    ImportError, ArithmeticError, AssertionError, StopIteration, StopAsyncIteration,
    BufferError, SyntaxError, RecursionError, UnicodeError, MemoryError, EOFError,
    _ET.ParseError, queue.Empty, queue.Full,
]

# 이 코드베이스가 사용하는 라이브러리의 Exception-직속 예외 (설치돼 있으면 포함)
for _mod, _name in (
    ("convert_case.exceptions", "MixedCaseError"),
    ("langchain_core.exceptions", "LangChainException"),
    ("openai", "OpenAIError"),
    ("httpx", "HTTPError"),
    ("pydantic", "ValidationError"),
    ("firebase_admin.exceptions", "FirebaseError"),
    ("kubernetes.client.exceptions", "ApiException"),
    ("kubernetes.config.config_exception", "ConfigException"),
    ("a2a.utils.errors", "A2AServerError"),
    ("starlette.exceptions", "HTTPException"),
    ("fastapi.exceptions", "ValidationException"),
):
    try:
        _m = __import__(_mod, fromlist=[_name])
        _t = getattr(_m, _name)
        if isinstance(_t, type) and issubclass(_t, BaseException):
            _types.append(_t)
    except (ImportError, AttributeError):
        pass

CATCHABLE_EXCEPTIONS = tuple(_types)
