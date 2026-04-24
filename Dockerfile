FROM python:3.12-slim

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# non-root 사용자 생성 (Kubernetes securityContext와 일치)
RUN groupadd -r -g 1000 appuser && \
    useradd -r -u 1000 -g appuser -m -d /home/appuser appuser

# Python 의존성 파일 복사
COPY pyproject.toml ./
COPY uv.lock ./

# 애플리케이션 코드 복사 (uv sync 이전에 필요)
COPY src/ ./src/

# /app 디렉토리 소유권을 non-root 사용자로 변경
RUN chown -R appuser:appuser /app

# 시간 맞추기 (USER 전환 전에 root 권한으로 실행)
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# non-root 사용자로 전환
USER appuser

# uv 설치 및 의존성 설치 (빌드 시점에만 네트워크 사용)
RUN pip install --user uv
ENV PATH="/home/appuser/.local/bin:$PATH"
RUN uv sync

# Python 경로 설정
ENV PYTHONPATH=/app/src

# 포트 노출 (헬스체크용)
EXPOSE 2024

# 런타임에는 uv를 거치지 않고 .venv의 python을 직접 호출한다.
# (uv run은 기본적으로 pyproject.toml을 보고 프로젝트 wheel을 재빌드하려 하므로,
#  pypi가 차단된 폐쇄망에서 "failed to build ... @ file:///app" 오류가 발생한다.)
CMD ["/app/.venv/bin/python", "-m", "project_generator.main"]