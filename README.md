# Project Generator Backend (LangGraph)

MSA-EZ Event Storming 자동화를 위한 LangGraph 기반 UserStory 생성 백엔드 서버

## 📋 개요

이 프로젝트는 Firebase 또는 AceBase Job Queue를 통해 프론트엔드와 통신하며, LangGraph 워크플로우를 사용하여 요구사항으로부터 User Story, Actor, Business Rule을 추출합니다.

**지원하는 배포 환경:**
- 🔥 **Firebase**: 클라우드 배포 (Kubernetes)
- 🏠 **AceBase**: 설치형(온프레미스) 환경 (로컬 실행)

### 주요 기능

- ✅ **UserStory Generator**: RAG 기반 User Story 자동 생성
- ✅ **Bounded Context Generator**: 요구사항 기반 Bounded Context 자동 생성
- ✅ **Aggregate Draft Generator**: Aggregate 초안 자동 생성
- ✅ **Standard Transformer**: 회사 표준 문서 기반 자동 변환 (RAG)
- ✅ **Traceability Generator**: 도메인 객체와 요구사항 간 추적성(refs) 생성
- ✅ **Preview Fields Generator**: Aggregate 필드 자동 생성
- ✅ **DDL Fields Generator**: DDL 기반 필드 매핑
- ✅ **Requirements Mapper**: 요구사항 매핑
- ✅ **Requirements Validator**: 요구사항 검증
- 🔥 **Storage Integration**: Firebase 또는 AceBase Job Queue 방식의 비동기 처리
- 🚀 **Auto Scaling**: Kubernetes 환경에서 자동 스케일링
- 📊 **Health Check**: `/ok` 엔드포인트 제공

## 🏗️ 아키텍처

### 클라우드 배포 (Firebase)
```
Frontend (Vue.js)
    ↓ (Firebase)
    ↓ jobs/user_story_generator/{jobId}
    ↓
Backend (Python/LangGraph) - Kubernetes Pod
    ├── DecentralizedJobManager (Job 감시)
    ├── UserStoryWorkflow (LangGraph)
    │   ├── RAG Retriever
    │   ├── LLM (GPT-4o)
    │   └── Output Parser
    └── Firebase System (결과 저장)
```

### 설치형 환경 (AceBase)
```
Frontend (Vue.js)
    ↓ (AceBase - localhost:5757)
    ↓ requestedJobs/user_story_generator/{jobId}
    ↓
Backend (Python/LangGraph) - 로컬 실행
    ├── DecentralizedJobManager (Job 감시)
    ├── UserStoryWorkflow (LangGraph)
    │   ├── RAG Retriever
    │   ├── LLM (GPT-4o)
    │   └── Output Parser
    └── AceBase System (결과 저장)
```

## 🛠️ 설치 및 실행

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치 (pyproject.toml 기반)
pip install -e .
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 다음 내용을 추가:

**기본 설정 (공통):**
```bash
# Storage Type (firebase 또는 acebase)
STORAGE_TYPE=acebase  # 또는 firebase

# OpenAI (표준 SDK 환경변수 그대로 사용)
OPENAI_API_KEY=sk-...         # 게이트웨이 사용 시에는 게이트웨이 발급 키로 교체
# OPENAI_BASE_URL=http://taigpt.posco.net/gpgpta01-gpt/v1   # P-GPT 게이트웨이 (선택)
# OPENAI_EMBEDDING_API_KEY=sk-...                           # 게이트웨이 사용 시 임베딩용 실제 OpenAI 키 (필수)

# Server
FLASK_PORT=2025  # Flask 서버 포트 (기본값: 2025, langgraph dev와 충돌 방지)
IS_LOCAL_RUN=true
NAMESPACE=project_generator
```

**AceBase 사용 시 추가:**
```bash
# AceBase 설정
ACEBASE_HOST=127.0.0.1
ACEBASE_PORT=5757
ACEBASE_DB_NAME=mydb
ACEBASE_HTTPS=false
# 인증은 선택적 (설정하지 않으면 인증 없이 진행)
ACEBASE_USERNAME=admin  # 선택적
ACEBASE_PASSWORD=75sdDSFg37w5  # 선택적 (AceBase의 ADMIN_PASSWORD 환경변수와 일치해야 함, 프로덕션 환경에서는 반드시 변경 권장)
```

**Firebase 사용 시 추가:**
```bash
# Firebase 설정
FIREBASE_DATABASE_URL=https://YOUR-PROJECT.firebaseio.com
FIREBASE_SERVICE_ACCOUNT_PATH=./firebase-credentials.json
FIREBASE_STORAGE_BUCKET=YOUR-BUCKET.appspot.com
```

**참고**: `.env` 파일에 모든 환경 변수를 넣어두고 `STORAGE_TYPE`만 변경해도 됩니다. 사용하지 않는 설정은 무시됩니다.

### 3. 인증 설정

- **AceBase**: 인증은 선택적 (환경 변수 `ACEBASE_USERNAME`, `ACEBASE_PASSWORD`가 설정되지 않으면 인증 없이 진행)
- **Firebase**: Firebase 콘솔에서 서비스 계정 키를 다운로드하여 `firebase-credentials.json`으로 저장

### 4. 서버 실행

```bash
# 간편 실행 (권장)
./start.sh

# 또는 수동 실행
source venv/bin/activate
export PYTHONPATH="$(pwd)/src"
python -m project_generator.main
```

서버가 `http://localhost:2025`에서 실행됩니다. (기본 포트, `FLASK_PORT` 환경 변수로 변경 가능)

**Health Check:**
```bash
curl http://localhost:2025/ok
```

**표준 문서 API (AceBase 환경용):**
```bash
# 파일 목록 조회
curl "http://localhost:2025/api/standard-documents/list?userId=YOUR_USER_ID"

# 파일 업로드
curl -X POST -F "files=@document.xlsx" -F "userId=YOUR_USER_ID" \
  http://localhost:2025/api/standard-documents/upload

# 파일 삭제
curl -X DELETE "http://localhost:2025/api/standard-documents/delete?userId=YOUR_USER_ID&filename=document.xlsx"
```

## 📦 배포 방법

### 설치형(온프레미스) 환경
- 로컬에서 직접 실행 (이 README 참고)
- Kubernetes 배포 불필요

### 클라우드 배포
- `README-DEPLOYMENT.md` 참고
- Firebase 또는 AceBase 전용 배포 가능

## 📁 프로젝트 구조

```
backend-generators/
└── src/project_generator/
    ├── main.py                 # 메인 서버 (Flask + Job Manager)
    ├── config.py               # 설정
    ├── workflows/
    │   ├── user_story/
    │   │   └── user_story_generator.py  # UserStory LangGraph 워크플로우
    │   ├── summarizer/
    │   │   └── requirements_summarizer.py  # 요구사항 요약
    │   ├── bounded_context/
    │   │   └── bounded_context_generator.py  # Bounded Context 생성
    │   ├── sitemap/
    │   │   ├── command_readmodel_extractor.py  # Command/ReadModel 추출
    │   │   └── sitemap_generator.py  # SiteMap 생성
    │   ├── aggregate_draft/
    │   │   ├── aggregate_draft_generator.py  # Aggregate 초안 생성
    │   │   ├── standard_transformer.py  # 표준 변환 (RAG 기반)
    │   │   ├── traceability_generator.py  # 추적성(refs) 생성
    │   │   ├── preview_fields_generator.py  # Preview Fields 생성
    │   │   ├── ddl_fields_generator.py  # DDL Fields 매핑
    │   │   ├── ddl_extractor.py  # DDL 필드 추출
    │   │   └── requirements_mapper.py  # 요구사항 매핑
    │   ├── requirements_validation/
    │   │   └── requirements_validator.py  # 요구사항 검증
    │   └── common/
    │       ├── rag_retriever.py  # RAG Knowledge Base
    │       ├── standard_loader.py  # 표준 문서 로더
    │       ├── standard_indexer.py  # 표준 문서 인덱서
    │       └── standard_rag_service.py  # 표준 RAG 서비스
    ├── systems/
    │   ├── firebase_system.py  # Firebase 연동
    │   ├── acebase_system.py  # AceBase 연동
    │   └── storage_system_factory.py  # Storage 시스템 팩토리 (Strategy 패턴)
    ├── utils/
    │   ├── decentralized_job_manager.py  # Job Queue 관리
    │   ├── job_util.py          # Job 유틸리티
    │   ├── logging_util.py      # 로깅
    │   ├── json_util.py         # JSON 처리
    │   └── convert_case_util.py # CamelCase 변환
    └── models/
        └── (Legacy compatibility models)
├── .env                        # 환경 변수
├── .gitignore                  # Git 제외 파일
├── pyproject.toml              # Python 프로젝트 설정
└── knowledge_base/             # 표준 문서 및 Vector Store
    ├── company_standards/       # 표준 문서 디렉토리
    └── vectorstore/             # ChromaDB 저장소
```

## 🔧 주요 컴포넌트

### 지원하는 워크플로우

백엔드는 다음 워크플로우를 지원합니다:

1. **UserStory Generator** (`usgen-*`): RAG 기반 User Story 자동 생성
2. **Requirements Summarizer** (`summ-*`): 요구사항 요약
3. **Bounded Context Generator** (`bcgen-*`): Bounded Context 자동 생성
4. **Command/ReadModel Extractor** (`cmrext-*`): Command와 ReadModel 추출
5. **SiteMap Generator** (`smapgen-*`): SiteMap 생성
6. **Requirements Mapper** (`reqmap-*`): 요구사항 매핑
7. **Aggregate Draft Generator** (`aggr-draft-*`): Aggregate 초안 생성
8. **Preview Fields Generator** (`preview-fields-*`): Preview Fields 생성
9. **DDL Fields Generator** (`ddl-fields-*`): DDL Fields 매핑
10. **Traceability Generator** (`trace-add-*`): 추적성(refs) 생성
11. **Standard Transformer** (`std-trans-*`): 표준 변환 (RAG 기반)
12. **DDL Extractor** (`ddl-extract-*`): DDL 필드 추출
13. **Requirements Validator** (`req-valid-*`): 요구사항 검증

### UserStoryWorkflow 예시

```python
from project_generator.workflows.user_story.user_story_generator import UserStoryWorkflow

workflow = UserStoryWorkflow()
result = workflow.run({
    "jobId": "usgen-...",
    "requirements": "사용자는 ...",
    "bounded_contexts": []
})

# result:
# {
#     "userStories": [...],
#     "actors": [...],
#     "businessRules": [...],
#     "boundedContexts": [...],
#     "isCompleted": True
# }
```

### Firebase Job Queue

**Job 생성 (Frontend)**:
```javascript
// jobs/user_story_generator/{jobId}
await storage.setObject(`jobs/user_story_generator/${jobId}`, {
    state: {
        inputs: {
            jobId,
            requirements,
            bounded_contexts: []
        }
    }
});

// requestedJobs/user_story_generator/{jobId}
await storage.setObject(`requestedJobs/user_story_generator/${jobId}`, {
    createdAt: firebase.database.ServerValue.TIMESTAMP
});
```

**Job 처리 (Backend)**:
```python
# 1. DecentralizedJobManager가 requestedJobs 감시
# 2. 새로운 Job 발견 시 claim
# 3. process_user_story_job() 호출
# 4. UserStoryWorkflow 실행
# 5. 결과를 jobs/.../state/outputs에 저장
# 6. requestedJobs에서 삭제
```

## 🚀 배포

### Docker

```bash
# 이미지 빌드
docker build -t user-story-generator:latest .

# 실행
docker run -p 2025:2025 \
  -e FIREBASE_DATABASE_URL=... \
  -e OPENAI_API_KEY=... \
  user-story-generator:latest
```

### Kubernetes

```bash
# 배포
kubectl apply -f k8s/deployment.yaml

# 스케일링
kubectl scale deployment user-story-generator --replicas=3
```

## 📊 모니터링

### Health Check

```bash
curl http://localhost:2025/ok
```

### 표준 문서 API (AceBase 환경)

AceBase 로컬 환경에서 표준 문서(Excel, PowerPoint)를 관리하는 API:

- **목록 조회**: `GET /api/standard-documents/list?userId={userId}`
- **파일 업로드**: `POST /api/standard-documents/upload` (multipart/form-data)
- **파일 삭제**: `DELETE /api/standard-documents/delete?userId={userId}&filename={filename}`

파일은 `knowledge_base/company_standards/{userId}/` 경로에 저장됩니다.

### 로그

```bash
# 로그 확인
tail -f server.log

# 또는 Docker
docker logs -f <container-id>
```

## 🔍 트러블슈팅

### 1. Firebase 연결 오류

```bash
# Firebase 인증 확인
cat firebase-credentials.json

# 환경 변수 확인
echo $FIREBASE_DATABASE_URL
```

### 2. Job이 처리되지 않음

```bash
# Backend 로그 확인
grep "Job 시작" server.log

# Firebase에서 직접 확인
# https://console.firebase.google.com/
```

### 3. LLM API 오류

```bash
# OpenAI API 키 확인
echo $OPENAI_API_KEY

# Rate Limit 확인
# 로그에서 "rate_limit_exceeded" 검색
```

## 📝 개발 가이드

### 새로운 워크플로우 추가

1. `workflows/` 폴더에 새 디렉토리 생성
2. LangGraph `StateGraph` 정의
3. `main.py`에 `process_*_job` 함수 추가
4. `main.py`의 `process_job_async`에 Job ID prefix 라우팅 추가
5. `decentralized_job_manager.py`의 `monitored_namespaces`에 네임스페이스 추가
6. `job_util.py`의 `_get_namespace_from_job_id`에 Job ID prefix 패턴 추가

### 테스트

```bash
# 단위 테스트
pytest tests/

# 통합 테스트
python -m src.eventstorming_generator.runs.run_user_story_generator
```

## 🤝 기여

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이센스

This project is licensed under the MIT License.

## 🔗 관련 문서

- [LangGraph 아키텍처 문서](../LANGGRAPH_ARCHITECTURE.md)
- [Frontend Integration Guide](../docs/frontend-integration.md)
- [Firebase Setup Guide](../docs/firebase-setup.md)

## 📧 문의

프로젝트에 대한 문의사항은 이슈로 등록해주세요.
