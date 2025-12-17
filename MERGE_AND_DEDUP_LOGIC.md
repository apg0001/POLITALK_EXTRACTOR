# 행합치기 및 중복제거 로직 정리

## 1. 행합치기 (Merge) 로직

### 1.1 전체 흐름 (`data_merger.py`)

**목적**: 같은 기사, 같은 발언자, 같은 날짜, 같은 신문사의 연속된 발언을 병합

**병합 조건**:
1. **메타데이터 일치**: 기사 제목, 발언자, 날짜, 신문사가 모두 동일
2. **문맥적 연결**: 이전 발언과 문맥적으로 연결되어 병합 가능한 경우 (`Merger.check_cases`)

**병합 과정**:
- 큰따옴표 발언: 중복 제거 후 `"  "` (공백 2개)로 연결
- 문단: 중복 문장 제거 후 병합
- 문장: `"  "`로 연결

### 1.2 병합 가능 여부 판단 (`text_manager.py` - `Merger.check_cases`)

**판단 기준**:
1. **구조적 케이스** (5가지 중 하나를 만족해야 함)
2. **문맥적 연결** (`case_base`)
3. **예외 접속사 아님** (`is_exceptional_conjunction`)
4. **또는 같은 문장 내 포함** (`case_same_sentence`)

#### 구조적 케이스 5가지:

**Case 1: 접속사만 + 단일 동사**
- 예: `"이어 말했다"`
- 조건: 큰따옴표 앞 = 순접 접속사만, 뒤 = 단일 동사(2단어 이하)

**Case 2: 접속사 + 주어(은/는) + 단일 동사**
- 예: `"이어서 그는 말했다"`
- 조건: 앞 = 접속사 + 주어(은/는), 뒤 = 단일 동사

**Case 3: 대명사 주어(그는/그녀는) + 단일 동사**
- 예: `"그는 말했다"`
- 조건: 앞 = "그는" 또는 "그녀는"만, 뒤 = 단일 동사

**Case 4: 주어(은/는)만 + 단일 동사**
- 예: `"홍길동 의원은 말했다"`
- 조건: 앞 = 주어(은/는, 4단어 이하), 뒤 = 단일 동사

**Case 5: 공백 + 단일 동사**
- 예: `""말했다"`
- 조건: 앞 = 공백, 뒤 = 단일 동사

#### 문맥적 연결 확인 (`case_base`):
- 현재 문장의 앞 문장이 이전 발언문과 관련이 있는지 확인
- 큰따옴표가 닫히지 않은 경우 이전 문장과 합쳐서 확인

#### 같은 문장 내 포함 확인 (`case_same_sentence`):
- 현재 문장 자체에 이전 발언문이 포함되어 있으면 병합 가능

---

## 2. 중복제거 (Deduplication) 로직

### 2.1 발언문 중복 제거 (`duplicate_remover.py`)

**목적**: 유사도 검사를 통해 중복된 발언문을 제거

**비교 방식**:
- 각 엔트리의 발언문을 이전 엔트리들과 비교
- 텍스트 정규화 후 비교 (공백, 특수문자 제거)

**중복 판단 기준**:

1. **포함 관계**:
   - 현재 문장이 이전 문장을 완전히 포함 → 이전 문장 제거
   - 이전 문장이 현재 문장을 완전히 포함 → 현재 문장 제거

2. **유사도 검사 (min 기준)**:
   - 유사도가 높으면 → 이전 문장 제거

3. **완전 일치 또는 유사도 검사 (max 기준)**:
   - 완전히 동일하거나 유사도가 매우 높으면
   - 더 짧은 쪽을 제거 (같은 길이면 현재 문장 제거)

**처리 과정**:
1. 현재 엔트리의 발언문을 문장 단위로 분리
2. 각 문장을 정규화 (공백, 특수문자 제거)
3. 이전 엔트리들의 정규화된 발언문과 비교
4. 중복 발견 시 해당 문장 제거
5. 모든 문장이 제거되면 엔트리 자체 삭제
6. 남은 문장이 있으면 결과에 추가

### 2.2 텍스트 내 중복 구간 제거 (`extract_topic_summary.py` - `RedundancyRemover`)

**목적**: 텍스트 내에서 반복되는 구간을 찾아 제거

**작동 방식**:
1. 텍스트를 토큰화 및 원형화(lemmatize)
2. 각 원형(lemma)이 나타나는 위치를 인덱스로 저장
3. 같은 원형이 2번 이상 나타나는 경우, 중복 구간 찾기
4. 두 위치에서 시작하는 구간이 얼마나 일치하는지 확인
5. 최소 길이(`min_common_len`, 기본값 3) 이상이면 제거

**예시**:
- 입력: `"그는 말했다. 그는 말했다."`
- 처리: 토큰화 → 원형화 → 중복 구간 발견 → 제거
- 출력: `"그는 말했다."`

---

## 3. 주요 개선 사항

### 3.1 변수명 개선
- `part_a`, `part_c` → `part_before_quote`, `part_after_quote`
- `prev` → `previous_quoted_speeches`
- `merged_data` → `merged_result`
- `duplicate_removed_data` → `deduplicated_result`
- `sentence_sets` → `previous_entries_cache`

### 3.2 중첩 반복문 개선
- 복잡한 중첩 루프에 명확한 주석 추가
- 각 루프의 목적과 종료 조건 명시
- 변수명을 통해 루프의 역할 명확화

### 3.3 주석 추가
- 각 함수에 상세한 docstring 추가
- 주요 로직 단계별 주석 추가
- 조건문의 의미를 명확히 설명

### 3.4 코드 구조 개선
- 접속사 리스트를 클래스 상수로 분리 (`SEQUENTIAL_CONJUNCTIONS`)
- 조건문을 명확한 변수명으로 분리
- 반환값의 의미를 명확히 표현

---

## 4. 사용 예시

### 행합치기
```python
from data_merger import DataMerger

merger = DataMerger()
merged_data = merger.merge_data(data, progress_tracker)
```

### 중복제거
```python
from duplicate_remover import DuplicateRemover

remover = DuplicateRemover()
deduplicated_data = remover.remove_duplicates(data, progress_tracker)
```

### 텍스트 내 중복 구간 제거
```python
from extract_topic_summary import RedundancyRemover

remover = RedundancyRemover(min_common_len=3)
cleaned_text = remover.trim_redundant_block(text)
```

