"""중복 제거를 담당하는 클래스"""

import traceback
from text_manager import nlp


class DuplicateRemover:
    """중복 제거를 담당하는 클래스"""
    
    def __init__(self):
        """DuplicateRemover 초기화"""
        # 정규화 결과 캐시 (같은 문장에 대한 반복 분석 방지)
        self._normalization_cache = {}
    
    def _normalize_quote(self, quote, use_cache=True):
        """큰따옴표 발언 정규화 (형태소 분석을 통한 원형 변환)
        
        stanza를 사용하여 형태소 분석을 수행하고, 각 단어를 원형(lemma)으로 변환합니다.
        이를 통해 "말했다", "말했다고", "말하는" 등이 모두 "말하다"로 통일되어
        유사도 계산이 더 정확해집니다.
        
        캐싱을 통해 같은 문장에 대한 반복 분석을 방지합니다.
        
        Args:
            quote (str): 정규화할 큰따옴표 발언
            use_cache (bool): 캐시 사용 여부 (기본값 True)
            
        Returns:
            str: 정규화된 발언 (원형으로 변환됨)
        """
        if not quote or not quote.strip():
            return ""
        
        quote_stripped = quote.strip()
        
        # 캐시 확인
        if use_cache and quote_stripped in self._normalization_cache:
            return self._normalization_cache[quote_stripped]
        
        try:
            # stanza를 사용하여 형태소 분석
            doc = nlp(quote_stripped)
            
            # 각 단어의 원형(lemma) 추출
            lemmas = []
            for sent in doc.sentences:
                for word in sent.words:
                    # lemma는 "원형+접사" 형태일 수 있으므로 '+'로 분리하여 첫 번째만 사용
                    lemma = word.lemma.split('+')[0] if word.lemma else word.text
                    if lemma:  # 빈 문자열이 아닌 경우만 추가
                        lemmas.append(lemma)
            
            # 원형들을 공백으로 결합
            normalized = ' '.join(lemmas)
            
            # 캐시에 저장
            if use_cache:
                self._normalization_cache[quote_stripped] = normalized
            
            return normalized
            
        except Exception as e:
            # 형태소 분석 실패 시 원본 반환 (방어적 처리)
            # traceback.print_exc()  # 디버깅용 (필요시 주석 해제)
            result = quote_stripped
            if use_cache:
                self._normalization_cache[quote_stripped] = result
            return result
    
    def _calculate_similarity(self, quote1, quote2, norm1=None, norm2=None):
        """두 큰따옴표 발언의 유사도 계산
        
        Args:
            quote1 (str): 첫 번째 발언 (norm1이 제공되면 사용하지 않음)
            quote2 (str): 두 번째 발언 (norm2가 제공되면 사용하지 않음)
            norm1 (str, optional): 이미 정규화된 첫 번째 발언
            norm2 (str, optional): 이미 정규화된 두 번째 발언
            
        Returns:
            float: 유사도 (0.0 ~ 1.0)
        """
        # 정규화된 문장이 제공되지 않으면 정규화 수행
        if norm1 is None:
            norm1 = self._normalize_quote(quote1)
        if norm2 is None:
            norm2 = self._normalize_quote(quote2)
        
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if not words1 or not words2:
            return 0.0
        
        common_words = words1 & words2
        similarity = len(common_words) / max(len(words1), len(words2))
        
        return similarity
    
    def _is_subset(self, quote_a, quote_b, norm_a=None, norm_b=None, threshold=0.7):
        """룰 3: A가 B의 부분집합인지 확인 (A ⊂ B)
        
        부분 포함 체크: A의 단어 중 70% 이상이 B에 포함되면서 A != B인 경우
        예: C = "안녕하세요", C' = "안녕하세요 반갑습니다" → C ⊂ C'
        
        Args:
            quote_a (str): 첫 번째 발언 (짧은 쪽)
            quote_b (str): 두 번째 발언 (긴 쪽)
            norm_a (str, optional): 이미 정규화된 첫 번째 발언
            norm_b (str, optional): 이미 정규화된 두 번째 발언
            threshold (float): 일치 비율 임계값 (기본값 0.7, 70%)
            
        Returns:
            bool: quote_a가 quote_b의 부분집합이면 True
        """
        # 정규화된 문장이 제공되지 않으면 정규화 수행
        if norm_a is None:
            norm_a = self._normalize_quote(quote_a)
        if norm_b is None:
            norm_b = self._normalize_quote(quote_b)
        
        # A가 B에 완전히 포함되면서 A != B인 경우 (완전 일치 케이스)
        if norm_a in norm_b and norm_a != norm_b:
            return True
        
        # 단어 단위 부분 집합 체크 (70% 이상 일치)
        words_a = set(norm_a.split())
        words_b = set(norm_b.split())
        
        if not words_a or not words_b:
            return False
        
        # A의 단어 중 B에 포함된 단어의 비율 계산
        common_words = words_a & words_b
        match_ratio = len(common_words) / len(words_a) if len(words_a) > 0 else 0
        
        # A의 70% 이상 단어가 B에 포함되고, A의 단어 수가 B보다 적으면 부분집합
        return match_ratio >= threshold and len(words_a) < len(words_b)
    
    def _are_similar(self, quote1, quote2, threshold=0.625, norm1=None, norm2=None):
        """룰 1: 두 발언이 임계값 이상 유사한지 확인

        Args:
            quote1 (str): 첫 번째 발언 (norm1이 제공되면 사용하지 않음)
            quote2 (str): 두 번째 발언 (norm2가 제공되면 사용하지 않음)
            threshold (float): 유사도 임계값 (기본값 0.625)
            norm1 (str, optional): 이미 정규화된 첫 번째 발언
            norm2 (str, optional): 이미 정규화된 두 번째 발언

        Returns:
            bool: 유사도가 임계값 이상이면 True
        """
        similarity = self._calculate_similarity(quote1, quote2, norm1=norm1, norm2=norm2)
        return similarity >= threshold

    def _is_duplicate(self, norm_a, norm_b, quote_a=None, quote_b=None):
        """두 발언이 중복인지 확인 (부분집합 관계 또는 유사도 임계값 이상).
        기존 임계값 유지.
        """
        if self._is_subset(quote_a or "", quote_b or "", norm_a=norm_a, norm_b=norm_b):
            return True
        if self._is_subset(quote_b or "", quote_a or "", norm_a=norm_b, norm_b=norm_a):
            return True
        if self._are_similar(
            quote_a or "", quote_b or "", norm1=norm_a, norm2=norm_b
        ):
            return True
        return False

    def remove_duplicates(self, data, progress_tracker):
        """중복 발언 제거 (배치 단위 3단계 룰)

        1. 큰따옴표 1개인 행: 자기 자신을 제외한 다른 모든 행과 중복 검토 → 중복이 있으면 해당 행 전체 제거
        2. 큰따옴표 2개인 행: 다른 모든 행과 비교하여 중복 문장이 있으면 자기 행에서 해당 문장만 제거
        3. 큰따옴표 3개 이상인 행: 같은 3개 이상 행들끼리만 비교, 중복 시 자기 행에서 해당 문장 제거 (문장 수 비교 없음)
        기존 중복 임계값(유사도/부분집합) 유지.
        """
        if not data:
            print("[중복 제거] 저장할 데이터가 없습니다.")
            return []

        import time
        start_time = time.time()
        total_entries = len(data)
        progress_tracker.progress_bar['maximum'] = total_entries
        progress_tracker.initialize_tqdm(total_entries, "[4단계 중 3단계] 중복 제거 중")

        try:
            # 전처리: 각 행을 발언 리스트·정규화 리스트로 파싱하고, 원본 순서 인덱스와 함께 그룹 분류
            parsed = []
            for idx, entry in enumerate(data):
                quotes = [q.strip() for q in entry.get("발언", "").split("  ") if q.strip()]
                if not quotes:
                    parsed.append((idx, entry, None, None))
                    continue
                norm = [self._normalize_quote(q) for q in quotes]
                parsed.append((idx, entry, quotes, norm))

            # 그룹 분류: 1개 / 2개 / 3개 이상
            group_1 = [(i, e, q, n) for i, e, q, n in parsed if q and len(q) == 1]
            group_2 = [(i, e, q, n) for i, e, q, n in parsed if q and len(q) == 2]
            group_3plus = [(i, e, q, n) for i, e, q, n in parsed if q and len(q) >= 3]

            # 비교용: "다른 모든 행"의 (원문, 정규화) 쌍 리스트 (인덱스 제외용은 나중에 반복에서 처리)
            def all_quotes_except(entries, exclude_idx):
                out = []
                for i, _e, q, n in entries:
                    if i == exclude_idx or not q:
                        continue
                    for qi, (orig, norm_val) in enumerate(zip(q, n)):
                        out.append((orig, norm_val))
                return out

            kept = []  # (original_index, entry_dict)

            # 1단계: 큰따옴표 1개인 행 — 자기 제외 다른 모든 행과 중복이면 자기 전체 제거
            for pos, (idx, entry, quotes, norms) in enumerate(group_1):
                progress_tracker.update_progress(
                    pos + 1, total_entries,
                    "[4단계 중 3단계] 중복 제거 중 (1개 문장 행)",
                    start_time
                )
                my_q, my_n = quotes[0], norms[0]
                others = all_quotes_except(group_1 + group_2 + group_3plus, exclude_idx=idx)
                is_dup = False
                for other_q, other_n in others:
                    if self._is_duplicate(my_n, other_n, quote_a=my_q, quote_b=other_q):
                        is_dup = True
                        break
                if not is_dup:
                    kept.append((idx, entry))

            # 2단계: 큰따옴표 2개인 행 — 다른 모든 행과 비교, 중복 문장은 자기 자신에서 제거
            for pos, (idx, entry, quotes, norms) in enumerate(group_2):
                progress_tracker.update_progress(
                    len(group_1) + pos + 1, total_entries,
                    "[4단계 중 3단계] 중복 제거 중 (2개 문장 행)",
                    start_time
                )
                others = all_quotes_except(group_1 + group_2 + group_3plus, exclude_idx=idx)
                new_quotes = []
                for my_q, my_n in zip(quotes, norms):
                    is_dup = False
                    for other_q, other_n in others:
                        if self._is_duplicate(my_n, other_n, quote_a=my_q, quote_b=other_q):
                            is_dup = True
                            break
                    if not is_dup:
                        new_quotes.append(my_q)
                if new_quotes:
                    new_entry = dict(entry)
                    new_entry["발언"] = "  ".join(new_quotes)
                    kept.append((idx, new_entry))

            # 3단계: 큰따옴표 3개 이상 행들끼리만 비교, 중복 시 자기 자신에서 해당 문장 제거
            only_3plus_quotes = []
            for i, _e, q, n in group_3plus:
                if not q:
                    continue
                for orig, norm_val in zip(q, n):
                    only_3plus_quotes.append((i, orig, norm_val))

            for pos, (idx, entry, quotes, norms) in enumerate(group_3plus):
                progress_tracker.update_progress(
                    len(group_1) + len(group_2) + pos + 1, total_entries,
                    "[4단계 중 3단계] 중복 제거 중 (3개 이상 문장 행)",
                    start_time
                )
                # 다른 3+ 행들의 모든 문장 (자기 행 제외)
                others_3plus = [(o, no) for i, o, no in only_3plus_quotes if i != idx]
                new_quotes = []
                for my_q, my_n in zip(quotes, norms):
                    is_dup = False
                    for other_q, other_n in others_3plus:
                        if self._is_duplicate(my_n, other_n, quote_a=my_q, quote_b=other_q):
                            is_dup = True
                            break
                    if not is_dup:
                        new_quotes.append(my_q)
                if new_quotes:
                    new_entry = dict(entry)
                    new_entry["발언"] = "  ".join(new_quotes)
                    kept.append((idx, new_entry))

            # 원본 행 순서로 정렬 후 반환
            deduplicated_result = [entry for _idx, entry in sorted(kept, key=lambda x: x[0])]

            progress_tracker.update_progress(
                total_entries, total_entries,
                "[4단계 중 3단계] 중복 제거 중",
                start_time
            )
        except Exception as e:
            print(f"중복 제거 중 오류 발생: {e}")
            traceback.print_exc()
            deduplicated_result = []
        finally:
            progress_tracker.close_tqdm()

        return deduplicated_result