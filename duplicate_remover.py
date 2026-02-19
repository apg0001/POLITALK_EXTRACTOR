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
    
    
    def _get_removal_target(self, count1, count2, is_similar_only=False):
        """룰 2: 개수 기반 우선순위 결정
        
        단일 문장 우선 제거: 단일 문장인 행은 복수 문장을 가진 행보다 우선 제거 대상
        개수가 적은 쪽 우선 제거: 개수가 다르면 적은 쪽을 제거
        
        Args:
            count1 (int): 첫 번째 행의 큰따옴표 문장 개수
            count2 (int): 두 번째 행의 큰따옴표 문장 개수
            is_similar_only (bool): 유사도만 체크한 경우 (부분 포함이 아닌 경우) True
            
        Returns:
            str or None: 'current' (count1에서 제거), 'previous' (count2에서 제거), None (우선순위 없음)
        """
        # 단일 문장 우선 제거: 단일 문장인 행을 우선 제거
        if count1 == 1 and count2 > 1:
            return 'current'  # current가 단일 문장 → current 제거
        elif count1 > 1 and count2 == 1:
            return 'previous'  # previous가 단일 문장 → previous 제거
        
        # 개수가 다르면 적은 쪽 제거 (단일/복수 구분 없이)
        if count1 < count2:
            return 'current'  # 개수가 적은 쪽(current)에서 제거
        elif count1 > count2:
            return 'previous'  # 개수가 적은 쪽(previous)에서 제거
        
        # 개수가 같으면 우선순위 없음 (길이 기준 등으로 판단)
        return None
    
    def remove_duplicates(self, data, progress_tracker):
        """중복 발언 제거
        
        중복제거 룰에 따라 중복된 큰따옴표 발언을 제거합니다.
        각 엔트리의 큰따옴표 발언을 이전 엔트리들과 비교하여 중복을 찾아 제거합니다.
        
        중복 판단 룰 (우선순위 순):
        1. 룰 3: 부분 포함 (A ⊂ B) → A 제거
        2. 룰 1: 70% 이상 유사도 → 동일 처리
        3. 룰 2: 개수 기반 우선순위 → 개수가 적은 행에서 제거, 많은 행 유지
        
        Args:
            data (list): 중복 제거할 데이터 리스트
            progress_tracker (ProgressTracker): 진행률 추적기
            
        Returns:
            list: 중복이 제거된 데이터 리스트
        """
        if not data:
            print("[중복 제거] 저장할 데이터가 없습니다.")
            return []
        
        # 이전 엔트리들의 정규화된 발언문을 저장 (비교용)
        # 각 항목: {'original': 원본 발언문 리스트, 'normalized': 정규화된 발언문 리스트}
        previous_entries_cache = []
        
        # 중복이 제거된 최종 결과
        deduplicated_result = []
        
        total_entries = len(data)
        progress_tracker.progress_bar['maximum'] = total_entries
        progress_tracker.initialize_tqdm(total_entries, "[4단계 중 3단계] 중복 제거 중")

        import time
        start_time = time.time()

        try:
            # 각 엔트리를 순회하며 중복 확인
            for current_idx, current_entry in enumerate(data):
                # 현재 엔트리의 발언문을 문장 단위로 분리
                current_quotes = [q.strip() for q in current_entry["발언"].split("  ") if q.strip()]
                
                # 큰따옴표 발언이 없는 행은 제거 (결과에 추가하지 않음)
                if not current_quotes:
                    # 진행률 업데이트
                    progress_tracker.update_progress(
                        current_idx + 1, total_entries,
                        "[4단계 중 3단계] 중복 제거 중",
                        start_time
                    )
                    continue
                
                # 현재 엔트리의 모든 문장을 미리 정규화 (한 번만 수행하여 성능 최적화)
                current_normalized_quotes = [self._normalize_quote(q) for q in current_quotes]

                # 이전 엔트리들과 비교하여 중복 제거
                # 성능 최적화: 위로 최대 30개 엔트리만 비교
                comparison_window = 30
                start_idx = max(0, len(previous_entries_cache) - comparison_window)
                previous_entry_idx = start_idx
                entry_removed = False  # 현재 엔트리가 전체 제거되었는지 여부
                
                while previous_entry_idx < len(previous_entries_cache):                    
                    # 현재 엔트리의 각 문장을 모든 이전 엔트리와 비교
                    # 같은 엔트리 내의 여러 문장이 각각 다른 행과 중복되는 경우를 처리하기 위해
                    # 각 문장을 독립적으로 모든 이전 엔트리와 비교한 후 제거 결정
                    current_quote_idx = 0
                    
                    while current_quote_idx < len(current_quotes):
                        current_quote = current_quotes[current_quote_idx]
                        current_normalized = current_normalized_quotes[current_quote_idx]
                        is_current_duplicate = False  # 현재 문장이 중복인지
                        remove_previous_indices_map = {}  # {previous_entry_idx: [prev_idx, ...]}
                        
                        # 모든 이전 엔트리와 비교
                        # 성능 최적화: 위로 최대 30개 엔트리만 비교
                        # 핵심: 같은 엔트리 내의 다른 문장이 다른 이전 엔트리와 중복될 수 있으므로
                        # 각 문장을 독립적으로 이전 엔트리와 비교해야 함
                        temp_comparison_window = 30
                        temp_start_idx = max(0, len(previous_entries_cache) - temp_comparison_window)
                        temp_previous_entry_idx = temp_start_idx
                        while temp_previous_entry_idx < len(previous_entries_cache):
                            temp_previous_cache = previous_entries_cache[temp_previous_entry_idx]
                            temp_previous_quotes = temp_previous_cache['original']
                            temp_previous_normalized = temp_previous_cache['normalized']
                            temp_remove_previous_indices = []
                            
                            # 이전 엔트리의 각 문장과 비교
                            for prev_idx, previous_quote in enumerate(temp_previous_quotes):
                                previous_normalized = temp_previous_normalized[prev_idx]
                                
                                # 1순위: 룰 3 - 부분 포함 체크 (최우선)
                                # 정규화된 문장으로 비교
                                is_current_subset = self._is_subset(current_normalized, previous_normalized)
                                is_previous_subset = self._is_subset(previous_normalized, current_normalized)
                                
                                if is_current_subset or is_previous_subset:
                                    # 부분 포함 관계가 가장 먼저 고려됨
                                    # A < A'인 경우 A 문장을 삭제
                                    if is_current_subset:
                                        # current가 previous의 부분집합 → current 제거
                                        is_current_duplicate = True
                                        break
                                    elif is_previous_subset:
                                        # previous가 current의 부분집합 → previous 제거
                                        temp_remove_previous_indices.append(prev_idx)
                                        continue
                                
                                # 2순위: 룰 1 - 유사도 체크 (기본값 0.5 사용)
                                # 정규화된 문장을 직접 전달하여 재정규화 방지
                                elif self._are_similar(current_quote, previous_quote,
                                                       norm1=current_normalized, norm2=previous_normalized):
                                    len_current = len(current_normalized)
                                    len_previous = len(previous_normalized)
                                    
                                    # 룰 2: 개수 기반 우선순위 적용
                                    removal_target = self._get_removal_target(
                                        len(current_quotes),
                                        len(temp_previous_quotes),
                                        is_similar_only=True
                                    )
                                    
                                    if removal_target == 'current':
                                        # current 문장 제거
                                        is_current_duplicate = True
                                        break
                                    elif removal_target == 'previous':
                                        # previous 문장 제거
                                        temp_remove_previous_indices.append(prev_idx)
                                        continue
                                    else:
                                        # 개수 같거나 비슷함 → 길이 기준으로 판단
                                        if len_current < len_previous:
                                            is_current_duplicate = True
                                            break
                                        elif len_current > len_previous:
                                            temp_remove_previous_indices.append(prev_idx)
                                            continue
                                        else:
                                            # 길이도 같음 → current 제거 (기본)
                                            is_current_duplicate = True
                                            break
                            
                            # 이전 엔트리에서 제거할 문장들 기록
                            if temp_remove_previous_indices:
                                remove_previous_indices_map[temp_previous_entry_idx] = temp_remove_previous_indices
                            
                            # 현재 문장이 중복으로 판단되면 더 이상 비교하지 않음
                            if is_current_duplicate:
                                break
                            
                            temp_previous_entry_idx += 1
                        
                        # 이전 엔트리들에서 제거할 문장들 처리 (역순으로 제거)
                        if remove_previous_indices_map:
                            for prev_entry_idx in sorted(remove_previous_indices_map.keys(), reverse=True):
                                prev_quotes = previous_entries_cache[prev_entry_idx]['original']
                                prev_entry = deduplicated_result[prev_entry_idx]
                                
                                for idx in sorted(remove_previous_indices_map[prev_entry_idx], reverse=True):
                                    prev_quotes.pop(idx)
                                
                                # 이전 엔트리 업데이트
                                if prev_quotes:
                                    prev_entry["발언"] = "  ".join(prev_quotes)
                                    previous_entries_cache[prev_entry_idx]['original'] = prev_quotes
                                    previous_entries_cache[prev_entry_idx]['normalized'] = [self._normalize_quote(q) for q in prev_quotes]
                                else:
                                    # 이전 엔트리의 모든 문장이 제거되면 엔트리 자체 삭제
                                    del deduplicated_result[prev_entry_idx]
                                    del previous_entries_cache[prev_entry_idx]
                        
                        # 현재 문장 제거
                        if is_current_duplicate:
                            current_quotes.pop(current_quote_idx)
                            current_normalized_quotes.pop(current_quote_idx)  # 정규화된 버전도 함께 제거
                            # 인덱스는 증가시키지 않음 (다음 문장이 같은 인덱스로 이동)
                        else:
                            current_quote_idx += 1
                    
                    previous_entry_idx += 1

                # 전체 엔트리가 제거된 경우 스킵
                if entry_removed:
                    # 진행률 업데이트
                    progress_tracker.update_progress(
                        current_idx + 1, total_entries,
                        "[4단계 중 3단계] 중복 제거 중",
                        start_time
                    )
                    continue

                # 중복 제거 후 남은 문장이 있으면 결과에 추가
                if current_quotes:
                    current_entry["발언"] = "  ".join(current_quotes)
                    deduplicated_result.append(current_entry)
                    
                    # 캐시에 추가 (다음 비교를 위해)
                    # 남은 문장들에 대한 정규화된 버전도 함께 저장 (캐시 활용)
                    # current_normalized_quotes는 이미 정규화되어 있으므로 재정규화 불필요
                    remaining_normalized = current_normalized_quotes[:]  # 복사본 사용
                    
                    previous_entries_cache.append({
                        'original': current_quotes,
                        'normalized': remaining_normalized
                    })

                # 진행률 업데이트
                progress_tracker.update_progress(
                    current_idx + 1, total_entries,
                    "[4단계 중 3단계] 중복 제거 중",
                    start_time
                )

        except Exception as e:
            print(f"중복 제거 중 오류 발생: {e}")
            traceback.print_exc()
        finally:
            progress_tracker.close_tqdm()

        return deduplicated_result