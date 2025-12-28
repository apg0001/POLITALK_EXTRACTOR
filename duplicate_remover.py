"""중복 제거를 담당하는 클래스"""

import traceback


class DuplicateRemover:
    """중복 제거를 담당하는 클래스"""
    
    def __init__(self):
        """DuplicateRemover 초기화"""
        pass
    
    def _normalize_quote(self, quote):
        """큰따옴표 발언 정규화 (앞뒤 공백만 제거)
        
        Args:
            quote (str): 정규화할 큰따옴표 발언
            
        Returns:
            str: 정규화된 발언
        """
        return quote.strip()
    
    def _calculate_similarity(self, quote1, quote2):
        """두 큰따옴표 발언의 유사도 계산
        
        Args:
            quote1 (str): 첫 번째 발언
            quote2 (str): 두 번째 발언
            
        Returns:
            float: 유사도 (0.0 ~ 1.0)
        """
        norm1 = self._normalize_quote(quote1)
        norm2 = self._normalize_quote(quote2)
        
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if not words1 or not words2:
            return 0.0
        
        common_words = words1 & words2
        similarity = len(common_words) / max(len(words1), len(words2))
        
        return similarity
    
    def _is_subset(self, quote_a, quote_b):
        """룰 3: A가 B의 부분집합인지 확인 (A ⊂ B)
        
        부분 포함 체크: A가 B에 완전히 포함되면서 A != B인 경우
        예: C = "안녕하세요", C' = "안녕하세요 반갑습니다" → C ⊂ C'
        
        Args:
            quote_a (str): 첫 번째 발언 (짧은 쪽)
            quote_b (str): 두 번째 발언 (긴 쪽)
            
        Returns:
            bool: quote_a가 quote_b의 부분집합이면 True
        """
        norm_a = self._normalize_quote(quote_a)
        norm_b = self._normalize_quote(quote_b)
        
        # A가 B에 완전히 포함되면서 A != B인 경우
        # 단순 문자열 포함 체크로는 부족할 수 있으므로 단어 단위로도 확인
        if norm_a in norm_b and norm_a != norm_b:
            return True
        
        # 단어 단위 부분 집합 체크 (더 정확한 판단)
        words_a = set(norm_a.split())
        words_b = set(norm_b.split())
        
        if words_a and words_b:
            # A의 모든 단어가 B에 포함되고, A의 단어 수가 B보다 적으면 부분집합
            return words_a.issubset(words_b) and len(words_a) < len(words_b)
        
        return False
    
    def _are_similar(self, quote1, quote2, threshold=0.7):
        """룰 1: 두 발언이 70% 이상 유사한지 확인
        
        Args:
            quote1 (str): 첫 번째 발언
            quote2 (str): 두 번째 발언
            threshold (float): 유사도 임계값 (기본값 0.7)
            
        Returns:
            bool: 유사도가 임계값 이상이면 True
        """
        similarity = self._calculate_similarity(quote1, quote2)
        return similarity >= threshold
    
    def _are_quotes_identical(self, quotes1, quotes2, similarity_threshold=0.7):
        """두 엔트리의 모든 문장이 유사도 기준으로 일치하는지 확인
        
        전체 엔트리 제거를 위한 체크: 두 엔트리의 모든 문장이 70% 이상 유사하면 True
        각 문장은 다른 엔트리의 문장 중 하나와 매칭되어야 함 (순서 무관)
        
        Args:
            quotes1 (list): 첫 번째 엔트리의 문장 리스트
            quotes2 (list): 두 번째 엔트리의 문장 리스트
            similarity_threshold (float): 유사도 임계값 (기본값 0.7)
            
        Returns:
            bool: 모든 문장이 유사도 기준으로 일치하면 True
        """
        if len(quotes1) != len(quotes2):
            return False
        
        if len(quotes1) == 0:
            return True
        
        # quotes2의 사용 가능한 인덱스 (매칭되지 않은 문장들)
        available_indices = set(range(len(quotes2)))
        
        # quotes1의 각 문장에 대해 quotes2에서 가장 유사한 문장 찾기
        for quote1 in quotes1:
            best_match_idx = None
            best_similarity = 0.0
            
            # 사용 가능한 quotes2 문장 중 가장 유사한 것 찾기
            for idx in available_indices:
                quote2 = quotes2[idx]
                similarity = self._calculate_similarity(quote1, quote2)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_idx = idx
            
            # 유사도가 임계값 이상이고 매칭된 경우
            if best_similarity >= similarity_threshold and best_match_idx is not None:
                # 매칭된 인덱스를 사용 불가능한 목록에서 제거
                available_indices.remove(best_match_idx)
            else:
                # 매칭 실패 → 전체 엔트리가 일치하지 않음
                return False
        
        # 모든 문장이 매칭되었는지 확인 (available_indices가 비어있어야 함)
        return len(available_indices) == 0
    
    def _get_removal_target(self, count1, count2, is_similar_only=False):
        """룰 2: 개수 기반 우선순위 결정
        
        단일 문장 우선 제거: 단일 문장인 행은 복수 문장을 가진 행보다 우선 제거 대상
        
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
        
        # 둘 다 단일 문장이거나 둘 다 복수 문장인 경우
        # 유사도만 체크한 경우 (부분 포함이 아닌 경우)는 문장 단위로 판단
        if is_similar_only:
            # 같은 엔트리 내의 여러 문장이 각각 다른 행과 중복되는 경우를 처리
            # 개수가 같거나 비슷하면 문장 단위로 판단 (길이 기준)
            return None  # 문장 단위로 판단하도록 None 반환
        
        # 부분 포함 관계가 있는 경우는 개수 비교
        if count1 < count2:
            return 'current'  # 개수가 적은 쪽(current)에서 제거
        elif count1 > count2:
            return 'previous'  # 개수가 적은 쪽(previous)에서 제거
        else:
            return None  # 개수 같음, 우선순위 없음
    
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
                current_quotes = [q.strip() for q in current_entry["큰따옴표 발언"].split("  ") if q.strip()]

                # 이전 엔트리들과 비교하여 중복 제거
                previous_entry_idx = 0
                entry_removed = False  # 현재 엔트리가 전체 제거되었는지 여부
                
                while previous_entry_idx < len(previous_entries_cache):
                    previous_entry = deduplicated_result[previous_entry_idx]
                    previous_cache = previous_entries_cache[previous_entry_idx]
                    
                    previous_quotes = previous_cache['original']
                    
                    # 전체 엔트리 중복 체크: 두 엔트리의 모든 문장이 일치하면 한 엔트리 전체 제거
                    if self._are_quotes_identical(current_quotes, previous_quotes):
                        # 단일 문장 우선 제거: 단일 문장인 엔트리를 우선 제거
                        if len(current_quotes) == 1 and len(previous_quotes) > 1:
                            # current가 단일 문장 → current 전체 제거
                            entry_removed = True
                            break
                        elif len(current_quotes) > 1 and len(previous_quotes) == 1:
                            # previous가 단일 문장 → previous 전체 제거
                            del deduplicated_result[previous_entry_idx]
                            del previous_entries_cache[previous_entry_idx]
                            previous_entry_idx -= 1
                            continue
                        else:
                            # 둘 다 단일이거나 둘 다 복수 → 개수가 적은 쪽 제거
                            if len(current_quotes) < len(previous_quotes):
                                entry_removed = True
                                break
                            elif len(current_quotes) > len(previous_quotes):
                                del deduplicated_result[previous_entry_idx]
                                del previous_entries_cache[previous_entry_idx]
                                previous_entry_idx -= 1
                                continue
                            else:
                                # 개수도 같음 → current 제거 (기본)
                                entry_removed = True
                                break
                    
                    # 현재 엔트리의 각 문장을 모든 이전 엔트리와 비교
                    # 같은 엔트리 내의 여러 문장이 각각 다른 행과 중복되는 경우를 처리하기 위해
                    # 각 문장을 모든 이전 엔트리와 비교한 후 제거 결정
                    current_quote_idx = 0
                    
                    while current_quote_idx < len(current_quotes):
                        current_quote = current_quotes[current_quote_idx]
                        is_duplicate = False
                        remove_previous_indices_map = {}  # {previous_entry_idx: [prev_idx, ...]}
                        
                        # 모든 이전 엔트리와 비교 (같은 문장이 여러 이전 엔트리와 중복될 수 있음)
                        temp_previous_entry_idx = 0
                        while temp_previous_entry_idx < len(previous_entries_cache):
                            temp_previous_entry = deduplicated_result[temp_previous_entry_idx]
                            temp_previous_cache = previous_entries_cache[temp_previous_entry_idx]
                            temp_previous_quotes = temp_previous_cache['original']
                            temp_remove_previous_indices = []
                            
                            # 이전 엔트리의 각 문장과 비교
                            for prev_idx, previous_quote in enumerate(temp_previous_quotes):
                                # 1순위: 룰 3 - 부분 포함 체크 (최우선)
                                # 부분 포함 관계가 있을 때는 항상 긴 문장을 유지
                                is_current_subset = self._is_subset(current_quote, previous_quote)
                                is_previous_subset = self._is_subset(previous_quote, current_quote)
                                
                                if is_current_subset or is_previous_subset:
                                    # 부분 포함 관계가 있는 경우, 문장 길이 비교
                                    len_current = len(current_quote)
                                    len_previous = len(previous_quote)
                                    
                                    if len_current < len_previous:
                                        # current가 짧음 → current 제거 (긴 문장인 previous 유지)
                                        is_duplicate = True
                                        break
                                    elif len_current > len_previous:
                                        # previous가 짧음 → previous 제거 (긴 문장인 current 유지)
                                        temp_remove_previous_indices.append(prev_idx)
                                        continue
                                    else:
                                        # 길이가 같으면 부분 포함 관계에 따라 판단
                                        if is_current_subset:
                                            # current ⊂ previous → current 제거
                                            is_duplicate = True
                                            break
                                        elif is_previous_subset:
                                            # previous ⊂ current → previous 제거
                                            temp_remove_previous_indices.append(prev_idx)
                                            continue
                                
                                # 2순위: 룰 1 - 70% 이상 유사도 체크
                                elif self._are_similar(current_quote, previous_quote, 0.7):
                                    # 유사도가 높은 경우, 문장 단위로 비교
                                    # 같은 엔트리 내의 여러 문장이 각각 다른 행과 중복되는 경우를 처리
                                    len_current = len(current_quote)
                                    len_previous = len(previous_quote)
                                    
                                    # 룰 2: 개수 기반 우선순위 적용
                                    # 유사도만 체크한 경우는 문장 단위로 판단
                                    removal_target = self._get_removal_target(
                                        len(current_quotes),
                                        len(temp_previous_quotes),
                                        is_similar_only=True
                                    )
                                    
                                    # 개수 기반 우선순위가 명확한 경우
                                    if removal_target == 'current':
                                        # current 엔트리가 단일 문장이거나 개수가 적음 → current 문장 제거
                                        is_duplicate = True
                                        break
                                    elif removal_target == 'previous':
                                        # previous 엔트리가 단일 문장이거나 개수가 적음 → previous 문장 제거
                                        temp_remove_previous_indices.append(prev_idx)
                                        continue
                                    else:
                                        # 개수 같거나 비슷함 → 길이 기준으로 판단 (긴 쪽 유지, 짧은 쪽 제거)
                                        # 또는 문장 단위 비교 (같은 엔트리 내의 여러 문장이 각각 다른 행과 중복)
                                        if len_current < len_previous:
                                            # current가 짧음 → current 제거
                                            is_duplicate = True
                                            break
                                        elif len_current > len_previous:
                                            # previous가 짧음 → previous 제거
                                            temp_remove_previous_indices.append(prev_idx)
                                            continue
                                        else:
                                            # 길이도 같음 → current 제거 (기본)
                                            is_duplicate = True
                                            break
                            
                            # 이전 엔트리에서 제거할 문장들 기록
                            if temp_remove_previous_indices:
                                remove_previous_indices_map[temp_previous_entry_idx] = temp_remove_previous_indices
                            
                            # 중복이 발견되면 더 이상 비교하지 않음
                            if is_duplicate:
                                break
                            
                            temp_previous_entry_idx += 1
                        
                        # 이전 엔트리들에서 제거할 문장들 처리 (역순으로 제거)
                        if remove_previous_indices_map:
                            # 역순으로 처리 (인덱스 변경 방지)
                            for prev_entry_idx in sorted(remove_previous_indices_map.keys(), reverse=True):
                                prev_quotes = previous_entries_cache[prev_entry_idx]['original']
                                prev_entry = deduplicated_result[prev_entry_idx]
                                
                                for idx in sorted(remove_previous_indices_map[prev_entry_idx], reverse=True):
                                    prev_quotes.pop(idx)
                                
                                # 이전 엔트리 업데이트
                                if prev_quotes:
                                    prev_entry["큰따옴표 발언"] = "  ".join(prev_quotes)
                                    previous_entries_cache[prev_entry_idx]['original'] = prev_quotes
                                    previous_entries_cache[prev_entry_idx]['normalized'] = [self._normalize_quote(q) for q in prev_quotes]
                                else:
                                    # 이전 엔트리의 모든 문장이 제거되면 엔트리 자체 삭제
                                    del deduplicated_result[prev_entry_idx]
                                    del previous_entries_cache[prev_entry_idx]
                        
                        # 현재 문장 제거
                        if is_duplicate:
                            current_quotes.pop(current_quote_idx)
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
                    current_entry["큰따옴표 발언"] = "  ".join(current_quotes)
                    deduplicated_result.append(current_entry)
                    
                    # 캐시에 추가 (다음 비교를 위해)
                    previous_entries_cache.append({
                        'original': current_quotes,
                        'normalized': [self._normalize_quote(q) for q in current_quotes]
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