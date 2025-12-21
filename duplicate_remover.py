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
        
        Args:
            quote_a (str): 첫 번째 발언
            quote_b (str): 두 번째 발언
            
        Returns:
            bool: quote_a가 quote_b의 부분집합이면 True
        """
        norm_a = self._normalize_quote(quote_a)
        norm_b = self._normalize_quote(quote_b)
        return norm_a in norm_b and norm_a != norm_b
    
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
    
    def _get_removal_target(self, count1, count2):
        """룰 2: 개수 기반 우선순위 결정
        
        Args:
            count1 (int): 첫 번째 행의 큰따옴표 문장 개수
            count2 (int): 두 번째 행의 큰따옴표 문장 개수
            
        Returns:
            str or None: 'current' (count1에서 제거), 'previous' (count2에서 제거), None (우선순위 없음)
        """
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
                while previous_entry_idx < len(previous_entries_cache):
                    previous_entry = deduplicated_result[previous_entry_idx]
                    previous_cache = previous_entries_cache[previous_entry_idx]
                    
                    previous_quotes = previous_cache['original']
                    current_quote_idx = 0
                    
                    # 현재 엔트리의 각 문장을 이전 엔트리와 비교
                    while current_quote_idx < len(current_quotes):
                        current_quote = current_quotes[current_quote_idx]
                        is_duplicate = False
                        remove_previous_indices = []
                        
                        # 이전 엔트리의 각 문장과 비교
                        for prev_idx, previous_quote in enumerate(previous_quotes):
                            # 1순위: 룰 3 - 부분 포함 체크 (최우선)
                            if self._is_subset(current_quote, previous_quote):
                                # current ⊂ previous → current 제거
                                is_duplicate = True
                                break
                            elif self._is_subset(previous_quote, current_quote):
                                # previous ⊂ current → previous 제거
                                remove_previous_indices.append(prev_idx)
                                continue
                            
                            # 2순위: 룰 1 - 70% 이상 유사도 체크
                            elif self._are_similar(current_quote, previous_quote, 0.7):
                                # 룰 2: 개수 기반 우선순위 적용
                                removal_target = self._get_removal_target(
                                    len(current_quotes),
                                    len(previous_quotes)
                                )
                                
                                if removal_target == 'current':
                                    # current에서 제거
                                    is_duplicate = True
                                    break
                                elif removal_target == 'previous':
                                    # previous에서 제거
                                    remove_previous_indices.append(prev_idx)
                                    continue
                                else:
                                    # 우선순위 없음 → current 제거 (기본)
                                    is_duplicate = True
                                    break
                        
                        # 이전 엔트리에서 제거할 문장들 처리 (역순으로 제거)
                        if remove_previous_indices:
                            for idx in sorted(remove_previous_indices, reverse=True):
                                previous_quotes.pop(idx)
                            
                            # 이전 엔트리 업데이트
                            if previous_quotes:
                                previous_entry["큰따옴표 발언"] = "  ".join(previous_quotes)
                                previous_cache['original'] = previous_quotes
                                previous_cache['normalized'] = [self._normalize_quote(q) for q in previous_quotes]
                            else:
                                # 이전 엔트리의 모든 문장이 제거되면 엔트리 자체 삭제
                                del deduplicated_result[previous_entry_idx]
                                del previous_entries_cache[previous_entry_idx]
                                previous_entry_idx -= 1
                                break
                        
                        # 현재 문장 제거
                        if is_duplicate:
                            current_quotes.pop(current_quote_idx)
                        else:
                            current_quote_idx += 1
                    
                    previous_entry_idx += 1

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