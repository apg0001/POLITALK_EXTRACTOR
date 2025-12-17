"""중복 제거를 담당하는 클래스"""

import traceback


class DuplicateRemover:
    """중복 제거를 담당하는 클래스"""
    
    def __init__(self):
        """DuplicateRemover 초기화"""
        pass
    
    def remove_duplicates(self, data, progress_tracker):
        """중복 발언 제거
        
        유사도 검사를 통해 중복된 발언문을 제거합니다.
        각 엔트리의 발언문을 이전 엔트리들과 비교하여 중복을 찾아 제거합니다.
        
        중복 판단 기준:
        1. 한 문장이 다른 문장을 완전히 포함하는 경우
        2. 유사도가 높은 경우 (min/max 기준)
        3. 완전히 동일한 경우
        
        Args:
            data (list): 중복 제거할 데이터 리스트
            progress_tracker (ProgressTracker): 진행률 추적기
            
        Returns:
            list: 중복이 제거된 데이터 리스트
        """
        if not data:
            print("[중복 제거] 저장할 데이터가 없습니다.")
            return []

        from text_manager import TextProcessor
        
        text_processor = TextProcessor()
        
        # 이전 엔트리들의 정규화된 발언문을 저장 (비교용)
        # 각 항목: {'original': 원본 발언문, 'normalized': 정규화된 발언문 리스트}
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
                current_original_sentences = current_entry["큰따옴표 발언"].split("  ")
                current_normalized_sentences = [
                    text_processor.normalize_text(sentence) 
                    for sentence in current_original_sentences
                ]

                # 이전 엔트리들과 비교하여 중복 제거
                previous_entry_idx = 0
                while previous_entry_idx < len(previous_entries_cache):
                    previous_entry = deduplicated_result[previous_entry_idx]
                    previous_cache = previous_entries_cache[previous_entry_idx]
                    
                    previous_original_sentences = previous_entry['큰따옴표 발언'].split("  ")
                    previous_normalized_sentences = previous_cache['normalized']

                    # 현재 엔트리의 각 문장을 이전 엔트리와 비교
                    current_sentence_idx = 0
                    while current_sentence_idx < len(current_normalized_sentences):
                        current_sentence_normalized = current_normalized_sentences[current_sentence_idx]
                        is_duplicate_found = False
                        
                        # 이전 엔트리의 각 문장과 비교
                        previous_sentence_idx = 0
                        while previous_sentence_idx < len(previous_normalized_sentences):
                            previous_sentence_normalized = previous_normalized_sentences[previous_sentence_idx]
                            
                            # 중복 판단: Case 1 - 현재 문장이 이전 문장을 포함하거나 유사도 높음
                            if (current_sentence_normalized in previous_sentence_normalized or
                                text_processor.calculate_similarity(
                                    current_sentence_normalized, 
                                    previous_sentence_normalized, 
                                    "min"
                                )):
                                # 이전 엔트리에서 중복 문장 제거
                                del previous_original_sentences[previous_sentence_idx]
                                del previous_normalized_sentences[previous_sentence_idx]
                                
                                # 이전 엔트리에 남은 문장이 있으면 업데이트
                                if previous_original_sentences:
                                    previous_entry["큰따옴표 발언"] = "  ".join(previous_original_sentences)
                                    previous_cache['original'] = previous_entry["큰따옴표 발언"]
                                    previous_cache['normalized'] = previous_normalized_sentences
                                    continue  # 같은 이전 문장 인덱스로 다시 비교
                                else:
                                    # 이전 엔트리의 모든 문장이 제거되면 엔트리 자체 삭제
                                    del deduplicated_result[previous_entry_idx]
                                    del previous_entries_cache[previous_entry_idx]
                                    previous_entry_idx -= 1
                                    break
                            
                            # 중복 판단: Case 2 - 이전 문장이 현재 문장을 포함
                            elif previous_sentence_normalized in current_sentence_normalized:
                                # 현재 엔트리에서 중복 문장 제거
                                del current_original_sentences[current_sentence_idx]
                                del current_normalized_sentences[current_sentence_idx]
                                is_duplicate_found = True
                                break
                            
                            # 중복 판단: Case 3 - 완전히 동일하거나 유사도가 매우 높음
                            elif (current_sentence_normalized == previous_sentence_normalized or
                                  text_processor.calculate_similarity(
                                      current_sentence_normalized, 
                                      previous_sentence_normalized, 
                                      "max"
                                  )):
                                # 더 짧은 쪽을 제거 (같은 길이면 현재 문장 제거)
                                if len(current_normalized_sentences) <= len(previous_normalized_sentences):
                                    del current_original_sentences[current_sentence_idx]
                                    del current_normalized_sentences[current_sentence_idx]
                                    is_duplicate_found = True
                                    break
                                else:
                                    # 이전 엔트리에서 중복 문장 제거
                                    del previous_original_sentences[previous_sentence_idx]
                                    del previous_normalized_sentences[previous_sentence_idx]
                                    
                                    if previous_original_sentences:
                                        previous_entry["큰따옴표 발언"] = "  ".join(previous_original_sentences)
                                        previous_cache['original'] = previous_entry["큰따옴표 발언"]
                                        previous_cache['normalized'] = previous_normalized_sentences
                                        continue
                                    else:
                                        del deduplicated_result[previous_entry_idx]
                                        del previous_entries_cache[previous_entry_idx]
                                        previous_entry_idx -= 1
                                        break
                            
                            previous_sentence_idx += 1
                        
                        # 중복이 발견되지 않았으면 다음 현재 문장으로
                        if not is_duplicate_found:
                            current_sentence_idx += 1
                    
                    previous_entry_idx += 1

                # 중복 제거 후 남은 문장이 있으면 결과에 추가
                if current_original_sentences:
                    current_entry["큰따옴표 발언"] = "  ".join(current_original_sentences)
                    deduplicated_result.append(current_entry)
                    
                    # 캐시에 추가 (다음 비교를 위해)
                    previous_entries_cache.append({
                        'original': current_entry["큰따옴표 발언"],
                        'normalized': [
                            text_processor.normalize_text(sentence) 
                            for sentence in current_entry["큰따옴표 발언"].split("  ")
                        ]
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