"""중복 제거를 담당하는 클래스"""

import traceback
from utils import ProgressTracker


class DuplicateRemover:
    """중복 제거를 담당하는 클래스"""
    
    def __init__(self):
        """DuplicateRemover 초기화"""
        pass
    
    def remove_duplicates(self, data, progress_tracker):
        """중복 발언 제거
        
        유사도 검사를 통해 중복된 발언문을 제거합니다.
        최근 200개 문장과 비교하여 중복을 찾아 제거합니다.
        
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
        sentence_sets = []
        duplicate_removed_data = []
        total_entries = len(data)
        progress_tracker.progress_bar['maximum'] = total_entries

        import time
        start_time = time.time()

        try:
            for i, entry in enumerate(data):
                original_sentences = entry["큰따옴표 발언"].split("  ")
                normalized_sentences = [text_processor.normalize_text(s) for s in original_sentences]

                j = 0
                while j < len(sentence_sets):
                    existing_entry = duplicate_removed_data[j]
                    existing_sentences = existing_entry['큰따옴표 발언'].split("  ")
                    existing_normalized = sentence_sets[j]['normalized']

                    idx_new = 0
                    while idx_new < len(normalized_sentences):
                        matched = False
                        idx_exist = 0
                        while idx_exist < len(existing_normalized):
                            # 한 문장이 다른 문장을 완전히 포함하는 경우
                            if normalized_sentences[idx_new] in existing_normalized[idx_exist] or \
                               text_processor.calculate_similarity(normalized_sentences[idx_new], existing_normalized[idx_exist], "min"):
                                del existing_sentences[idx_exist]
                                del existing_normalized[idx_exist]
                                
                                if existing_sentences:
                                    existing_entry["큰따옴표 발언"] = "  ".join(existing_sentences)
                                    sentence_sets[j] = {
                                        'original': existing_entry["큰따옴표 발언"],
                                        'normalized': existing_normalized
                                    }
                                    continue
                                else:
                                    del duplicate_removed_data[j]
                                    del sentence_sets[j]
                                    j -= 1
                                    break
                            elif existing_normalized[idx_exist] in normalized_sentences[idx_new]:
                                del original_sentences[idx_new]
                                del normalized_sentences[idx_new]
                                matched = True
                                break
                            elif normalized_sentences[idx_new] == existing_normalized[idx_exist] or \
                                 text_processor.calculate_similarity(normalized_sentences[idx_new], existing_normalized[idx_exist], "max"):
                                if len(normalized_sentences) <= len(existing_normalized):
                                    del original_sentences[idx_new]
                                    del normalized_sentences[idx_new]
                                    matched = True
                                    break
                                else:
                                    del existing_sentences[idx_exist]
                                    del existing_normalized[idx_exist]
                                    if existing_sentences:
                                        existing_entry["큰따옴표 발언"] = "  ".join(existing_sentences)
                                        sentence_sets[j] = {
                                            'original': existing_entry["큰따옴표 발언"],
                                            'normalized': existing_normalized
                                        }
                                        continue
                                    else:
                                        del duplicate_removed_data[j]
                                        del sentence_sets[j]
                                        j -= 1
                                        break
                            idx_exist += 1
                        if not matched:
                            idx_new += 1
                    j += 1

                if original_sentences:
                    entry["큰따옴표 발언"] = "  ".join(original_sentences)
                    duplicate_removed_data.append(entry)
                    sentence_sets.append({
                        'original': entry["큰따옴표 발언"],
                        'normalized': [text_processor.normalize_text(s) for s in entry["큰따옴표 발언"].split("  ")]
                    })
                    if len(sentence_sets) > 200:
                        sentence_sets.pop(0)
                        duplicate_removed_data.pop(0)

                progress_tracker.update_progress(
                    i + 1, total_entries,
                    "[4단계 중 3단계] 중복 제거 중",
                    start_time
                )

        except Exception as e:
            print(f"중복 제거 중 오류 발생: {e}")
            traceback.print_exc()

        return duplicate_removed_data