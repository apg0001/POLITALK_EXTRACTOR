"""데이터 병합을 담당하는 클래스"""

import traceback


class DataMerger:
    """데이터 병합을 담당하는 클래스"""

    def __init__(self):
        """DataMerger 초기화"""
        pass

    def merge_data(self, data, progress_tracker):
        """내용 병합

        같은 기사, 같은 발언자, 같은 날짜, 같은 신문사의 연속된 발언을 병합합니다.
        접속사나 문맥을 고려하여 관련된 발언들을 하나로 합칩니다.

        병합 조건:
        1. 기사 제목, 발언자, 날짜, 신문사가 모두 동일
        2. 이전 발언과 문맥적으로 연결되어 병합 가능한 경우 (Merger.check_cases)

        Args:
            data (list): 병합할 데이터 리스트
            progress_tracker (ProgressTracker): 진행률 추적기

        Returns:
            list: 병합된 데이터 리스트
        """
        if not data:
            print("[병합] 저장할 데이터가 없습니다.")
            return []

        from text_manager import Merger

        total_entries = len(data)
        progress_tracker.progress_bar['maximum'] = total_entries
        progress_tracker.initialize_tqdm(total_entries, "[4단계 중 2단계] 내용 병합 중")
        merged_result = []  # 병합된 결과를 저장할 리스트

        import time
        start_time = time.time()

        try:
            # 각 엔트리를 순회하며 병합 가능 여부 확인
            for current_idx, current_entry in enumerate(data):
                # 첫 번째 엔트리는 그대로 추가
                if not merged_result:
                    merged_result.append(current_entry)
                    continue

                # 마지막으로 병합된 엔트리 가져오기
                last_merged_entry = merged_result[-1]
                
                # 병합 가능 여부 확인: 메타데이터가 모두 동일한지 확인
                is_same_article = (last_merged_entry["기사 제목"] == current_entry["기사 제목"])
                is_same_speaker = (last_merged_entry["발언자 성명 및 직책"] == current_entry["발언자 성명 및 직책"])
                is_same_date = (last_merged_entry["날짜"] == current_entry["날짜"])
                is_same_newspaper = (last_merged_entry["신문사"] == current_entry["신문사"])
                
                # 이전 발언의 큰따옴표 발언들을 리스트로 변환 (문맥 확인용)
                previous_quoted_speeches = []
                if current_idx > 0:
                    previous_quoted_speeches = data[current_idx - 1]["큰따옴표 발언"].split("  ")
                
                # 문맥적으로 병합 가능한지 확인 (접속사, 문장 구조 등 고려)
                is_contextually_mergeable = Merger.check_cases(
                    current_entry["문장"], 
                    current_entry["문단"], 
                    previous_quoted_speeches
                )

                # 모든 조건을 만족하면 병합 수행
                if (is_same_article and is_same_speaker and is_same_date and 
                    is_same_newspaper and is_contextually_mergeable):
                    
                    # 큰따옴표 발언 병합 (중복 제거)
                    current_quoted_speech = current_entry["큰따옴표 발언"]
                    last_quoted_speech = last_merged_entry["큰따옴표 발언"]
                    
                    if current_quoted_speech not in last_quoted_speech:
                        last_merged_entry["큰따옴표 발언"] = f"{last_quoted_speech}  {current_quoted_speech}"

                    # 문단 병합 (중복 문장 제거)
                    last_merged_entry["문단"] = self.merge_paragraphs(
                        last_merged_entry["문단"], 
                        current_entry["문단"]
                    )
                    
                    # 문장 병합
                    last_merged_entry["문장"] = f"{last_merged_entry['문장']}  {current_entry['문장']}"
                else:
                    # 병합 불가능한 경우 새 엔트리로 추가
                    merged_result.append(current_entry)

                # 진행률 업데이트
                progress_tracker.update_progress(
                    current_idx + 1, total_entries,
                    "[4단계 중 2단계] 내용 병합 중",
                    start_time
                )

        except Exception as e:
            print(f"내용 병합 중 오류 발생: {e}")
            traceback.print_exc()
        finally:
            progress_tracker.close_tqdm()

        return merged_result

    def merge_paragraphs(self, first_paragraph, second_paragraph):
        """두 문단을 병합하여 중복 문장 제거
        
        두 문단을 문장 단위로 분리한 후, 중복되는 문장을 제거하고 병합합니다.
        
        Args:
            first_paragraph (str): 첫 번째 문단
            second_paragraph (str): 두 번째 문단
            
        Returns:
            str: 중복이 제거된 병합된 문단
        """
        # 문장 단위로 분리 (마침표와 공백 기준)
        sentences_from_first = first_paragraph.split('. ')
        sentences_from_second = second_paragraph.split('. ')

        # 첫 번째 문단의 문장들을 복사
        merged_sentences = sentences_from_first[:]

        # 두 번째 문단의 문장들을 추가 (중복 제거)
        for sentence in sentences_from_second:
            if sentence and sentence not in merged_sentences:
                merged_sentences.append(sentence)

        # 문장들을 다시 문단으로 결합
        return '. '.join(merged_sentences)
