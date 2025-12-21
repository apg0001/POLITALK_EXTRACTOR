"""데이터 병합을 담당하는 클래스"""

import traceback


class DataMerger:
    """데이터 병합을 담당하는 클래스"""

    def __init__(self, debug=False):
        """DataMerger 초기화
        
        Args:
            debug (bool): 디버깅 모드 활성화 여부
        """
        self.debug = debug

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
                
                # 디버깅: 메타데이터 비교 결과 출력
                if self.debug:
                    print(f"\n[병합 시도 #{current_idx + 1}]")
                    print(f"  기사 제목 일치: {is_same_article}")
                    print(f"  발언자 일치: {is_same_speaker} ({last_merged_entry['발언자 성명 및 직책']} == {current_entry['발언자 성명 및 직책']})")
                    print(f"  날짜 일치: {is_same_date} ({last_merged_entry['날짜']} == {current_entry['날짜']})")
                    print(f"  신문사 일치: {is_same_newspaper} ({last_merged_entry['신문사']} == {current_entry['신문사']})")
                
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
                
                # 디버깅: 문맥적 병합 가능 여부 출력
                if self.debug:
                    print(f"  문맥적 병합 가능: {is_contextually_mergeable}")
                    if previous_quoted_speeches:
                        print(f"  이전 발언: {previous_quoted_speeches[:2]}...")  # 처음 2개만 출력
                    print(f"  현재 문장: {current_entry['문장'][:50]}...")  # 처음 50자만 출력

                # 모든 조건을 만족하면 병합 수행
                if (is_same_article and is_same_speaker and is_same_date and 
                    is_same_newspaper and is_contextually_mergeable):
                    
                    if self.debug:
                        print(f"  ✓ 병합 수행")
                    
                    # 큰따옴표 발언 병합 (중복 제거)
                    current_quoted_speech = current_entry["큰따옴표 발언"]
                    last_quoted_speech = last_merged_entry["큰따옴표 발언"]
                    
                    if current_quoted_speech not in last_quoted_speech:
                        last_merged_entry["큰따옴표 발언"] = f"{last_quoted_speech}  {current_quoted_speech}"
                        if self.debug:
                            print(f"  큰따옴표 발언 병합: {len(last_quoted_speech)}자 + {len(current_quoted_speech)}자")
                    else:
                        if self.debug:
                            print(f"  큰따옴표 발언 중복 감지 (추가 안 함)")

                    # 문단 병합 (중복 문장 제거)
                    if self.debug:
                        print(f"  문단 병합 전:")
                        print(f"    첫 번째: {last_merged_entry['문단'][:100]}...")
                        print(f"    두 번째: {current_entry['문단'][:100]}...")
                    
                    merged_paragraph = self.merge_paragraphs(
                        last_merged_entry["문단"], 
                        current_entry["문단"],
                        debug=self.debug
                    )
                    last_merged_entry["문단"] = merged_paragraph
                    
                    if self.debug:
                        print(f"  문단 병합 후: {merged_paragraph[:100]}...")
                    
                    # 문장 병합
                    last_merged_entry["문장"] = f"{last_merged_entry['문장']}  {current_entry['문장']}"
                else:
                    # 병합 불가능한 경우 새 엔트리로 추가
                    if self.debug:
                        print(f"  ✗ 병합 불가 (새 엔트리로 추가)")
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

    def merge_paragraphs(self, first_paragraph, second_paragraph, debug=False):
        """두 문단을 병합하여 중복 문장 제거
        
        두 문단을 줄바꿈 기준으로 분리한 후, 중복되는 문장을 제거하고 병합합니다.
        같은 기사에서 발췌한 것이므로 완전 일치하는 문장만 중복으로 판단합니다.
        
        Args:
            first_paragraph (str): 첫 번째 문단
            second_paragraph (str): 두 번째 문단
            debug (bool): 디버깅 모드 활성화 여부
            
        Returns:
            str: 중복이 제거된 병합된 문단 (줄바꿈으로 구분)
        """
        # 정규화 함수: 앞뒤 공백만 제거
        def normalize(text):
            return text.strip()
        
        # 문장 분리: 줄바꿈으로 분리 (리스트 컴프리헨션)
        def split_by_newline(paragraph):
            if not paragraph:
                return []
            # 줄바꿈으로 분리하고 앞뒤 공백 제거
            return [line.strip() for line in paragraph.split('\n') if line.strip()]
        
        # 두 문단을 줄바꿈으로 분리
        sentences_first = split_by_newline(first_paragraph)
        sentences_second = split_by_newline(second_paragraph)
        
        if debug:
            print(f"    [문장 분리] 첫 번째: {len(sentences_first)}개, 두 번째: {len(sentences_second)}개")
        
        # 정규화된 문장을 키로 하는 딕셔너리 (중복 제거용)
        # {정규화된_문장: 원본_문장} 형태
        normalized_to_original = {}
        
        # 첫 번째 문단 추가
        for sent in sentences_first:
            normalized = normalize(sent)
            if normalized:
                normalized_to_original[normalized] = sent
        
        if debug:
            print(f"    [병합 시작] 첫 번째 문단 {len(normalized_to_original)}개 문장 추가됨")
        
        # 두 번째 문단 추가 (중복 제거)
        duplicate_count = 0
        for sent in sentences_second:
            normalized = normalize(sent)
            if normalized:
                if normalized in normalized_to_original:
                    duplicate_count += 1
                    if debug:
                        print(f"    [중복 감지 #{duplicate_count}] {sent[:60]}...")
                else:
                    normalized_to_original[normalized] = sent
                    if debug:
                        print(f"    [추가] {sent[:60]}...")
        
        if debug:
            print(f"    [병합 완료] 총 {len(normalized_to_original)}개 문장 (중복 {duplicate_count}개 제거)")
        
        # 순서 유지: 첫 번째 문단 순서 + 두 번째 문단의 새 문장들
        seen = set()
        result = []
        
        # 첫 번째 문단 순서대로
        for sent in sentences_first:
            normalized = normalize(sent)
            if normalized and normalized not in seen:
                result.append(normalized_to_original[normalized])
                seen.add(normalized)
        
        # 두 번째 문단에서 새로 추가된 것만
        for sent in sentences_second:
            normalized = normalize(sent)
            if normalized and normalized not in seen:
                result.append(normalized_to_original[normalized])
                seen.add(normalized)
        
        # 줄바꿈으로 결합
        return '\n'.join(result)
