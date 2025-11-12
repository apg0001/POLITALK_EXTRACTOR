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
        merged_data = []

        import time
        start_time = time.time()

        try:
            for i, entry in enumerate(data):
                name = entry["발언자 성명 및 직책"]
                keywords = [name, f"{name[0]} "]

                if not merged_data:
                    merged_data.append(entry)
                elif (merged_data[-1]["기사 제목"] == entry["기사 제목"]) and \
                    (merged_data[-1]["발언자 성명 및 직책"] == entry["발언자 성명 및 직책"]) and \
                    (merged_data[-1]["날짜"] == entry["날짜"]) and \
                    (merged_data[-1]["신문사"] == entry["신문사"]) and \
                        (Merger.check_cases(entry["문장"], entry["문단"], data[i-1]["큰따옴표 발언"].split("  "))):

                    if entry["큰따옴표 발언"] not in merged_data[-1]["큰따옴표 발언"]:
                        merged_data[-1]["큰따옴표 발언"] += ("  " + entry["큰따옴표 발언"])

                    # if entry["문단"] != merged_data[-1]["문단"]:
                    #     merged_data[-1]["문단"] += entry["문단"]
                    # entry["문단"] = self.merge_paragraphs(
                    #     merged_data[-1]["문단"], entry["문단"])
                    merged_data[-1]["문단"] = self.merge_paragraphs(
                        merged_data[-1]["문단"], entry["문단"])
                    merged_data[-1]["문장"] += (" " + entry["문장"])
                else:
                    merged_data.append(entry)

                progress_tracker.update_progress(
                    i + 1, total_entries,
                    "[4단계 중 2단계] 내용 병합 중",
                    start_time
                )

        except Exception as e:
            print(f"내용 병합 중 오류 발생: {e}")
            traceback.print_exc()

        return merged_data

    def merge_paragraphs(self, para1, para2):
        sentences1 = para1.split('. ')  # 문장 단위 분리 (필요하면 문장 구분자 조정)
        sentences2 = para2.split('. ')

        merged = sentences1[:]  # 첫 번째 문단 문장 복사

        for s in sentences2:
            if s not in merged:
                merged.append(s)
                
        print("para1: ", para1)
        print("para2: ", para2)
        print("merged: ", '. '.join(merged))
        print("\n\n----------------------------------------------------------\n\n")

        return '. '.join(merged)
