"""Excel 파일 저장을 담당하는 클래스"""

import openpyxl
import traceback
import csv
import os
from utils import ProgressTracker


class ExcelWriter:
    """Excel 파일 저장을 담당하는 클래스"""
    
    def __init__(self):
        """ExcelWriter 초기화"""
        pass
    
    def save_data_to_excel(self, data, excel_file, progress_tracker):
        """추출된 데이터를 엑셀 파일로 저장
        
        AI를 사용하여 발언의 목적과 배경을 추출한 후 Excel 파일로 저장합니다.
        
        Args:
            data (list): 저장할 데이터 리스트
            excel_file (str): 저장할 Excel 파일 경로
            progress_tracker (ProgressTracker): 진행률 추적기
        """
        if not data:
            print("[엑셀 파일 저장] 저장할 데이터가 없습니다.")
            return

        try:
            # 데이터 전처리 및 AI 추출 수행 후, 저장에 필요한 headers/rows 생성
            headers, rows = self._prepare_rows_with_ai(data, progress_tracker)

            # 저장 전용 함수로 Excel + CSV 저장 수행
            self._save_rows_to_excel_and_csv(headers, rows, excel_file)
            print(f"엑셀 파일이 '{excel_file}'로 저장되었습니다.")
        except Exception as e:
            print(f"발언의 배경 추출 및 엑셀 파일 저장 중 오류 발생: {e}")
            traceback.print_exc()

    def _prepare_rows_with_ai(self, data, progress_tracker):
        """AI 추출을 수행하여 저장에 필요한 headers와 rows를 생성합니다.

        Returns:
            (tuple): (headers: list[str], rows: list[list])
        """
        workbook_headers = [
            "날짜", "발언자 성명 및 직책", "신문사", "기사 제목",
            "발언의 배경", "문단", "발언의 목적 취지", "큰따옴표 발언",
        ]

        total_entries = len(data)
        progress_tracker.progress_bar['maximum'] = total_entries

        prev_paragraph = None
        prev_title = None

        # AI 추출기 초기화
        from extract_purpose import PurposeExtractor
        from extract_topic_summary import TopicExtractor
        purpose_extractor = PurposeExtractor()
        topic_extractor = TopicExtractor()

        # None 값을 빈 문자열로 정규화
        for _, entry in enumerate(data):
            for key, value in entry.items():
                if value is None:
                    entry[key] = ""

        import time
        start_time = time.time()

        rows = []
        for i, entry in enumerate(data):
            if prev_title != entry["기사 제목"]:
                prev_title = entry["기사 제목"]

            entry["발언의 목적 취지"] = purpose_extractor.extract_purpose(
                entry["발언자 성명 및 직책"], entry["기사 제목"], entry.get("문장", ""), entry["문단"]
            )

            if prev_title is not None and prev_paragraph is not None:
                if prev_title == entry["기사 제목"] and prev_paragraph == entry["문단"]:
                    pp = prev_paragraph
                else:
                    pp = None
            else:
                pp = None

            entry["발언의 배경"] = topic_extractor.extract_topic(
                entry["기사 제목"], entry["문단"], entry["발언의 목적 취지"],
                entry["큰따옴표 발언"], entry["발언자 성명 및 직책"], pp
            )

            row = [entry.get(header, "") for header in workbook_headers]
            rows.append(row)

            prev_title = entry["기사 제목"]
            prev_paragraph = entry["문단"]

            progress_tracker.update_progress(
                i + 1, total_entries,
                "[4단계 중 4단계] 발언의 배경 추출 및 엑셀 파일 저장 중",
                start_time
            )

        return workbook_headers, rows

    def _save_rows_to_excel_and_csv(self, headers, rows, excel_file):
        """행 데이터(headers, rows)를 받아 Excel과 CSV로 저장합니다."""
        # Excel 저장
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "발언 내용 정리"
        sheet.append(headers)
        for row in rows:
            sheet.append(row)
        workbook.save(excel_file)

        # CSV 저장 (동일한 headers/rows)
        csv_file = os.path.splitext(excel_file)[0] + ".csv"
        with open(csv_file, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)