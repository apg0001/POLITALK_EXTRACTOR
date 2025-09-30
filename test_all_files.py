import os
import pandas as pd
import time
import chardet
import traceback
from tqdm import tqdm
import openpyxl
import datetime
from file_manager import FileProcessor
from utils import TimeFormatter, DataValidator


class BatchFileProcessor:
    """여러 파일을 일괄 처리하는 클래스
    
    FileProcessor를 사용하여 여러 CSV 파일을 일괄 처리합니다.
    """
    
    def __init__(self):
        """BatchFileProcessor 초기화"""
        self.file_processor = FileProcessor()
        self.time_formatter = TimeFormatter()
        self.validator = DataValidator()
        self.temp_title = []

    def format_remaining_time(self, remaining_seconds):
        """남은 시간을 00시간 00분 00초 형식으로 변환"""
        return self.time_formatter.format_remaining_time(remaining_seconds)

    def is_empty(self, value):
        """값이 None이거나 공백 문자열인지 판별"""
        return self.validator.is_empty(value)

    def process_file(self, csv_file, output_excel_file, output_csv_file):
        """주어진 CSV 파일에 대해 텍스트 추출, 데이터 병합, 중복 제거 후 엑셀로 저장
        
        Args:
            csv_file (str): 처리할 CSV 파일 경로
            output_excel_file (str): 출력할 Excel 파일 경로
            output_csv_file (str): 출력할 CSV 파일 경로 (사용하지 않음)
        """
        try:
            # FileProcessor를 사용하여 처리
            # GUI가 없으므로 더미 진행률 추적기 생성
            from utils import ProgressTracker
            import tkinter as tk
            
            # 더미 GUI 컴포넌트 생성
            root = tk.Tk()
            root.withdraw()  # 창을 숨김
            
            progress_bar = tk.ttk.Progressbar(root, mode='determinate')
            progress_label = tk.Label(root)
            
            progress_tracker = ProgressTracker(progress_bar, progress_label)
            
        # 1. CSV에서 텍스트 추출
            extracted_data = self.file_processor.csv_reader.extract_text_from_csv(csv_file, progress_tracker)

        # 2. 데이터 병합
            merged_data = self.file_processor.data_merger.merge_data(extracted_data, progress_tracker)

        # 3. 중복 제거
            cleaned_data = self.file_processor.duplicate_remover.remove_duplicates(merged_data, progress_tracker)

        # 4. 엑셀로 저장
            self.file_processor.excel_writer.save_data_to_excel(cleaned_data, output_excel_file, progress_tracker)

            
            root.destroy()
        except Exception as e:
            print(f"{csv_file} 처리 중 오류 발생: {e}")
        print(f"처리 완료: {csv_file} -> {output_excel_file}")

    def get_csv_files_from_directory(self, directory_path):
        """주어진 디렉토리 내 모든 CSV 파일을 찾는 함수
        
        Args:
            directory_path (str): 검색할 디렉토리 경로

        Returns:
            list: CSV 파일 경로 리스트
        """
        csv_files = []
        for file_name in os.listdir(directory_path):
            if file_name.endswith('.csv'):
                csv_files.append(os.path.join(directory_path, file_name))
        return csv_files

    def process_multiple_files(self, directory_path, output_dir):
        """디렉토리 내 모든 CSV 파일을 처리하는 함수
        
        Args:
            directory_path (str): CSV 파일들이 있는 디렉토리 경로
            output_dir (str): 출력 파일들이 저장될 디렉토리 경로
        """
        csv_files = self.get_csv_files_from_directory(directory_path)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for csv_file in tqdm(csv_files, desc="파일 처리 중", unit="file"):
            formatted_date = datetime.datetime.now().strftime('%y%m%d')
            output_excel_file = os.path.join(
                output_dir, f"{os.path.basename(csv_file).replace('.csv', f'_AI변환_{formatted_date}.xlsx')}")
            output_csv_file = os.path.join(
                output_dir, f"{os.path.basename(csv_file).replace('.csv', f'_AI변환_{formatted_date}.csv')}")

            self.process_file(csv_file, output_excel_file, output_csv_file)


# 하위 호환성을 위한 함수들
def format_remaining_time(remaining_seconds):
    processor = BatchFileProcessor()
    return processor.format_remaining_time(remaining_seconds)

def is_empty(value):
    processor = BatchFileProcessor()
    return processor.is_empty(value)

def extract_text_from_csv(csv_file):
    processor = BatchFileProcessor()
    return processor.file_processor.csv_reader.extract_text_from_csv(csv_file, None)

def merge_data(data):
    processor = BatchFileProcessor()
    return processor.file_processor.data_merger.merge_data(data, None)

def remove_duplicates(data):
    processor = BatchFileProcessor()
    return processor.file_processor.duplicate_remover.remove_duplicates(data, None)

def save_data_to_excel(data, excel_file):
    processor = BatchFileProcessor()
    return processor.file_processor.excel_writer.save_data_to_excel(data, excel_file, None)

def process_file(csv_file, output_excel_file, output_csv_file):
    processor = BatchFileProcessor()
    return processor.process_file(csv_file, output_excel_file, output_csv_file)

def get_csv_files_from_directory(directory_path):
    processor = BatchFileProcessor()
    return processor.get_csv_files_from_directory(directory_path)

def process_multiple_files(directory_path, output_dir):
    processor = BatchFileProcessor()
    return processor.process_multiple_files(directory_path, output_dir)


# 테스트용 코드
if __name__ == "__main__":
    formatted_date = datetime.datetime.now().strftime('%y%m%d')
    directory_path = "C:/Users/apg00/Downloads/Input Sample Files"
    output_dir = f"C:/Users/apg00/Downloads/output_{formatted_date}"

    processor = BatchFileProcessor()
    processor.process_multiple_files(directory_path, output_dir)