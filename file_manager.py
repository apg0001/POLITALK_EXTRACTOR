import openpyxl
import pandas as pd
import time
import chardet
import traceback
from text_manager import Merger
from extract_purpose import PurposeExtractor
from extract_topic_summary import TopicExtractor
from utils import TimeFormatter, DataValidator, ProgressTracker
from csv_reader import CSVReader
from data_merger import DataMerger
from duplicate_remover import DuplicateRemover
from excel_writer import ExcelWriter


class FileProcessor:
    """CSV 파일 처리 및 Excel 저장을 담당하는 클래스
    
    이 클래스는 CSV 파일에서 데이터를 추출하고, 병합, 중복 제거 등의 전처리를 수행한 후
    Excel 파일로 저장하는 전체 파이프라인을 관리합니다.
    """
    
    def __init__(self):
        """FileProcessor 초기화
        
        필요한 하위 프로세서들을 초기화합니다:
        - CSVReader: CSV 파일 읽기
        - DataMerger: 데이터 병합
        - DuplicateRemover: 중복 제거
        - ExcelWriter: Excel 파일 저장
        """
        self.csv_reader = CSVReader()
        self.data_merger = DataMerger()
        self.duplicate_remover = DuplicateRemover()
        self.excel_writer = ExcelWriter()
    
    @staticmethod
    def format_remaining_time(remaining_seconds):
        """남은 시간을 00시간 00분 00초 형식으로 변환"""
        return TimeFormatter.format_remaining_time(remaining_seconds)

    @staticmethod
    def is_empty(value):
        """값이 None이거나 공백 문자열인지 판별"""
        return DataValidator.is_empty(value)

    def extract_text_from_csv(self, csv_file, progress_bar, progress_label):
        """CSV 파일에서 텍스트를 추출하고 각 필드를 구분
        
        Args:
            csv_file (str): 처리할 CSV 파일 경로
            progress_bar: GUI 진행률 표시바
            progress_label: GUI 진행률 라벨
            
        Returns:
            list: 추출된 데이터 리스트
        """
        progress_tracker = ProgressTracker(progress_bar, progress_label)
        return self.csv_reader.extract_text_from_csv(csv_file, progress_tracker)

    def merge_data(self, data, progress_bar, progress_label):
        """내용 병합
        
        같은 기사, 같은 발언자, 같은 날짜, 같은 신문사의 연속된 발언을 병합합니다.
        접속사나 문맥을 고려하여 관련된 발언들을 하나로 합칩니다.
        
        Args:
            data (list): 병합할 데이터 리스트
            progress_bar: GUI 진행률 표시바
            progress_label: GUI 진행률 라벨
            
        Returns:
            list: 병합된 데이터 리스트
        """
        progress_tracker = ProgressTracker(progress_bar, progress_label)
        return self.data_merger.merge_data(data, progress_tracker)

    def remove_duplicates(self, data, progress_bar, progress_label):
        """중복 발언 제거
        
        유사도 검사를 통해 중복된 발언문을 제거합니다.
        최근 200개 문장과 비교하여 중복을 찾아 제거합니다.
        
        Args:
            data (list): 중복 제거할 데이터 리스트
            progress_bar: GUI 진행률 표시바
            progress_label: GUI 진행률 라벨
            
        Returns:
            list: 중복이 제거된 데이터 리스트
        """
        progress_tracker = ProgressTracker(progress_bar, progress_label)
        return self.duplicate_remover.remove_duplicates(data, progress_tracker)

    def save_data_to_excel(self, data, excel_file, progress_bar, progress_label):
        """추출된 데이터를 엑셀 파일로 저장
        
        AI를 사용하여 발언의 목적과 배경을 추출한 후 Excel 파일로 저장합니다.
        
        Args:
            data (list): 저장할 데이터 리스트
            excel_file (str): 저장할 Excel 파일 경로
            progress_bar: GUI 진행률 표시바
            progress_label: GUI 진행률 라벨
        """
        progress_tracker = ProgressTracker(progress_bar, progress_label)
        return self.excel_writer.save_data_to_excel(data, excel_file, progress_tracker)


# 하위 호환성을 위한 함수들
def format_remaining_time(remaining_seconds):
    return FileProcessor.format_remaining_time(remaining_seconds)

def is_empty(value):
    return FileProcessor.is_empty(value)

def extract_text_from_csv(csv_file, progress_bar, progress_label):
    processor = FileProcessor()
    return processor.extract_text_from_csv(csv_file, progress_bar, progress_label)

def merge_data(data, progress_bar, progress_label):
    processor = FileProcessor()
    return processor.merge_data(data, progress_bar, progress_label)

def remove_duplicates(data, progress_bar, progress_label):
    processor = FileProcessor()
    return processor.remove_duplicates(data, progress_bar, progress_label)

def save_data_to_excel(data, excel_file, progress_bar, progress_label):
    processor = FileProcessor()
    return processor.save_data_to_excel(data, excel_file, progress_bar, progress_label)