"""유틸리티 함수들을 담당하는 모듈"""

import pandas as pd
from tqdm import tqdm


class TimeFormatter:
    """시간 포맷팅을 담당하는 클래스"""
    
    @staticmethod
    def format_remaining_time(remaining_seconds):
        """남은 시간을 00시간 00분 00초 형식으로 변환
        
        Args:
            remaining_seconds (float): 남은 시간(초)
            
        Returns:
            str: 포맷된 시간 문자열
        """
        if remaining_seconds >= 3600:
            hours = int(remaining_seconds // 3600)
            minutes = int((remaining_seconds % 3600) // 60)
            seconds = int(remaining_seconds % 60)
            return f"{hours}시간 {minutes}분 {seconds}초"
        elif remaining_seconds >= 60:
            minutes = int(remaining_seconds // 60)
            seconds = int(remaining_seconds % 60)
            return f"{minutes}분 {seconds}초"
        else:
            return f"{int(remaining_seconds)}초"


class DataValidator:
    """데이터 검증을 담당하는 클래스"""
    
    @staticmethod
    def is_empty(value):
        """값이 None이거나 공백 문자열인지 판별
        
        Args:
            value: 검증할 값
            
        Returns:
            bool: 비어있으면 True, 아니면 False
        """
        return value is None or pd.isna(value) or str(value).strip() == ""


class ProgressTracker:
    """진행률 추적을 담당하는 클래스"""
    
    def __init__(self, progress_bar, progress_label):
        """ProgressTracker 초기화
        
        Args:
            progress_bar: GUI 진행률 표시바
            progress_label: GUI 진행률 라벨
        """
        self.progress_bar = progress_bar
        self.progress_label = progress_label
        self.time_formatter = TimeFormatter()
        self.tqdm_bar = None
        self.total = None
    
    def initialize_tqdm(self, total, description):
        """tqdm 프로그레스바 초기화
        
        Args:
            total (int): 전체 항목 수
            description (str): 프로그레스바 설명
        """
        self.total = total
        self.tqdm_bar = tqdm(total=total, desc=description, unit='item', ncols=100)
    
    def close_tqdm(self):
        """tqdm 프로그레스바 종료"""
        if self.tqdm_bar is not None:
            self.tqdm_bar.close()
            self.tqdm_bar = None
    
    def update_progress(self, current, total, stage, start_time):
        """진행률 업데이트
        
        Args:
            current (int): 현재 진행 단계
            total (int): 전체 단계
            stage (str): 현재 단계 설명
            start_time (float): 시작 시간
        """
        import time
        
        elapsed_time = time.time() - start_time
        if current > 0:
            time_per_step = elapsed_time / current
            remaining_time = time_per_step * (total - current)
        else:
            remaining_time = 0

        formatted_remaining_time = self.time_formatter.format_remaining_time(remaining_time)
        
        self.progress_bar['value'] = current
        self.progress_label.config(text=f"{stage}: {current}/{total} - 남은 예상 시간: {formatted_remaining_time}")
        self.progress_bar.update()
        
        # tqdm 업데이트
        if self.tqdm_bar is not None:
            increment = current - (self.tqdm_bar.n if self.tqdm_bar else 0)
            if increment > 0:
                self.tqdm_bar.update(increment)
                self.tqdm_bar.set_description(f"{stage}: {formatted_remaining_time} 남음")