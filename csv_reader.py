"""CSV 파일 읽기 전용 클래스"""

import pandas as pd
import chardet
from utils import DataValidator, ProgressTracker


class CSVReader:
    """CSV 파일 읽기를 담당하는 클래스"""
    
    def __init__(self):
        """CSVReader 초기화"""
        self.validator = DataValidator()
    
    def read_csv_file(self, csv_file):
        """CSV 파일을 읽어서 DataFrame으로 반환
        
        Args:
            csv_file (str): CSV 파일 경로
            
        Returns:
            pd.DataFrame: 읽은 데이터
        """
        with open(csv_file, 'rb') as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']

        return pd.read_csv(csv_file, encoding=encoding)
    
    def extract_text_from_csv(self, csv_file, progress_tracker):
        """CSV 파일에서 텍스트를 추출하고 각 필드를 구분
        
        Args:
            csv_file (str): 처리할 CSV 파일 경로
            progress_tracker (ProgressTracker): 진행률 추적기
            
        Returns:
            list: 추출된 데이터 리스트
        """
        from text_manager import TextProcessor
        
        data = self.read_csv_file(csv_file)
        text_processor = TextProcessor()
        extracted_data = []
        total_rows = len(data)
        progress_tracker.progress_bar['maximum'] = total_rows
        
        import time
        start_time = time.time()

        for i, row in data.iterrows():
            sentences = text_processor.split_sentences_by_comma(row['발췌문장'])

            for sentence in sentences:
                _, clean_sentence = text_processor.extract_and_clean_quotes(sentence)
                candidate_speakers = text_processor.merge_tokens(
                    text_processor.extract_speaker(clean_sentence))

                speakers = []

                # 단문이면 바로 추가
                if len(sentences) == 1:
                    add_flag = True
                else:
                    # 조사 판별: '은', '는'만 통과
                    for name in candidate_speakers:
                        if text_processor.is_valid_speaker_by_josa(name, clean_sentence):
                            speakers.append(name)

                    # 성이 다른 경우 + 중문일 경우 제거
                    if speakers:
                        add_flag = any(speaker.startswith(row['이름'][0]) for speaker in speakers)
                        for speaker in speakers:
                            if len(speaker) == 3 and speaker != row['이름']:
                                add_flag = False
                        if not add_flag:
                            continue
                    else:
                        add_flag = True  # 주어 없으면 그대로 추가

                if not add_flag:
                    continue

                current_data = {
                    "날짜": text_processor.to_string(row['일자']),
                    "발언자 성명 및 직책": text_processor.to_string(row['이름']),
                    "신문사": text_processor.to_string(row['신문사']),
                    "기사 제목": text_processor.to_string(row['제목']),
                    "문단": text_processor.to_string(row['발췌문단']),
                    "문장": text_processor.to_string(row['발췌문장']),
                    "큰따옴표 발언": text_processor.extract_quotes(sentence, text_processor.to_string(row['이름']))
                }

                if not any(self.validator.is_empty(v) for v in current_data.values()):
                    extracted_data.append(current_data)

            progress_tracker.update_progress(
                i + 1, total_rows, 
                "[4단계 중 1단계] 파일 불러오기 및 큰따옴표 발언 추출 중", 
                start_time
            )

        return extracted_data