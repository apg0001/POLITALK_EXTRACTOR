import openpyxl
import pandas as pd
import time
import chardet
import traceback
from text_manager import TextProcessor, Merger
from extract_purpose import PurposeExtractor
from extract_topic_summary import TopicExtractor


class FileProcessor:
    """CSV 파일 처리 및 Excel 저장을 담당하는 클래스
    
    이 클래스는 CSV 파일에서 데이터를 추출하고, 병합, 중복 제거 등의 전처리를 수행한 후
    Excel 파일로 저장하는 전체 파이프라인을 관리합니다.
    """
    
    def __init__(self):
        """FileProcessor 초기화
        
        필요한 하위 프로세서들을 초기화합니다:
        - TextProcessor: 텍스트 처리 및 정제
        - PurposeExtractor: 발언 목적 추출
        - TopicExtractor: 주제 및 배경 추출
        """
        self.text_processor = TextProcessor()
        self.purpose_extractor = PurposeExtractor()
        self.topic_extractor = TopicExtractor()
    
    @staticmethod
    def format_remaining_time(remaining_seconds):
        """남은 시간을 00시간 00분 00초 형식으로 변환"""
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

    @staticmethod
    def is_empty(value):
        """값이 None이거나 공백 문자열인지 판별"""
        return value is None or pd.isna(value) or str(value).strip() == ""

    def extract_text_from_csv(self, csv_file, progress_bar, progress_label):
        """CSV 파일에서 텍스트를 추출하고 각 필드를 구분
        
        Args:
            csv_file (str): 처리할 CSV 파일 경로
            progress_bar: GUI 진행률 표시바
            progress_label: GUI 진행률 라벨
            
        Returns:
            list: 추출된 데이터 리스트
        """
        with open(csv_file, 'rb') as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']

        data = pd.read_csv(csv_file, encoding=encoding)
        extracted_data = []
        total_rows = len(data)
        progress_bar['maximum'] = total_rows
        start_time = time.time()

        for i, row in data.iterrows():
            sentences = self.text_processor.split_sentences_by_comma(row['발췌문장'])

            for sentence in sentences:
                _, clean_sentence = self.text_processor.extract_and_clean_quotes(sentence)
                candidate_speakers = self.text_processor.merge_tokens(
                    self.text_processor.extract_speaker(clean_sentence))

                speakers = []

                if len(sentences) == 1:
                    add_flag = True
                else:
                    for name in candidate_speakers:
                        if self.text_processor.is_valid_speaker_by_josa(name, clean_sentence):
                            speakers.append(name)

                    if speakers:
                        add_flag = any(speaker.startswith(row['이름'][0]) for speaker in speakers)
                        for speaker in speakers:
                            if len(speaker) == 3 and speaker != row['이름']:
                                add_flag = False
                    else:
                        add_flag = True

                if not add_flag:
                    continue

                current_data = {
                    "날짜": self.text_processor.to_string(row['일자']),
                    "발언자 성명 및 직책": self.text_processor.to_string(row['이름']),
                    "신문사": self.text_processor.to_string(row['신문사']),
                    "기사 제목": self.text_processor.to_string(row['제목']),
                    "문단": self.text_processor.to_string(row['발췌문단']),
                    "문장": self.text_processor.to_string(row['발췌문장']),
                    "큰따옴표 발언": self.text_processor.extract_quotes(sentence, self.text_processor.to_string(row['이름']))
                }

                if not any(self.is_empty(v) for v in current_data.values()):
                    extracted_data.append(current_data)

            elapsed_time = time.time() - start_time
            time_per_step = elapsed_time / (i + 1)
            remaining_time = time_per_step * (total_rows - (i + 1))
            formatted_remaining_time = self.format_remaining_time(remaining_time)

            progress_bar['value'] = i + 1
            progress_label.config(
                text=f"[4단계 중 1단계] 파일 불러오기 및 큰따옴표 발언 추출 중 : {i + 1}/{total_rows} - 남은 예상 시간: {formatted_remaining_time}")
            progress_bar.update()

        return extracted_data

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
        if not data:
            print("[병합] 저장할 데이터가 없습니다.")
            return

        total_entries = len(data)
        progress_bar['maximum'] = total_entries
        start_time = time.time()
        merged_data = []

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
                    
                    if entry["문단"] != merged_data[-1]["문단"]:
                        merged_data[-1]["문단"] += entry["문단"]
                else:
                    merged_data.append(entry)

                elapsed_time = time.time() - start_time
                if i + 1 > 0:
                    time_per_step = elapsed_time / (i + 1)
                    remaining_time = time_per_step * (total_entries - (i + 1))
                else:
                    remaining_time = 0

                formatted_remaining_time = self.format_remaining_time(remaining_time)
                progress_bar['value'] = i + 1
                progress_label.config(
                    text=f"[4단계 중 2단계] 내용 병합 중: {i + 1}/{total_entries} - 남은 예상 시간: {formatted_remaining_time}")
                progress_bar.update()

        except Exception as e:
            print(f"내용 병합 중 오류 발생: {e}")
            traceback.print_exc()

        return merged_data

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
        if not data:
            print("[중복 제거] 저장할 데이터가 없습니다.")
            return []

        sentence_sets = []
        total_entries = len(data)
        progress_bar['maximum'] = total_entries
        start_time = time.time()
        duplicate_removed_data = []

        try:
            for i, entry in enumerate(data):
                original_sentences = entry["큰따옴표 발언"].split("  ")
                normalized_sentences = [self.text_processor.normalize_text(s) for s in original_sentences]

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
                            if normalized_sentences[idx_new] in existing_normalized[idx_exist] or \
                               self.text_processor.calculate_similarity(normalized_sentences[idx_new], existing_normalized[idx_exist], "min"):
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
                                 self.text_processor.calculate_similarity(normalized_sentences[idx_new], existing_normalized[idx_exist], "max"):
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
                        'normalized': [self.text_processor.normalize_text(s) for s in entry["큰따옴표 발언"].split("  ")]
                    })
                    if len(sentence_sets) > 200:
                        sentence_sets.pop(0)
                        duplicate_removed_data.pop(0)

                elapsed_time = time.time() - start_time
                if i + 1 > 0:
                    time_per_step = elapsed_time / (i + 1)
                    remaining_time = time_per_step * (total_entries - (i + 1))
                else:
                    remaining_time = 0

                formatted_remaining_time = self.format_remaining_time(remaining_time)
                progress_bar['value'] = i + 1
                progress_label.config(
                    text=f"[4단계 중 3단계] 중복 제거 중: {i + 1}/{total_entries} - 남은 예상 시간: {formatted_remaining_time}")
                progress_bar.update()

        except Exception as e:
            print(f"중복 제거 중 오류 발생: {e}")
            traceback.print_exc()

        return duplicate_removed_data

    def save_data_to_excel(self, data, excel_file, progress_bar, progress_label):
        """추출된 데이터를 엑셀 파일로 저장
        
        AI를 사용하여 발언의 목적과 배경을 추출한 후 Excel 파일로 저장합니다.
        
        Args:
            data (list): 저장할 데이터 리스트
            excel_file (str): 저장할 Excel 파일 경로
            progress_bar: GUI 진행률 표시바
            progress_label: GUI 진행률 라벨
        """
        if not data:
            print("[엑셀 파일 저장] 저장할 데이터가 없습니다.")
            return

        try:
            workbook = openpyxl.Workbook()
            sheet = workbook.active
            sheet.title = "발언 내용 정리"

            headers = ["날짜", "발언자 성명 및 직책", "신문사", "기사 제목",
                       "발언의 배경", "문단", "발언의 목적 취지", "큰따옴표 발언"]
            sheet.append(headers)

            total_entries = len(data)
            progress_bar['maximum'] = total_entries
            start_time = time.time()

            prev_paragraph = None
            prev_title = None

            for i, entry in enumerate(data):
                for key, value in entry.items():
                    if value is None:
                        entry[key] = ""

            for i, entry in enumerate(data):
                if prev_title != entry["기사 제목"]:
                    prev_title = entry["기사 제목"]
                
                entry["발언의 목적 취지"] = self.purpose_extractor.extract_purpose(
                    entry["발언자 성명 및 직책"], entry["기사 제목"], entry["문장"], entry["문단"])
                
                if prev_title is not None and prev_paragraph is not None:
                    if prev_title == entry["기사 제목"] and prev_paragraph == entry["문단"]:
                        pp = prev_paragraph
                    else:
                        pp = None
                else:
                    pp = None
                
                entry["발언의 배경"] = self.topic_extractor.extract_topic(
                    entry["기사 제목"], entry["문단"], entry["발언의 목적 취지"], 
                    entry["큰따옴표 발언"], entry["발언자 성명 및 직책"], pp)
                
                row = [entry.get(header, "") for header in headers]
                sheet.append(row)

                prev_title = entry["기사 제목"]
                prev_paragraph = entry["문단"]

                elapsed_time = time.time() - start_time
                if i + 1 > 0:
                    time_per_step = elapsed_time / (i + 1)
                    remaining_time = time_per_step * (total_entries - (i + 1))
                else:
                    remaining_time = 0

                formatted_remaining_time = self.format_remaining_time(remaining_time)
                progress_bar['value'] = i + 1
                progress_label.config(
                    text=f"[4단계 중 4단계] 발언의 배경 추출 및 엑셀 파일 저장 중: {i + 1}/{total_entries} - 남은 예상 시간: {formatted_remaining_time}")
                progress_bar.update()

            workbook.save(excel_file)
            print(f"엑셀 파일이 '{excel_file}'로 저장되었습니다.")
        except Exception as e:
            print(f"발언의 배경 추출 및 엑셀 파일 저장 중 오류 발생: {e}")
            traceback.print_exc()


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