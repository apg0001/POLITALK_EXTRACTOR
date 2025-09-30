import os
import pandas as pd
import time
import chardet
import traceback
from tqdm import tqdm
import openpyxl
import datetime
from file_manager import FileProcessor
from text_manager import TextProcessor
from extract_purpose import PurposeExtractor
from extract_topic_summary import TopicExtractor


class BatchFileProcessor:
    """여러 파일을 일괄 처리하는 클래스"""
    
    def __init__(self):
        self.file_processor = FileProcessor()
        self.text_processor = TextProcessor()
        self.purpose_extractor = PurposeExtractor()
        self.topic_extractor = TopicExtractor()
        self.temp_title = []

    def format_remaining_time(self, remaining_seconds):
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

    def is_empty(self, value):
        """값이 None이거나 공백 문자열인지 판별"""
        return value is None or pd.isna(value) or str(value).strip() == ""

    def extract_text_from_csv(self, csv_file):
        """CSV 파일에서 텍스트를 추출하고 각 필드를 구분"""
        with open(csv_file, 'rb') as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']

        data = pd.read_csv(csv_file, encoding=encoding)
        extracted_data = []
        total_rows = len(data)
        start_time = time.time()

        for i, row in data.iterrows():
            sentences = self.text_processor.split_sentences_by_comma(row['발췌문장'])

            for sentence in sentences:
                _, clean_sentence = self.text_processor.extract_and_clean_quotes(sentence)
                candidate_speakers = self.text_processor.merge_tokens(
                    self.text_processor.extract_speaker(clean_sentence))

                speakers = []

                # 단문이면 바로 추가
                if len(sentences) == 1:
                    add_flag = True
                else:
                    # 조사 판별: '은', '는'만 통과
                    for name in candidate_speakers:
                        if self.text_processor.is_valid_speaker_by_josa(name, clean_sentence):
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
                    "날짜": self.text_processor.to_string(row['일자']),
                    "발언자 성명 및 직책": self.text_processor.to_string(row['이름']),
                    "신문사": self.text_processor.to_string(row['신문사']),
                    "기사 제목": self.text_processor.to_string(row['제목']),
                    "문단": self.text_processor.to_string(row['발췌문단']),
                    "문장": self.text_processor.to_string(row['발췌문장']),
                    "발언": self.text_processor.extract_quotes(sentence, self.text_processor.to_string(row['이름']))
                }

                if not any(self.is_empty(v) for v in current_data.values()):
                    extracted_data.append(current_data)

            # 남은 예상 시간 계산
            elapsed_time = time.time() - start_time
            if i + 1 > 0:
                time_per_step = elapsed_time / (i + 1)
                remaining_time = time_per_step * (total_rows - (i + 1))
            else:
                remaining_time = 0

            formatted_remaining_time = self.format_remaining_time(remaining_time)

        return extracted_data

    def merge_data(self, data):
        """내용 병합"""
        if not data:
            print("[병합] 저장할 데이터가 없습니다.")
            return []

        merged_data = []
        total_entries = len(data)
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
                    (self.text_processor.Merger.check_cases(entry["문장"], entry["문단"], data[i-1]["발언"].split("  "))):

                    if entry["발언"] not in merged_data[-1]["발언"]:
                        merged_data[-1]["발언"] += ("  " + entry["발언"])
                    
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

        except Exception as e:
            print(f"내용 병합 중 오류 발생: {e}")
            traceback.print_exc()

        return merged_data

    def remove_duplicates(self, data):
        """중복 제거"""
        if not data:
            print("[중복 제거] 저장할 데이터가 없습니다.")
            return []

        sentence_sets = []
        duplicate_removed_data = []
        total_entries = len(data)
        start_time = time.time()

        try:
            for i, entry in enumerate(data):
                original_sentences = entry["발언"].split("  ")
                normalized_sentences = [self.text_processor.normalize_text(s) for s in original_sentences]

                j = 0
                while j < len(sentence_sets):
                    existing_entry = duplicate_removed_data[j]
                    existing_sentences = existing_entry['발언'].split("  ")
                    existing_normalized = sentence_sets[j]['normalized']
                    
                    idx_new = 0
                    while idx_new < len(normalized_sentences):
                        matched = False
                        idx_exist = 0
                        while idx_exist < len(existing_normalized):
                            # 한 문장이 다른 문장을 완전히 포함하는 경우
                            if normalized_sentences[idx_new] in existing_normalized[idx_exist] or \
                               self.text_processor.calculate_similarity(normalized_sentences[idx_new], existing_normalized[idx_exist], "min"):
                                del existing_sentences[idx_exist]
                                del existing_normalized[idx_exist]
                                
                                if existing_sentences:
                                    existing_entry["발언"] = "  ".join(existing_sentences)
                                    sentence_sets[j] = {
                                        'original': existing_entry["발언"],
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
                                        existing_entry["발언"] = "  ".join(existing_sentences)
                                        sentence_sets[j] = {
                                            'original': existing_entry["발언"],
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
                    entry["발언"] = "  ".join(original_sentences)
                    duplicate_removed_data.append(entry)
                    sentence_sets.append({
                        'original': entry["발언"],
                        'normalized': [self.text_processor.normalize_text(s) for s in entry["발언"].split("  ")]
                    })
                    if len(sentence_sets) > 200:
                        sentence_sets.pop(0)
                        duplicate_removed_data.pop(0)

                elapsed_time = time.time() - start_time
                time_per_step = elapsed_time / (i + 1)
                remaining_time = time_per_step * (total_entries - (i + 1))
                formatted_remaining_time = self.format_remaining_time(remaining_time)

        except Exception as e:
            print(f"중복 제거 중 오류 발생: {e}")
            traceback.print_exc()

        return duplicate_removed_data

    def save_data_to_excel(self, data, excel_file):
        """추출된 데이터를 엑셀 파일로 저장"""
        if not data:
            print("[엑셀 파일 저장] 저장할 데이터가 없습니다.")
            return

        try:
            workbook = openpyxl.Workbook()
            sheet = workbook.active
            sheet.title = "발언 내용 정리"

            headers = ["날짜", "발언자 성명 및 직책", "신문사", "기사 제목",
                       "발언의 배경", "문단", "발언의 목적 취지", "발언"]
            sheet.append(headers)

            total_entries = len(data)
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
                    entry["발언"], entry["발언자 성명 및 직책"], pp)

                row = [entry.get(header, "") for header in headers]
                sheet.append(row)

                prev_title = entry["기사 제목"]
                prev_paragraph = entry["문단"]

                elapsed_time = time.time() - start_time
                time_per_step = elapsed_time / (i + 1)
                remaining_time = time_per_step * (total_entries - (i + 1))
                formatted_remaining_time = self.format_remaining_time(remaining_time)

                print(f"[4단계 중 4단계] 발언의 배경 추출 및 엑셀 파일 저장 중: {i + 1}/{total_entries} - 남은 예상 시간: {formatted_remaining_time}")

            workbook.save(excel_file)
            print(f"엑셀 파일이 '{excel_file}'로 저장되었습니다.")
        except Exception as e:
            print(f"발언의 배경 추출 및 엑셀 파일 저장 중 오류 발생: {e}")
            traceback.print_exc()

    def process_file(self, csv_file, output_excel_file, output_csv_file):
        """주어진 CSV 파일에 대해 텍스트 추출, 데이터 병합, 중복 제거 후 엑셀로 저장"""
        try:
            # 1. CSV에서 텍스트 추출
            extracted_data = self.extract_text_from_csv(csv_file)

            # 2. 데이터 병합
            merged_data = self.merge_data(extracted_data)

            # 3. 중복 제거
            cleaned_data = self.remove_duplicates(merged_data)

            # 4. 엑셀로 저장
            self.save_data_to_excel(cleaned_data, output_excel_file)

            print(f"처리 완료: {csv_file} -> {output_excel_file}")
        except Exception as e:
            print(f"{csv_file} 처리 중 오류 발생: {e}")

    def get_csv_files_from_directory(self, directory_path):
        """주어진 디렉토리 내 모든 CSV 파일을 찾는 함수"""
        csv_files = []
        for file_name in os.listdir(directory_path):
            if file_name.endswith('.csv'):
                csv_files.append(os.path.join(directory_path, file_name))
        return csv_files

    def process_multiple_files(self, directory_path, output_dir):
        """디렉토리 내 모든 CSV 파일을 처리하는 함수"""
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
    return processor.extract_text_from_csv(csv_file)

def merge_data(data):
    processor = BatchFileProcessor()
    return processor.merge_data(data)

def remove_duplicates(data):
    processor = BatchFileProcessor()
    return processor.remove_duplicates(data)

def save_data_to_excel(data, excel_file):
    processor = BatchFileProcessor()
    return processor.save_data_to_excel(data, excel_file)

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