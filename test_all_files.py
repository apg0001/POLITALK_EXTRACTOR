import os
import pandas as pd
import time
import chardet
import traceback
from tqdm import tqdm
import openpyxl
from text_manager import *
from extract_purpose import extract_purpose
# from extract_purpose_summary import extract_purpose
# from extract_topic_title_and_summary import extract_topic
# from modify_title import Modifier, test
# from extract_topic_logic import extract_topic
from extract_topic_summary import TopicExtractor
import datetime

temp_title = []


def format_remaining_time(remaining_seconds):
    """남은 시간을 00시간 00분 00초 형식으로 변환"""
    if remaining_seconds >= 3600:  # 1시간 이상 남은 경우
        hours = int(remaining_seconds // 3600)
        minutes = int((remaining_seconds % 3600) // 60)
        seconds = int(remaining_seconds % 60)
        return f"{hours}시간 {minutes}분 {seconds}초"
    elif remaining_seconds >= 60:  # 1분 이상 남은 경우
        minutes = int(remaining_seconds // 60)
        seconds = int(remaining_seconds % 60)
        return f"{minutes}분 {seconds}초"
    else:  # 1분 미만 남은 경우
        return f"{int(remaining_seconds)}초"


def is_empty(value):
    """값이 None이거나 공백 문자열인지 판별"""
    return value is None or pd.isna(value) or str(value).strip() == ""


def extract_text_from_csv(csv_file):
    """CSV 파일에서 텍스트를 추출하고 각 필드를 구분."""
    with open(csv_file, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']

    data = pd.read_csv(csv_file, encoding=encoding)
    extracted_data = []
    total_rows = len(data)
    start_time = time.time()  # 작업 시작 시간 기록

    for i, row in data.iterrows():
        sentences = split_sentences_by_comma(row['발췌문장'])

        for sentence in sentences:
            _, clean_sentence = extract_and_clean_quotes(sentence)
            candidate_speakers = merge_tokens(extract_speaker(clean_sentence))
            # print(candidate_speakers)

            speakers = []

            # 단문이면 바로 추가
            if len(sentences) == 1:
                # print(f"[단문] 문장 추가: {sentence}")
                add_flag = True
            else:
                # 조사 판별: '은', '는'만 통과
                for name in candidate_speakers:
                    if is_valid_speaker_by_josa(name, clean_sentence):
                        speakers.append(name)

                # 성이 다른 경우 + 중문일 경우 제거
                if speakers:
                    # print(f"발언자: {speakers}\n문장: {sentence}")
                    add_flag = any(speaker.startswith(
                        row['이름'][0]) for speaker in speakers)
                    for speaker in speakers:
                        if len(speaker) == 3 and speaker != row['이름']:
                            add_flag = False
                            print(f"[추가 안함]이름 완전 불일치: {speaker}")
                    if not add_flag:
                        continue
                        # print(f"성 불일치: {sentence}")
                else:
                    # print(f"발언자 없음 or 조사 불일치: {sentence}")
                    add_flag = True  # 주어 없으면 그대로 추가

            if not add_flag:
                continue

            current_data = {
                "날짜": to_string(row['일자']),
                "발언자 성명 및 직책": to_string(row['이름']),
                "신문사": to_string(row['신문사']),
                "기사 제목": to_string(row['제목']),
                "문단": to_string(row['발췌문단']),
                # "문장": sentence,
                "문장": to_string(row['발췌문장']),
                "큰따옴표 발언": extract_quotes(sentence, to_string(row['이름']))
            }

            if not any(is_empty(v) for v in current_data.values()):
                extracted_data.append(current_data)

        # 남은 예상 시간 계산
        elapsed_time = time.time() - start_time
        if i + 1 > 0:
            time_per_step = elapsed_time / (i + 1)
            remaining_time = time_per_step * (total_rows - (i + 1))
        else:
            remaining_time = 0

        # 남은 시간을 형식에 맞춰 변환하여 표시
        formatted_remaining_time = format_remaining_time(remaining_time)

        # print(f"[4단계 중 1단계] 파일 불러오기 및 큰따옴표 발언 추출 중 : {i + 1}/{total_rows} - 남은 예상 시간: {formatted_remaining_time}")

    return extracted_data


def merge_data(data):
    """내용 병합"""
    if not data:
        print("[병합] 저장할 데이터가 없습니다.")
        return []

    prev_title = None
    merged_data = []
    total_entries = len(data)
    start_time = time.time()

    merged_data = []

    try:
        for i, entry in enumerate(data):
            name = entry["발언자 성명 및 직책"]
            keywords = [name, f"{name[0]} "]

            # if prev_title is None:
            if not merged_data:
                # 첫 번째 데이터는 그대로 추가
                merged_data.append(entry)

            # 유형 3-4
            # elif entry["문장"].startswith(("이에", "이에 대해", "이를 두고")):
            #     entry["문장"] = data[i-1]["문장"]
            #     merged_data.append(entry)

            # elif prev_title == entry["기사 제목"] and filter_sentences_by_name(entry["문장"], keywords):
            elif (merged_data[-1]["기사 제목"] == entry["기사 제목"]) and \
                (merged_data[-1]["발언자 성명 및 직책"] == entry["발언자 성명 및 직책"]) and \
                (merged_data[-1]["날짜"] == entry["날짜"]) and \
                (merged_data[-1]["신문사"] == entry["신문사"]) and \
                    (Merger.check_cases(entry["문장"], entry["문단"], data[i-1]["큰따옴표 발언"].split("  "))):

                # merged_data[-1]['문장'] += entry['문장']
                # 병합 조건이면 가장 마지막 데이터에 큰따옴표 발언 추가ㅎ
                if entry["큰따옴표 발언"] not in merged_data[-1]["큰따옴표 발언"]:
                    merged_data[-1]["큰따옴표 발언"] += ("  " + entry["큰따옴표 발언"])
                    # print("merged! : " + entry["큰따옴표 발언"] + " : " + entry["날짜"])
                # 병합할 때 문단이 다르다면 문단도 합치기
                if entry["문단"] != merged_data[-1]["문단"]:
                    merged_data[-1]["문단"] += entry["문단"]
                    # print("merged! : " + entry["큰따옴표 발언"] + " : " + entry["날짜"])
            else:
                # 병합 조건이 아니면 그대로 추가
                merged_data.append(entry)

            # 남은 예상 시간 계산
            elapsed_time = time.time() - start_time
            if i + 1 > 0:
                time_per_step = elapsed_time / (i + 1)
                remaining_time = time_per_step * (total_entries - (i + 1))
            else:
                remaining_time = 0

            # 남은 시간을 형식에 맞춰 변환하여 표시
            formatted_remaining_time = format_remaining_time(remaining_time)

            # print(f"[4단계 중 2단계] 내용 병합 중: {i + 1}/{total_entries} - 남은 예상 시간: {formatted_remaining_time}")

    except Exception as e:
        print(f"내용 병합 중 오류 발생: {e}")
        traceback.print_exc()  # 자세한 오류 정보를 출력

    return merged_data


def remove_duplicates(data):
    """중복 제거"""
    if not data:
        print("[중복 제거] 저장할 데이터가 없습니다.")
        return []

    sentence_sets = []  # 최근 200개 문장 유지
    duplicate_removed_data = []
    total_entries = len(data)
    start_time = time.time()

    try:
        for i, entry in enumerate(data):
            original_sentences = entry["큰따옴표 발언"].split("  ")
            normalized_sentences = [normalize_text(
                s) for s in original_sentences]

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
                        # 한 문장이 다른 문장을 완전히 포함하는 경우
                        if normalized_sentences[idx_new] in existing_normalized[idx_exist] or calculate_similarity(normalized_sentences[idx_new], existing_normalized[idx_exist], "min"):
                            del existing_sentences[idx_exist]
                            del existing_normalized[idx_exist]
                            # ✅ 기존 entry도 업데이트
                            if existing_sentences:
                                existing_entry["큰따옴표 발언"] = "  ".join(
                                    existing_sentences)
                                sentence_sets[j] = {
                                    'original': existing_entry["큰따옴표 발언"],
                                    'normalized': existing_normalized
                                }
                                continue  # 기존 문장 삭제 → idx_exist는 유지
                            else:
                                # ✅ 기존 entry에 남은 문장 없으면 entry 자체 삭제
                                # print(f"[{i}] 기존 entry {j} 삭제 (모든 문장 제거됨)")
                                del duplicate_removed_data[j]
                                del sentence_sets[j]
                                j -= 1  # entry 삭제했으니 인덱스 보정
                                break  # 현재 비교 종료
                        elif existing_normalized[idx_exist] in normalized_sentences[idx_new]:
                            del original_sentences[idx_new]
                            del normalized_sentences[idx_new]
                            matched = True
                            break  # 삭제 후 현재 idx_new로 계속
                        elif normalized_sentences[idx_new] == existing_normalized[idx_exist] or calculate_similarity(normalized_sentences[idx_new], existing_normalized[idx_exist], "max"):
                            if len(normalized_sentences) <= len(existing_normalized):
                                del original_sentences[idx_new]
                                del normalized_sentences[idx_new]
                                matched = True
                                break
                            else:
                                del existing_sentences[idx_exist]
                                del existing_normalized[idx_exist]
                                if existing_sentences:
                                    existing_entry["큰따옴표 발언"] = "  ".join(
                                        existing_sentences)
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
                    'normalized': [normalize_text(s) for s in entry["큰따옴표 발언"].split("  ")]
                })
                if len(sentence_sets) > 200:
                    sentence_sets.pop(0)
                    duplicate_removed_data.pop(0)

            elapsed_time = time.time() - start_time
            time_per_step = elapsed_time / (i + 1)
            remaining_time = time_per_step * (total_entries - (i + 1))
            formatted_remaining_time = format_remaining_time(remaining_time)

            # print(f"[4단계 중 3단계] 중복 제거 중: {i + 1}/{total_entries} - 남은 예상 시간: {formatted_remaining_time}")

    except Exception as e:
        print(f"중복 제거 중 오류 발생: {e}")
        traceback.print_exc()

    return duplicate_removed_data


def save_data_to_excel(data, excel_file):
    """추출된 데이터를 엑셀 파일로 저장."""
    if not data:
        print("[엑셀 파일 저장] 저장할 데이터가 없습니다.")
        return

    try:
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "발언 내용 정리"

        headers = ["날짜", "발언자 성명 및 직책", "신문사", "기사 제목",
                   "주제", "문단", "발언의 목적 배경 취지", "큰따옴표 발언"]
        sheet.append(headers)

        total_entries = len(data)
        start_time = time.time()

        prev_purpose = None
        prev_topic = None
        prev_title = None

        for i, entry in enumerate(data):
            for key, value in entry.items():
                if value is None:
                    entry[key] = ""

        for i, entry in enumerate(data):
            if prev_title != entry["기사 제목"]:
                prev_title = entry["기사 제목"]
                prev_topic = None
            entry["발언의 목적 배경 취지"] = prev_purpose = extract_purpose(
                entry["발언자 성명 및 직책"], entry["기사 제목"], entry["문장"], entry["문단"], prev_purpose)
            # def extract_topic(title, body, purpose, name, prev_topic):
            # entry["주제"] = extract_topic(entry["기사 제목"], entry["큰따옴표 발언"], entry["발언자 성명 및 직책"])
            # entry["주제"] = extract_topic(entry["기사 제목"], entry["큰따옴표 발언"], entry["발언의 목적 배경 취지"], entry["발언자 성명 및 직책"], prev_topic)
            # entry["주제"] = "test"
            # entry["주제"] = Modifier.modify_title(entry["기사 제목"])
            # entry["주제"] = test(entry["기사 제목"])
            # entry["주제"] = extract_topic(body = entry["문단"], name = entry["발언자 성명 및 직책"])
            # entry["주제"] = extract_topic(text = entry["문단"])
            extractor = TopicExtractor()
            entry["주제"] = extractor.extract_topic(entry["기사 제목"], entry["문단"], entry["발언의 목적 배경 취지"], entry["큰따옴표 발언"], entry["발언자 성명 및 직책"])

            # if (entry["주제"] == Modifier.normalize_text(entry["기사 제목"])):
                # temp_title.append(entry["기사 제목"])

            row = [entry.get(header, "") for header in headers]
            sheet.append(row)

            prev_topic = entry["주제"]
            prev_purpose = entry["발언의 목적 배경 취지"]

            elapsed_time = time.time() - start_time
            time_per_step = elapsed_time / (i + 1)
            remaining_time = time_per_step * (total_entries - (i + 1))
            formatted_remaining_time = format_remaining_time(remaining_time)

            print(
                f"[4단계 중 4단계] 주제 추출 및 엑셀 파일 저장 중: {i + 1}/{total_entries} - 남은 예상 시간: {formatted_remaining_time}")

        workbook.save(excel_file)
        print(f"엑셀 파일이 '{excel_file}'로 저장되었습니다.")
    except Exception as e:
        print(f"주제 추출 및 엑셀 파일 저장 중 오류 발생: {e}")
        traceback.print_exc()  # 자세한 오류 정보를 출력


def save_data_to_csv(data, csv_file):
    """추출된 데이터를 CSV 파일로 저장."""
    if not data:
        print("[CSV 파일 저장] 저장할 데이터가 없습니다.")
        return

    try:
        total_entries = len(data)
        start_time = time.time()

        prev_purpose = None
        prev_topic = None
        prev_title = None

        # None 값을 빈 문자열로 변환
        for entry in data:
            for key, value in entry.items():
                if value is None:
                    entry[key] = ""

        for i, entry in enumerate(data):
            if prev_title != entry["기사 제목"]:
                prev_title = entry["기사 제목"]
                prev_topic = None
            entry["발언의 목적 배경 취지"] = prev_purpose = extract_purpose(
                entry["발언자 성명 및 직책"], entry["기사 제목"], entry["문장"], entry["문단"], prev_purpose)
            # entry["주제"] = extract_topic(entry["기사 제목"], entry["큰따옴표 발언"], entry["발언자 성명 및 직책"])

            # entry["주제"] = extract_topic(entry["기사 제목"], entry["큰따옴표 발언"], entry["발언의 목적 배경 취지"], entry["발언자 성명 및 직책"], prev_topic)
            # entry["주제"] = "test"
            # entry["주제"] = Modifier.modify_title(entry["기사 제목"])

            prev_topic = entry["주제"]
            prev_purpose = entry["발언의 목적 배경 취지"]

            elapsed_time = time.time() - start_time
            time_per_step = elapsed_time / (i + 1)
            remaining_time = time_per_step * (total_entries - (i + 1))
            formatted_remaining_time = format_remaining_time(remaining_time)

            print(
                f"[4단계 중 4단계] 주제 추출 및 CSV 파일 저장 중: {i + 1}/{total_entries} - 남은 예상 시간: {formatted_remaining_time}")

        # DataFrame 생성 및 저장
        df = pd.DataFrame(data)
        columns = ["날짜", "발언자 성명 및 직책", "신문사", "기사 제목",
                   "주제", "문단", "발언의 목적 배경 취지", "큰따옴표 발언"]
        df = df.reindex(columns=columns)
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')

        print(f"CSV 파일이 '{csv_file}'로 저장되었습니다.")
    except Exception as e:
        print(f"주제 추출 및 CSV 파일 저장 중 오류 발생: {e}")
        traceback.print_exc()


def process_file(csv_file, output_excel_file, output_csv_file):
    """주어진 CSV 파일에 대해 텍스트 추출, 데이터 병합, 중복 제거 후 엑셀로 저장하는 함수"""
    try:
        # 1. CSV에서 텍스트 추출
        extracted_data = extract_text_from_csv(csv_file)

        # 2. 데이터 병합
        merged_data = merge_data(extracted_data)

        # 3. 중복 제거
        cleaned_data = remove_duplicates(merged_data)

        # 4. 엑셀로 저장
        save_data_to_excel(cleaned_data, output_excel_file)
        # save_data_to_csv(cleaned_data, output_csv_file)

        print(f"처리 완료: {csv_file} -> {output_excel_file}")
    except Exception as e:
        print(f"{csv_file} 처리 중 오류 발생: {e}")


def get_csv_files_from_directory(directory_path):
    """주어진 디렉토리 내 모든 CSV 파일을 찾는 함수"""
    csv_files = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.csv'):
            csv_files.append(os.path.join(directory_path, file_name))
    return csv_files


def process_multiple_files(directory_path, output_dir):
    """디렉토리 내 모든 CSV 파일을 처리하는 함수"""
    csv_files = get_csv_files_from_directory(directory_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for csv_file in tqdm(csv_files, desc="파일 처리 중", unit="file"):
        formatted_date = datetime.datetime.now().strftime('%y%m%d')
        output_excel_file = os.path.join(
            output_dir, f"{os.path.basename(csv_file).replace('.csv', f'_AI변환_{formatted_date}.xlsx')}")
        output_csv_file = os.path.join(
            output_dir, f"{os.path.basename(csv_file).replace('.csv', f'_AI변환_{formatted_date}.csv')}")

        process_file(csv_file, output_excel_file, output_csv_file)


# 테스트용 코드
if __name__ == "__main__":
    formatted_date = datetime.datetime.now().strftime('%y%m%d')
    # directory_path = "/Users/gichanpark/Downloads/Input Sample Files"  # CSV 파일들이 위치한 디렉토리
    directory_path = "C:/Users/apg00/Downloads/Input Sample Files"
    # 출력 엑셀 파일이 저장될 디렉토리
    output_dir = f"/Users/gichanpark/Downloads/output_{formatted_date}"
    output_dir = f"C:/Users/apg00/Downloads/output_{formatted_date}"

    process_multiple_files(directory_path, output_dir)

    # unique_title = list(set(temp_title))
    # for title in unique_title:
    #     print(title)
