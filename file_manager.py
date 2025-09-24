import openpyxl
import pandas as pd
import time
import chardet
import traceback
from text_manager import *
from extract_purpose import extract_purpose
from extract_topic_summary import TopicExtractor


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


def extract_text_from_csv(csv_file, progress_bar, progress_label):
    """CSV 파일에서 텍스트를 추출하고 각 필드를 구분."""
    with open(csv_file, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']

    data = pd.read_csv(csv_file, encoding=encoding)
    extracted_data = []
    total_rows = len(data)
    progress_bar['maximum'] = total_rows
    start_time = time.time()  # 작업 시작 시간 기록

    for i, row in data.iterrows():
        sentences = split_sentences_by_comma(row['발췌문장'])

        for sentence in sentences:
            _, clean_sentence = extract_and_clean_quotes(sentence)
            candidate_speakers = merge_tokens(extract_speaker(clean_sentence))
            print(candidate_speakers)

            speakers = []

            # 단문이면 바로 추가
            if len(sentences) == 1:
                print(f"[단문] 문장 추가: {sentence}")
                add_flag = True
            else:
                # 조사 판별: '은', '는'만 통과
                for name in candidate_speakers:
                    print("후보 이름: ", name)
                    if is_valid_speaker_by_josa(name, clean_sentence):
                        speakers.append(name)

                # 성이 다른 경우 + 중문일 경우 제거
                if speakers:
                    print(f"발언자: {speakers}\n문장: {sentence}")
                    add_flag = any(speaker.startswith(
                        row['이름'][0]) for speaker in speakers)
                    
                    if not add_flag:
                        print(f"[추가 안함]성 불일치: {sentence}")
                        continue
                else:
                    print(f"[추가]발언자 없음 or 조사 불일치: {sentence}")
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
            print(extract_quotes(sentence, to_string(row['이름'])))

            if not any(is_empty(v) for v in current_data.values()):
                extracted_data.append(current_data)

        # 진행 상황 표시
        elapsed_time = time.time() - start_time
        time_per_step = elapsed_time / (i + 1)
        remaining_time = time_per_step * (total_rows - (i + 1))
        formatted_remaining_time = format_remaining_time(remaining_time)

        progress_bar['value'] = i + 1
        progress_label.config(
            text=f"[4단계 중 1단계] 파일 불러오기 및 큰따옴표 발언 추출 중 : {i + 1}/{total_rows} - 남은 예상 시간: {formatted_remaining_time}")
        progress_bar.update()

    return extracted_data


"""
todo
문단 내용이 다르면 문단도 합치기 ㅇㅇ
다른 사람의 발언이면 합치지 않아야 함
"""


def merge_data(data, progress_bar, progress_label):
    """
    내용 병합
    유형 7. 이에, 그러면서, 그리고, 이어, 이어서, 이에 더해, 이에 덧붙여, 계속해서, 등의 접속사(앞 문장이 동일주어의 발언일 경우)
    앞 문장의 행에 합침. 발언 셀은 앞의 행에 연이어 배열하고, 다른 셀은 앞의 행과 하나가 되는 것
    유형 7-2. 접속사 없이 앞문장과 동일한 발언자를 주어로 바로 시작
    유형 7-3. 접속사, 주어 없이 바로 큰따옴표로 시작하는 문장
    """
    if not data:
        print("[병합] 저장할 데이터가 없습니다.")
        return

    prev_title = None
    total_entries = len(data)
    progress_bar['maximum'] = total_entries
    start_time = time.time()  # 작업 시작 시간 기록

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
                # 병합 조건이면 가장 마지막 데이터에 큰따옴표 발언 추가
                if entry["큰따옴표 발언"] not in merged_data[-1]["큰따옴표 발언"]:
                    merged_data[-1]["큰따옴표 발언"] += ("  " + entry["큰따옴표 발언"])
                    # print("merged! : " +
                    #       entry["큰따옴표 발언"] + " : " + entry["날짜"], "\n================================================================")
                # 병합할 때 문단이 다르다면 문단도 합치기
                if entry["문단"] != merged_data[-1]["문단"]:
                    merged_data[-1]["문단"] += entry["문단"]
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

            # 프로그레스바 및 레이블 업데이트
            progress_bar['value'] = i + 1
            progress_label.config(
                text=f"[4단계 중 2단계] 내용 병합 중: {i + 1}/{total_entries} - 남은 예상 시간: {formatted_remaining_time}")
            progress_bar.update()

    except Exception as e:
        print(f"내용 병합 중 오류 발생: {e}")
        traceback.print_exc()  # 자세한 오류 정보를 출력

    return merged_data

# 중복 발언 제거


def remove_duplicates(data, progress_bar, progress_label):
    if not data:
        print("[중복 제거] 저장할 데이터가 없습니다.")
        return []

    sentence_sets = []  # 최근 200개 문장 유지
    total_entries = len(data)
    progress_bar['maximum'] = total_entries
    start_time = time.time()
    duplicate_removed_data = []

    try:
        for i, entry in enumerate(data):
            original_sentences = entry["큰따옴표 발언"].split("  ")
            normalized_sentences = [normalize_text(
                s) for s in original_sentences]

            # 기존 entry와 비교
            j = 0
            while j < len(sentence_sets):
                # print("입력 전체: ", sentence_sets[j]['original'])
                existing_entry = duplicate_removed_data[j]
                existing_sentences = existing_entry['큰따옴표 발언'].split("  ")
                existing_normalized = sentence_sets[j]['normalized']
                # print(f"{i}행, 기존 문장 수 : {len(existing_sentences)}")

                idx_new = 0
                while idx_new < len(normalized_sentences):
                    # print(
                        #     f"\n비교문장\n입력[{i}행][{idx_new}]: {normalized_sentences[idx_new]}\n기존[{j}행][{idx_exist}]: {existing_sentences[idx_exist]}")   
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
                            # 큰따옴표 문장의 갯수가 적은 쪽에서 삭제
                            if len(normalized_sentences) <= len(existing_normalized):
                                # print(
                                #     f"[{i}] '{original_sentences[idx_new]}' 제거 (new)")
                                # print(
                                #     f"비교대상: {existing_sentences[idx_exist]}"
                                # )
                                del original_sentences[idx_new]
                                del normalized_sentences[idx_new]
                                matched = True
                                break  # 삭제 후 현재 idx_new로 계속
                            else:
                                # print(
                                #     f"[{i}] '{existing_sentences[idx_exist]}' 제거 (existing)")
                                # print(
                                #     f"비교대상: {original_sentences[idx_new]}"
                                # )
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
                                    print(f"[{i}] 기존 entry {j} 삭제 (모든 문장 제거됨)")
                                    del duplicate_removed_data[j]
                                    del sentence_sets[j]
                                    j -= 1  # entry 삭제했으니 인덱스 보정
                                    break  # 현재 비교 종료
                        # else:
                        #     print(f"[{i}] '{existing_sentences[idx_exist]}' 유지")
                        idx_exist += 1
                    if not matched:
                        idx_new += 1
                j += 1

            # 결과 저장 (입력 entry)
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

            # 남은 예상 시간 계산
            elapsed_time = time.time() - start_time
            if i + 1 > 0:
                time_per_step = elapsed_time / (i + 1)
                remaining_time = time_per_step * (total_entries - (i + 1))
            else:
                remaining_time = 0

            # 남은 시간을 형식에 맞춰 변환하여 표시
            formatted_remaining_time = format_remaining_time(remaining_time)

            # 프로그레스바 및 레이블 업데이트
            progress_bar['value'] = i + 1
            progress_label.config(
                text=f"[4단계 중 3단계] 중복 제거 중: {i + 1}/{total_entries} - 남은 예상 시간: {formatted_remaining_time}")
            progress_bar.update()

    except Exception as e:
        print(f"중복 제거 중 오류 발생: {e}")
        traceback.print_exc()

    return duplicate_removed_data


def save_data_to_excel(data, excel_file, progress_bar, progress_label):
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
        progress_bar['maximum'] = total_entries
        start_time = time.time()  # 작업 시작 시간 기록

        prev_purpose = None
        prev_topic = None
        prev_title = None

        for i, entry in enumerate(data):
            # 모든 None 값을 빈 문자열로 대체
            for key, value in entry.items():
                if value is None:
                    entry[key] = ""

        for i, entry in enumerate(data):
            if prev_title != entry["기사 제목"]:
                prev_title = entry["기사 제목"]
                prev_topic = None
                prev_topic = None
            entry["발언의 목적 배경 취지"] = prev_purpose = extract_purpose(
                entry["발언자 성명 및 직책"], entry["기사 제목"], entry["문장"], entry["문단"], prev_purpose)
            # entry["주제"] = extract_topic(
            #     entry["기사 제목"], entry["큰따옴표 발언"], entry["발언자 성명 및 직책"])
            extractor = TopicExtractor()
            entry["주제"] = extractor.extract_topic(entry["기사 제목"], entry["문단"], entry["발언의 목적 배경 취지"], entry["큰따옴표 발언"], entry["발언자 성명 및 직책"])
            row = [entry.get(header, "") for header in headers]
            sheet.append(row)

            prev_topic = entry["주제"]
            prev_purpose = entry["발언의 목적 배경 취지"]

            # 남은 예상 시간 계산
            elapsed_time = time.time() - start_time
            if i + 1 > 0:
                time_per_step = elapsed_time / (i + 1)
                remaining_time = time_per_step * (total_entries - (i + 1))
            else:
                remaining_time = 0

            # 남은 시간을 형식에 맞춰 변환하여 표시
            formatted_remaining_time = format_remaining_time(remaining_time)

            # 프로그레스바 및 레이블 업데이트
            progress_bar['value'] = i + 1
            progress_label.config(
                text=f"[4단계 중 4단계] 주제 추출 및 엑셀 파일 저장 중: {i + 1}/{total_entries} - 남은 예상 시간: {formatted_remaining_time}")
            progress_bar.update()

        workbook.save(excel_file)
        print(f"엑셀 파일이 '{excel_file}'로 저장되었습니다.")
    except Exception as e:
        print(f"주제 추출 및 엑셀 파일 저장 중 오류 발생: {e}")
        traceback.print_exc()  # 자세한 오류 정보를 출력


# todo
"""
사람 이름 인식하는 것부터 해보자
test_codes에 써둔 NER 코드로 각 문장에서 사람 이름과 직위를 가져오자
1. 성이 동일하다 <- 일단 이것만 처리하자
해당 문장이 그러하다면 인정!
"""
def save_data_to_csv(data, csv_file, progress_bar=None, progress_label=None):
    """추출된 데이터를 CSV 파일로 저장."""
    if not data:
        print("[CSV 파일 저장] 저장할 데이터가 없습니다.")
        return

    try:
        headers = ["날짜", "발언자 성명 및 직책", "신문사", "기사 제목",
                   "주제", "문단", "발언의 목적 배경 취지", "큰따옴표 발언"]

        total_entries = len(data)
        if progress_bar is not None:
            progress_bar['maximum'] = total_entries
        start_time = time.time()  # 작업 시작 시간 기록

        prev_purpose = None
        prev_topic = None
        prev_title = None

        # None 값을 빈 문자열로 대체
        for entry in data:
            for key, value in entry.items():
                if value is None:
                    entry[key] = ""

        # 주제 및 목적 필드 처리
        for i, entry in enumerate(data):
            if prev_title != entry["기사 제목"]:
                prev_title = entry["기사 제목"]
                prev_topic = None
                prev_purpose = None

            entry["발언의 목적 배경 취지"] = prev_purpose = extract_purpose(
                entry["발언자 성명 및 직책"], entry["기사 제목"], entry["문장"], entry["문단"], prev_purpose)
            # 주제 추출 부분은 필요에 따라 활성화하세요
            # entry["주제"] = extract_topic(entry["기사 제목"], entry["큰따옴표 발언"], entry["발언자 성명 및 직책"])
            extractor = TopicExtractor()
            entry["주제"] = extractor.extract_topic(entry["기사 제목"], entry["문단"], entry["발언의 목적 배경 취지"], entry["큰따옴표 발언"], entry["발언자 성명 및 직책"])

            # 진행 상태 표시
            if progress_bar is not None and progress_label is not None:
                elapsed_time = time.time() - start_time
                time_per_step = elapsed_time / (i + 1)
                remaining_time = time_per_step * (total_entries - (i + 1))
                formatted_remaining_time = format_remaining_time(remaining_time)
                progress_bar['value'] = i + 1
                progress_label.config(
                    text=f"[4단계 중 4단계] 주제 추출 및 CSV 파일 저장 중: {i + 1}/{total_entries} - 남은 예상 시간: {formatted_remaining_time}")

        # DataFrame 생성 후 CSV 저장
        df = pd.DataFrame(data, columns=headers)
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')

        print(f"CSV 파일이 '{csv_file}'로 저장되었습니다.")

    except Exception as e:
        print(f"주제 추출 및 CSV 파일 저장 중 오류 발생: {e}")
        traceback.print_exc()