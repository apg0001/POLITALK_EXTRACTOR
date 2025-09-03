# from transformers import BertTokenizer, BertModel
# import torch
# from collections import deque
# from csv2excel.text_manager import *
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re
import stanza

# Stanza 한국어 모델 로드
stanza.download("ko")
nlp = stanza.Pipeline("ko", processors='tokenize,pos,lemma,depparse')


def extract_quotes(text, name):
    """문단에서 큰따옴표로 묶인 발언을 추출하고, 특정 인물의 이름이 포함된 것만 반환."""
    text = text.replace("“", "\"").replace(
        "”", "\"").replace("\'", "'")  # 다양한 따옴표 처리
    quotes = []
    start = 0

    while start < len(text):
        start = text.find('"', start)
        if start == -1:
            break
        end = text.find('"', start + 1)
        if end == -1:
            break

        extracted_sentence = f'"{text[start + 1:end]}"'  # 불필요한 공백 제거
        quotes.append(extracted_sentence)
        # print(quotes)

        start = end + 1

    return "  ".join(quotes)


replacement_dict = {
    # 일반 동사 표현 간소화
    "지적했다": "지적",
    "강조했다": "강조",
    "반발했다": "반발",
    "말했다": "발언",
    "비판했다": "비판",
    "글을 올렸다": "게시",
    "썼다": "게시",
    "주장했다": "주장",
    "의심했다": "의심",
    "캐물었다": "질문",
    " 물었다": " 질문",
    "단정했다": "단정",
    "언급한 바 있다": "언급",
    "촉구했다": "촉구",
    "설명을 덧붙였다": "설명",
    "문제를 제기했다": "제기",
    "우려를 나타냈다": "우려",
    "확인을 요청했다": "요청",
    "지적을 가했다": "지적",
    "언급했다": "언급",
    "반박했다": "반박",

    # 관용적 표현 간소화
    "입장을 밝혔다": "입장",
    "강력히 주장했다": "주장",
    "중요성을 강조했다": "강조",
    "목소리를 냈다": "의견",
    "찬성 의견을 밝혔다": "찬성",
    "반대를 표명했다": "반대",
    "찬사를 보냈다": "찬사",
    "동의를 표했다": "동의",

    # 결론 및 평가 표현 간소화
    "결론을 내렸다": "결론",
    "평가를 내렸다": "평가",
    "확실히 했다": "확정",
    "입장을 표명했다": "입장",

    # 복합적 표현 간소화
    "질문을 던졌다": "질문",
    "알려진 바 있다": "알려짐",
    "조치를 취했다": "조치",
    "약속을 지켰다": "이행",
    "찬성을 표명했다": "찬성",

    # 긍정적 표현 간소화
    "찬사를 보냈다": "찬사",
    "환영을 표했다": "환영",
    "감사를 전했다": "감사",
    "공로를 치하했다": "치하",
    "동의를 표했다": "동의",
    "지지를 보냈다": "지지",
    "격려의 말을 전했다": "격려",
    "승인을 전했다": "승인",
    "축하를 전했다": "축하",
    "호평을 전했다": "호평",

    # 중립적 설명 및 요청
    "정보를 제공했다": "제공",
    "의견을 나눴다": "의견",
    "상황을 공유했다": "공유",
    "해결책을 제안했다": "제안",
    "문제를 설명했다": "설명",
    "진행 상황을 알렸다": "보고",
    "변화를 요구했다": "요구",
    "의미를 전달했다": "전달",
    "근거를 제시했다": "근거",
    "조언을 요청했다": "조언",

    # 갈등 및 비판 관련
    "비난을 가했다": "비난",
    "사과를 요구했다": "사과 요구",
    "잘못을 지적했다": "지적",
    "논란을 제기했다": "논란 제기",
    "불신을 드러냈다": "불신",
    "비판의 목소리를 냈다": "비판",
    "고발을 진행했다": "고발",
    "항의를 표했다": "항의",
    "불만을 드러냈다": "불만",
    "의혹을 제기했다": "의혹",
    "결과를 발표했다": "발표",
    "합의를 도출했다": "합의",
    "대화를 요청했다": "요청",
    "호소를 전했다": "호소",
    "결정권을 주장했다": "주장",
    "합의안을 제시했다": "제안",
    "필요성을 강조했다": "강조",
    "중요성을 지적했다": "지적",
    "타협을 제안했다": "타협",
    "문제를 제시했다": "제시",
    "대안을 주장했다": "주장",
    "목표를 강조했다": "강조",
    "이점을 설명했다": "설명",
    "문제를 고발했다": "고발",
    "해결책을 주장했다": "주장",

    # 기타 동작 표현 간소화
    "우려를 표명했다": "우려",
    "입장을 정리했다": "정리",
    "입장을 조율했다": "조율",
    "목표를 제시했다": "제시",
    "요청을 전달했다": "요청",
    "결과를 발표했다": "발표",
    "합의를 도출했다": "합의",

    # 기타 커스텀
    "마이크를 했다": "마이크를 잡고 발언",
    "마이크를 했고": "마이크를 잡고 발언",
    "밝혔다": "밝힘",
    "요청했다": "요청",
    "열기도 했다": "열기도 함",
    "제안했다": "제안",
    "언성을 높였고": "언성을 높여 발언",
    "라고도 했다": "대해 발언",
    "맞았다": "맞음",
    "제기했다": "제기",
    "보탰다": "보탬",
    "질문에 했다": "질문에 답변",
    "것에 밝혔다": "것에 대해 발언",
    "발언 했다": "발언",
    "관련해선": "관련해서",
    "많다 지적에": "많다는 지적에",
    "소감문을 밝혔다": "소감문을 통해 밝힘",
    " 고 발언": "발언",
    "호소했다": "호소",
    "입장문을 이라고 발언": "입장문을 통해 발언",
    "는 내용의 기자회견": "기자회견",
    "비난했다": "비난",
    "덧붙였다": "덧붙임",
    "진행 중단했다": "진행하려다가 일단 중단",
    " 고 했다": "발언",
    " 라고 했다": "발언",
    "검토에 했다": "검토에 대해 발언",
    "따져물었다": "발언",
    "평가했다": "평가",
    "말했지만": "발언",
    "질의했다": "질문",
    "역설하기도 했다": "역설함",
    "고 목소리를 높였다": "목소리를 높여 발언",
    " 했다": " 발언",

    # "했다": "함",
}


def simplify_purpose(sentence, name):
    """
    문장에서 대체 가능한 표현을 간소화.
    """
    for key, value in replacement_dict.items():
        if key in sentence:
            sentence = sentence.replace(key, value)
            # print(f"{key} -> {value} : {sentence}")
    if sentence in ["발언", "했다"]:
        sentence = f"{name}의 발언"
    elif sentence in ["물었다"]:
        sentence = f"{name}의 질문"
    return sentence


def to_string(text):
    """None값이나 숫자 등 문자열이 아닌 값을 문자열로 변경

    Args:
        text (_type_): _description_

    Returns:
        _type_: _description_
    """
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)
    return text


# 순접 접속사 리스트
sequential_conjunctions = [
    # 시간적 흐름을 나타내는 순접
    "이어", "이어서", "그러고는", "그러고 나서", "그 후에",
    "이후", "뒤이어", "잠시 후", "곧", "한편", "계속해서", "마침", "잠깐 후", "이때부터"

    # 원인과 결과를 나타내는 순접
    "그래서", "그러므로", "따라서", "그 결과", "이로 인해",
    "결국", "이 때문에", "그로 인해", "이와 같이", "그 덕분에", "그 탓에",

    # 추가 설명을 나타내는 순접
    "또한", "그리고", "더불어", "덧붙여", "나아가", "특히"
    "게다가", "더욱이", "한편", "그와 함께", "뿐만 아니라", "비슷하게",
    "같은 맥락에서", "추가적으로", "이와 동시에", "아울러", "같은 맥락에서",

    # 조건 충족 후 결과를 나타내는 순접
    "그러면", "그렇다면", "그때", "이 조건에서", "그럴 경우", "그러면서",

    # 기타 흐름을 이어주는 순접 표현
    "그러자", "결과적으로", "그리하여", "이에 따라", "따라서",
    "이렇게 해서", "다음으로", "그와 동시에", "종합하면", "총괄적으로",
    "이를 통해", "이런 점에서", "결국에는",

    # 기타
    # "그는", "그녀는",
    "이와 함께", "또"
]

exceptional_conjunctions = [
    "이에 대해",
]

# 문장에서 처음 5개 단어를 확인하고 접속사 탐지


def find_sequential_conjunction(sentence):
    # 문장을 단어로 분리 (공백 기준)
    words = sentence.split()

    # 처음 5개 단어 추출
    first_five_words = words[:5]

    # 순접 접속사와 비교
    for word in first_five_words:
        if word in sequential_conjunctions:
            return word  # 일치하는 접속사 반환

    return None  # 일치하는 접속사가 없을 경우 None 반환


def add_comma_after_target_words(sentence):
    """
    특정 단어(말했지만, 하지만 등) 뒤에 ','를 추가하는 함수.

    예제:
    "그는 말했다 하지만 나는 동의하지 않았다." → "그는 말했다 하지만, 나는 동의하지 않았다."
    """
    # 대상 단어 리스트 (뒤에 ',' 추가할 단어들)
    target_words = ["말했지만", "하지만", "그러나", "그럼에도",
                    "그렇지만", "다만", "반면에", "한편", "그렇다면"]

    # 정규 표현식 패턴 생성: 단어 뒤에 공백이 있으면 공백 포함하여 ',' 추가
    pattern = r'\b(' + '|'.join(target_words) + r')\b\s*'

    # 정규 표현식을 이용해 변환 (대상 단어 뒤에 `,` 추가)
    modified_sentence = re.sub(pattern, r'\1, ', sentence)

    return modified_sentence


def filter_sentences_by_name(sentences, keywords):
    """
    특정 이름이나 성이 포함된 문장 조각(컴마 기준) 또는 순접 접속사가 포함된 문장 조각만 필터링하는 함수.

    :param sentences: 리스트 형태의 문장들
    :param keywords: 필터링할 이름 및 관련 키워드 리스트
    :return: 필터링된 문장 리스트
    """
    filtered_sentences = []

    for sentence in sentences:

        sentence = sentence.replace("“", '"')
        sentence = sentence.replace("”", '"')
        sentence = sentence.replace("‘", "'")
        sentence = sentence.replace("’", " '")

        sentence = add_comma_after_target_words(sentence)
        # print(f"sentence: {sentence}")

        # 문장을 ',' 기준으로 분할
        # sentence_parts = sentence.split(',')
        pattern = r',(?=(?:[^\'"]*[\'"][^\'"]*[\'"])*[^\'"]*$)'
        sentence_parts = re.split(pattern, sentence)
        # print("sentence_parts: "  + '\n'.join(sentence_parts))

        # 필터링된 문장 조각을 저장할 리스트
        filtered_parts = []

        for part in sentence_parts:
            part = part.strip()  # 앞뒤 공백 제거
            # print(f"part: {part}")

            # 문장 조각이 큰따옴표로 시작하는지 확인
            starts_with_quote = part.startswith('"')

            # 키워드 포함 여부 확인
            contains_keyword = any(keyword in part for keyword in keywords)

            # 순접 접속사 포함 여부 확인
            contains_conjunction = find_sequential_conjunction(part)

            # 둘 중 하나라도 포함된 경우 해당 문장 조각을 추가
            if starts_with_quote or contains_keyword or contains_conjunction:
                filtered_parts.append(part)

        # 필터링된 문장 조각이 있으면 다시 ','로 결합하여 최종 리스트에 추가
        if filtered_parts:
            filtered_sentences.append(', '.join(filtered_parts))

    return filtered_sentences


def split_preserving_quotes(text):
    # 따옴표 안의 마침표를 무시하면서 문장을 나누는 정규식 패턴
    pattern = r'\.(?=(?:[^\'"]*["\'][^\'"]*["\'])*[^\'"]*$)'

    # 정규식 기반으로 문장 나누기
    sentences = re.split(pattern, text)

    # 양쪽 공백 제거 및 빈 문자열 제거
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def normalize_spaces_inside_single_quotes(text):
    # 작은따옴표 안의 공백을 정리하는 정규식
    return re.sub(r"'(.*?)'", lambda m: "'{}'".format(re.sub(r"\s+", " ", m.group(1).strip())), text)


exceptional_conjunctions = [
    "이에 대해",
]


exceptional_conjunctions = [
    "이에 대해",
]


class Merger:
    # todo : 쉼표가 있을 때 누구의 발언인지 확인하고 배제 가능?
    """
    행 합치기 로직에 사용되는 함수들
    """
    @staticmethod
    def split_text_by_quotes(text):
        """
        가장 첫 번째 큰따옴표 앞까지를 part_a,
        가장 마지막 큰따옴표 뒤부터를 part_c로 반환
        """
        matches = list(re.finditer(r'"', text))

        if len(matches) >= 2:
            first_quote_index = matches[0].start()
            last_quote_index = matches[-1].end()
            part_a = text[:first_quote_index].strip()
            part_c = text[last_quote_index:].strip()
            if len(part_c.split(" ")) > 1:
                part_c = " ".join(part_c.split(" ")[1:])
        else:
            part_a = text.strip()
            part_c = ""

        return part_a, part_c

    @staticmethod
    def extract_sentences(paragraph):
        """문단을 문장 단위로 분리하는 함수"""
        paragraph = paragraph.replace("“", '"')
        paragraph = paragraph.replace("”", '"')
        paragraph = paragraph.replace("‘", "'")
        paragraph = paragraph.replace("’", "'")
        doc = nlp(paragraph)
        return [sentence.text for sentence in doc.sentences]

    @classmethod
    def case_base(cls, paragraph, target_sentence, prev):
        """
        주어진 문단에서 특정 문장의 바로 앞 문장에 큰따옴표가 포함되어 있는지 확인
        todo : 앞문장의 발언이 동일 인물의 발언인가?
        """
        prev = [p.replace("\'", "'") for p in prev]
        # print("paragraph: ", paragraph)
        sentences = cls.extract_sentences(paragraph)
        # print("extracted: ", sentences)
        target_sentence = cls.extract_sentences(target_sentence)[0]
        idx = -1

        for i, sentence in enumerate(sentences):
            print(i, " : ", sentence)
            if target_sentence in sentence:
                idx = i
                print(idx, " : 해당 문장!!!!!!")

        if idx == -1:
            print("⚠ 입력된 문장이 문단 내에서 발견되지 않음.")
            return False

        if idx == 0:
            print("⚠ 입력된 문장이 문단의 첫 번째 문장이므로 앞 문장이 없음.")
            return False
        # print(idx)
        prev_sentence = sentences[idx - 1].replace("\'", "'")
        cur_sentence = sentences[idx]

        print("이전 입력: ", prev)
        print("이전 문장: ", prev_sentence)
        
        if prev_sentence.count('"'):
            try:
                prev_sentence = sentences[idx - 2].replace("\'", "'") + prev_sentence
            except:
                prev_sentence = prev_sentence

        if any((sent in prev_sentence) or (prev_sentence in sent) for sent in prev) or any((sent in cur_sentence) or (cur_sentence in sent) for sent in prev):
            print("case_base: 행합치기 대상.")
            return True
        else:
            print("case_base: 앞문장이 큰따옴표 문장이 아니므로 행 합치기 대상이 아님.")
            return False

    @classmethod
    def is_exceptional_conjunction(cls, part_a):  # 완료
        """
        예외 접속사가 있는 확인
        """
        if not part_a:
            return True

        condition_1 = part_a not in exceptional_conjunctions
        condition_2 = not any(
            conj in part_a for conj in exceptional_conjunctions)
        return condition_1 and condition_2

    @classmethod
    def case_same_sentence(cls, paragraph, target_sentence, prev):
        """
        주어진 문단에서 특정 문장의 바로 앞 문장에 큰따옴표가 포함되어 있는지 확인
        todo : 앞문장의 발언이 동일 인물의 발언인가?
        """
        sentences = cls.extract_sentences(paragraph)
        target_sentence = cls.extract_sentences(target_sentence)[0]
        idx = -1

        for i, sentence in enumerate(sentences):
            # print(i, " : ", sentence)
            if target_sentence in sentence:
                idx = i
                # print(idx, " : 해당 문장!!!!!!")

        if idx == -1:
            # print("⚠ 입력된 문장이 문단 내에서 발견되지 않음.")
            return False

        cur_sentence = sentences[idx]

        # if '"' in prev_sentence:
        if any(sent in cur_sentence for sent in prev):
            # print("같은 문장에 포함된 발언이므로 행합치기 대상.")
            return True
        else:
            return False

    @classmethod
    def is_case_1(cls, part_a, part_c):  # 완료
        """
        Case 1:
        A 파트에 접속사만 존재
        C 파트는 단일 동사로만 이루어짐
        """
        if not part_a or not part_c:
            return False

        condition_1 = part_a in sequential_conjunctions
        condition_2 = len(part_c.split(" ")) <= 2
        return condition_1 and condition_2

    @classmethod
    def is_case_2(cls, part_a, part_c):  # 완료
        """
        Case 2:
        A 파트에 접속사 + 'OOO 은, 는, 그는'으로 이루어짐
        C 파트는 단일 동사로 이루어짐
        """
        if not part_a or not part_c:
            return False

        condition_1 = (part_a.split(" ")[0] in sequential_conjunctions and (
            part_a.endswith("은") or part_a.endswith("는")))
        condition_2 = (("은" in part_a or "는" in part_a) and any(
            conj in part_a for conj in sequential_conjunctions) and len(part_a.split(" ")) <= 5)
        condition_3 = len(part_c.split(" ")) <= 2
        # print(("은" in part_a or "는" in part_a), any(conj in part_a for conj in sequential_conjunctions))
        # print("이와 함께" in part_a)
        # print(condition_1, condition_2, condition_3)
        # print(any(conj in part_a for conj in sequential_conjunctions))
        return (condition_1 or condition_2) and condition_3

    @classmethod
    def is_case_3(cls, part_a, part_c):  # 완료
        """
        Case 3:
        A 파트에 '그는'만 존재
        C 파트는 단일 동사로만 이루어짐
        """
        if not part_a or not part_c:
            return False

        condition_1 = part_a == "그는" or part_a == "그녀는"
        condition_2 = len(part_c.split(" ")) <= 2
        return condition_1 and condition_2

    @classmethod
    def is_case_4(cls, part_a, part_c):
        """
        Case 4:
        A 파트에 'OOO 은' 형태(세 글자 이상)만 포함
        C 파트는 단일 동사로만 이루어짐
        """
        if not part_a or not part_c:
            return False
        condition_1 = len(part_a.split(" ")) <= 4 and (
            part_a.endswith("은") or part_a.endswith("는"))
        condition_2 = len(part_c.split(" ")) <= 2

        # todo
        # 3번째 조건 자세히 추가해야 함
        return condition_1 and condition_2

    @classmethod
    def is_case_5(cls, part_a, part_c):  # 완료
        """
        Case 5:
        A 파트가 공란이고
        C 파트는 단일 동사로만 이루어짐
        """
        if not part_c:
            return False

        condition_1 = not part_a
        condition_2 = len(part_c.split(" ")) <= 2
        return condition_1 and condition_2

    @classmethod
    def check_cases(cls, text, paragraph, prev):
        # print(text, "\n", prev)
        """입력된 텍스트가 5가지 케이스 중 하나라도 만족하는지 확인"""
        part_a, part_c = cls.split_text_by_quotes(text)

        case_1 = cls.is_case_1(part_a, part_c)
        case_2 = cls.is_case_2(part_a, part_c)
        case_3 = cls.is_case_3(part_a, part_c)
        case_4 = cls.is_case_4(part_a, part_c)
        case_5 = cls.is_case_5(part_a, part_c)
        case_base = cls.case_base(paragraph, text, prev)
        case_exceptional_conjunction = cls.is_exceptional_conjunction(part_a)
        case_same_sentence = cls.case_same_sentence(paragraph, text, prev)

        print(case_1, case_2, case_3, case_4, case_5,
              case_base, case_exceptional_conjunction)
        print((case_1 or case_2 or case_3 or case_4 or case_5)
              and case_base and case_exceptional_conjunction, text, "\n================================================================")

        return ((case_1 or case_2 or case_3 or case_4 or case_5) and case_base and case_exceptional_conjunction) or case_same_sentence


def extract_and_clean_quotes(text):
    """
    텍스트에서 쌍따옴표로 묶인 문장을 추출하고 원래 문단에서 제거

    Args:
        text (str): 입력 텍스트

    Returns:
        tuple: (추출된 쌍따옴표 문장 리스트, 쌍따옴표 문장이 제거된 원래 문단)
    """

    text = text.replace("“", "\"")
    text = text.replace("”", "\"")
    text = text.replace("‘", "'")
    text = text.replace("'", "'")
    text = text.replace("’", "'")
    text = text.replace("\'", "'")
    quotes = re.findall(r'"(.*?)"', text)
    cleaned_text = re.sub(r'"(.*?)"', '""', text).strip()
    return quotes, cleaned_text


def split_sentences_by_comma(text):
    """쉼표를 기준으로 문장을 분리하되, 큰따옴표 및 작은따옴표 내부의 쉼표는 무시"""
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("‘", "'")
    text = text.replace("’", "'")

    result = []
    current_sentence = []
    in_single_quote = False  # 작은따옴표 상태
    in_double_quote = False  # 큰따옴표 상태

    for char in text:
        if char == '"' and not in_single_quote:  # 큰따옴표 상태 변경
            in_double_quote = not in_double_quote
        elif char == "'" and not in_double_quote:  # 작은따옴표 상태 변경
            in_single_quote = not in_single_quote

        if char == ',' and not in_single_quote and not in_double_quote:
            # 쉼표를 만나면 현재까지의 문장을 리스트에 추가
            result.append("".join(current_sentence).strip())
            current_sentence = []  # 새 문장 시작
        else:
            current_sentence.append(char)  # 현재 문장에 문자 추가

    # 마지막 문장 추가
    if current_sentence:
        result.append("".join(current_sentence).strip())

    return result


def merge_tokens(ner_results):
    """ BERT 토큰을 하나의 단어로 합치는 후처리 함수 """
    merged = []
    current_word = ""

    for entity in ner_results:
        word = entity["word"]
        if word.startswith("##"):  # 서브워드라면 이전 단어와 결합
            current_word += word[2:]
        elif entity['entity_group'] == "LABEL_35":
            current_word += (" " + word)
        else:  # 새로운 단어라면 이전 단어 저장 후 갱신
            if current_word:
                merged.append(current_word)
            current_word = word

    if current_word:  # 마지막 단어 추가
        merged.append(current_word)

    return merged


# 모델과 토크나이저 로드
model_name = "KPF/KPF-bert-ner"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# NER 파이프라인 생성
ner_pipeline = pipeline(
    "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")


def extract_speaker(text):
    ner_results = ner_pipeline(text)
    # print(ner_results)
    speakers = [entity for entity in ner_results if entity['entity_group'] in [
        # "LABEL_96", "LABEL_35", "LABEL_185", "LABEL_187", "LABEL_246"]]  # PS는 사람 이름(Person)을 의미
        "LABEL_96", "LABEL_185", "LABEL_187", "LABEL_246"]]  # LABEL_35는 직책임
    # return list(set(speakers))  # 중복 제거 후 반환

    # NER 결과 순회하며 예외 케이스 처리
    for i, entity in enumerate(ner_results[:-1]):
        # 현재 단어가 "이"이고 다음 단어가 직위/직책으로 인식될 경우
        if (
            entity['word'] == "이"
            and ner_results[i + 1]['entity_group'] == "LABEL_35"
        ):
            # '이'를 인물로 추가
            speakers.append({
                "word": "이",
                "entity_group": "LABEL_96",  # 사람 이름으로 간주
                "start": entity['start'],
                "end": entity['end']
            })

    return speakers


def calculate_similarity(sentence1, sentence2):
    """두 문장을 단어 단위로 비교하여 유사도 계산 (80% 이상이면 동일한 문장으로 간주)"""
    words1 = set(sentence1.split())  # 첫 번째 문장을 단어 단위로 분리
    words2 = set(sentence2.split())  # 두 번째 문장을 단어 단위로 분리
    if not words1 or not words2:
        return False
    # 공통 단어 수 계산
    common_words = words1 & words2
    total_words = min(len(words1), len(words2))  # 작은 쪽 기준으로 유사도 측정

    # 유사도 계산 (공통 단어 / 작은 쪽의 단어 수)
    similarity = len(common_words) / total_words if total_words > 0 else 0
    # print("sentence1: ", sentence1)
    # print("sentence2: ", sentence2)
    # print(f"중복 단어 수 : {len(common_words)}개, 전체 단어 수 : {total_words}개, 유사도 : {similarity*100}%")
    return similarity >= 0.7  # 유사도가 80% 이상이면 True


def normalize_text(text):
    """문장을 비교 전에 정규화 (공백, 특수문자 제거)"""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # 연속된 공백 제거
    text = re.sub(r'[^\w\s]', '', text)  # 특수문자 제거 (원하면 유지 가능)
    # text = re.sub(r'[“”"‘’,.?!]', '', text)  # 특수문자 제거 (필요에 따라 조정 가능)
    return text


def is_valid_speaker_by_josa(speakers, sentence):
    """주어 다음 조사 판단: '은', '는'이면 발언자 인정"""
    for speaker in speakers:
        if speaker in sentence:
            idx = sentence.find(speaker) + len(speaker)
            next_char = sentence[idx:idx+1]
            # print(next_char)
            if next_char in ["은", "는", "도", "또한"]:
                return True
            # 혹은 speaker 다음 단어가 있는 경우 조사 확인
            next_word = sentence[idx:].split()[0] if len(
                sentence[idx:].split()) > 0 else ""
            # print(next_word)
            if next_word.endswith(("은", "는")):
                return True
    return False


if __name__ == "__main__":
    # text = """이에 대해 고 최고위원은 "혁신위에서 내놓은 안들에 대해 오히려 더 강하게 추진해야 한다는 의원님들도 많기에 번복하려는 의도는 아닐 것 같다"면서도 "약속을 지키는 게 정치"라고 강조했다"""
    # name = "박기찬"

    # print(Merger.split_text_by_quotes(text))
    # part_a, part_c = Merger.split_text_by_quotes(text)

    # ✅ 테스트 실행
    paragraph = '''
민주당은 한국당 의원들을 검찰에 고발하면서도 검찰을 비판했다. 검찰은 전날 작년 4월 패스트트랙(신속 처리 안건) 지정 과정에서 있었던 충돌 사건과 관련해 한국당 의원 23명과 함께 민주당 이종걸·박범계 의원 등 5명을 기소했다.
이에 대해 이해찬 대표는 "폭력을 행사해 국회법을 위반한 한국당 의원들을 해를 넘겨 무려 8개월 만에 기소했다"며 "증거가 차고 넘치는데도 제대로 소환 조사도 하지 않다가 비로소 늑장 기소를 했다. 게다가 검찰이 자의적으로 기소권을 남용하는 행위라 개탄하지 않을 수 없다"고 했다. 이 대표는 "이래서 검찰 개혁이 필요한 거다"라고 했다.    

    '''

    sentence = """
    이 대표는 "이래서 검찰 개혁이 필요한 거다"라고 했다.

    """

    prev = ["""\"증거가 차고 넘치는데도 제대로 소환 조사도 하지 않다가 비로소 늑장 기소를 했다. 게다가 검찰이 자의적으로 기소권을 남용하는 행위라 개탄하지 않을 수 없다\""""]

    print(Merger.check_cases(sentence, paragraph, prev))

#     sentence = """
# 이 후보자는 "좋은 아이디어"라고 답했다

#     """

#     _, clean_sentence = extract_and_clean_quotes(sentence)
#     speakers = merge_tokens(extract_speaker(clean_sentence))

    # print(speakers)
