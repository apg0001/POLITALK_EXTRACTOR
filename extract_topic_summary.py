from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import nltk
import re
from collections import defaultdict
from text_manager import nlp


class Summarizer:
    def __init__(self, model_dir="lcw99/t5-base-korean-text-summary"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        self.max_input_length = 2048

    def summarize(self, text, max_length=128):
        inputs = self.tokenizer([text], max_length=self.max_input_length,
                                truncation=True, return_tensors="pt", padding=True)
        output = self.model.generate(
            **inputs, num_beams=16, do_sample=False, min_length=1, max_length=max_length)
        decoded = self.tokenizer.batch_decode(
            output, skip_special_tokens=True)[0]
        return nltk.sent_tokenize(decoded.strip())[0]


def remove_parentheses_content(text: str) -> str:
    # 괄호쌍: (), [], <>, 〈〉, 《》
    pattern = r'[\(\[\<〈《][^)\]\>〉》]*[\)\]\>〉》]'
    cleaned = re.sub(pattern, '', text)
    return re.sub(r'\s{2,}', ' ', cleaned).strip()


import re

def remove_parentheses_content(text: str) -> str:
    pattern = r'[\(\[\<〈《][^)\]\>〉》]*[\)\]\>〉》]'
    cleaned = re.sub(pattern, '', text)
    return re.sub(r'\s{2,}', ' ', cleaned).strip()


def restore_names_from_original(original: str, summary: str) -> str:
    POSITION_SUFFIXES = ["의원", "장", "전", "당", "대표", "수석"]
    MAX_NAME_BLOCK = 4  # 최대 4단어까지 이름 블록으로 간주

    def split_words(text):
        return re.findall(r'\b[\w가-힣]+\b', text)

    def get_position_suffix(word: str) -> str | None:
        for suffix in POSITION_SUFFIXES:
            if suffix in word:
                return suffix
        return None
    
    def ends_with_particle(text):
        return text.endswith(("은", "는", "이", "가", "와", "과", "도"))

    original = remove_parentheses_content(original)
    original_words = split_words(original)
    summary_words = split_words(summary)

    # 원문에서 2~4단어씩 블록 추출
    original_blocks = []
    for i in range(len(original_words)):
        for size in range(2, MAX_NAME_BLOCK + 1):
            if i + size <= len(original_words):
                block = original_words[i:i + size]
                original_blocks.append(block)

    # 요약문 2단어쌍
    summary_pairs = [(summary_words[i], summary_words[i + 1])
                     for i in range(len(summary_words) - 1)]

    replacement_map = {}

    for block in original_blocks:
        if len(block) < 2:
            continue
        full_name = ' '.join(block)
        o1 = block[0]
        o2 = block[-1]  # 직책 추정

        for s1, s2 in summary_pairs:
            suffix_o = get_position_suffix(o2)
            suffix_s = get_position_suffix(s2)
            if (
                o1[0] == s1 and
                (o2 == s2 or (suffix_o and suffix_o == suffix_s)) and
                len(o1) >= 2 and
                len(o1) <= 3
            ):
                short_form = f"{s1} {s2}"
                if (
                    short_form not in replacement_map or
                    len(full_name) < len(replacement_map[short_form])
                ):
                    replacement_map[short_form] = full_name

    print(replacement_map)

    # 실제 치환
    for short, full in replacement_map.items():
        if short in full:
            continue
        if (ends_with_particle(short) and ends_with_particle(full)) or \
            (not ends_with_particle(short) and not ends_with_particle(full)):
            summary = summary.replace(short, full)

    return summary


class RedundancyRemover:
    def __init__(self, min_common_len=3):
        self.min_common_len = min_common_len
        self._init_nlp()

    def _init_nlp(self):
        # stanza.download('ko')
        # self.nlp = stanza.Pipeline(
        #     lang='ko', processors='tokenize,pos,lemma', verbose=False)
        self.nlp = nlp

    def tokenize(self, text: str):
        doc = self.nlp(text)
        return [word.text for sent in doc.sentences for word in sent.words]

    def lemmatize(self, text: str):
        doc = self.nlp(text)
        return [word.lemma.split('+')[0] for sent in doc.sentences for word in sent.words]

    def trim_redundant_block(self, text: str) -> str:
        tokens = self.tokenize(text)
        lemmas = self.lemmatize(text)

        # lemma -> 모든 등장 인덱스 기록
        lemma_map = defaultdict(list)
        for idx, lemma in enumerate(lemmas):
            lemma_map[lemma].append(idx)

        # 연속된 반복 구간 후보 찾기
        max_start, max_end = -1, -1
        max_len = 0

        for lemma, indices in lemma_map.items():
            if len(indices) < 2:
                continue
            # 모든 가능한 (i, j) 쌍 비교 (i < j)
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    start1, start2 = indices[i], indices[j]
                    length = 0
                    while (start1 + length < start2 and
                           start2 + length < len(lemmas) and
                           lemmas[start1 + length] == lemmas[start2 + length]):
                        length += 1
                    if length >= self.min_common_len and length > max_len:
                        max_len = length
                        max_start = start1
                        max_end = start1 + length

        # 제거할 중복 구간이 있다면 제거
        if max_len >= self.min_common_len:
            new_tokens = tokens[:max_start] + tokens[max_end:]
            return ' '.join(new_tokens).replace(" .", ".")

        return text


# class RedundancyRemover:
#     POSITION_SUFFIXES = ["의원", "장", "당", "대표", "수석"]
#     def __init__(self, min_common_len=3):
#         self.min_common_len = min_common_len
#         self._init_nlp()

#     def _init_nlp(self):
#         # self.nlp = stanza.Pipeline(...)
#         self.nlp = nlp  # 외부에서 주입한 stanza Pipeline

#     def tokenize(self, text: str):
#         doc = self.nlp(text)
#         return [word.text for sent in doc.sentences for word in sent.words]

#     def lemmatize(self, text: str):
#         doc = self.nlp(text)
#         # print(doc)
#         return [word.lemma.split('+')[0] for sent in doc.sentences for word in sent.words]

#     def trim_redundant_block(self, text: str) -> str:
#         while True:
#             tokens = self.tokenize(text)
#             lemmas = self.lemmatize(text)
            
#             print(lemmas)

#             # lemma → 등장 인덱스 기록
#             lemma_map = defaultdict(list)
#             for idx, lemma in enumerate(lemmas):
#                 lemma_map[lemma].append(idx)

#             # 가장 긴 반복 구간 탐색
#             max_start, max_end, max_len = -1, -1, 0

#             for lemma, indices in lemma_map.items():
#                 if len(indices) < 2:
#                     continue
#                 for i in range(len(indices)):
#                     for j in range(i + 1, len(indices)):
#                         start1, start2 = indices[i], indices[j]
#                         length = 0
#                         while (start1 + length < start2 and
#                                start2 + length < len(lemmas) and
#                                lemmas[start1 + length] == lemmas[start2 + length]):
#                             length += 1
#                         if length >= self.min_common_len and length > max_len:
#                             max_len = length
#                             max_start = start1
#                             max_end = start1 + length

#             # 제거할 중복 구간이 없다면 종료
#             if max_len < self.min_common_len:
#                 break

#             # 중복 구간 제거
#             tokens = tokens[:max_start] + tokens[max_end:]

#             text = ' '.join(tokens).replace(" .", ".")

#         return text

class TopicExtractor:
    def __init__(self):
        self.summarizer = Summarizer()
        self.remover = RedundancyRemover()

    def extract_topic(self, title=None, body=None, purpose=None, sentence=None, name=None):
        summary = self.summarizer.summarize(body.replace("\n", " "))
        print(f"\n요약 결과:\t{summary}")

        # 본문이 없는 경우 빈칸 반환
        if body == "" or "nan" in summary:
            return ""

        removed = self.remover.trim_redundant_block(summary)
        print(f"중복 제거:\t{removed}")

        replaced = restore_names_from_original(body, removed)
        print(f"이름 복원:\t{replaced}")

        return replaced


# 🔍 예시 실행
if __name__ == "__main__":
    title = "김 의원, 장애인예술단 설립 질의"
    body1 = """
민주당 의원들은 집회 참석에 이어 사회관계망서비스(SNS)를 통해서도 정부·여당을 향한 규탄 메시지를 앞다퉈 쏟아냈다. 이연희 의원은 "총선에서 국민이 심판했는데 대통령이 듣지 않는다면 국민들이 나서야 한다"며 "윤석열 정권이 국정 기조를 전환하고 인적 쇄신을 이룰 때까지 국민들이 나서서 윤 대통령을 굴복시켜야 한다. 그 길에 민주당이 앞장설 것"이라고 했다. 윤건영 의원은 "정부와 여당은 한 몸으로 해병대원 특검법을 거부했다. 진실을 숨기고 자기 자신만 지키기 위한 합동 권한남용 작전"이라며 "끝까지 숨길 수 있는 진실은 없다"고 강조했다. 염태영 의원은 "국방의 의무를 다하다 순직한 한 젊은 군인과 그 가족들의 한을 풀 수 있도록 해달라"며 "손바닥으로 하늘을 가리려는 대통령과 여당을 국민의 매서운 회초리로 응징해달라"고 호소했다. 김동아 의원은 "(정부·여당이) 권력을 사적으로 악용하는 모습을 더 이상 우리는 용납하지 않을 것이다. 신속하고 강력하게 국민이 위임한 권한을 행사해나갈 것"이라고 했다.

"""

    extractor = TopicExtractor()
    topic = extractor.extract_topic(title=title, body=body1)
