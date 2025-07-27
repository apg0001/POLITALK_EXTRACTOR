from transformers import BartForConditionalGeneration
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import nltk
import stanza
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


def restore_names_from_original(original: str, summary: str) -> str:
    def split_words(text):
        return re.findall(r'\b\w+\b', text)

    original_words = split_words(original)
    # print(original_words)
    summary_words = split_words(summary)
    # print(summary_words)

    # 2단어씩 묶은 후보들
    original_pairs = [(original_words[i], original_words[i+1])
                      for i in range(len(original_words) - 1)]
    summary_pairs = [(summary_words[i], summary_words[i+1])
                     for i in range(len(summary_words) - 1)]

    # 매핑된 short → full 딕셔너리
    replacement_map = {}

    for o1, o2 in original_pairs:
        for s1, s2 in summary_pairs:
            # short: 김 의원 / full: 김철수 의원
            if o1[0] == s1 and o2 == s2 and len(o1) >= 2:
                short_form = f"{s1} {s2}"
                full_form = f"{o1} {o2}"
                replacement_map[short_form] = full_form
                
    print(replacement_map)

    # 실제 교체 수행
    for short, full in replacement_map.items():
        summary = summary.replace(short, full)

    return summary


class TopicExtractor:
    def __init__(self):
        self.summarizer = Summarizer()
        self.remover = RedundancyRemover()

    def extract_topic(self, title = None, body = None, purpose = None, sentence = None, name = None):
        summary = self.summarizer.summarize(body)
        print(f"\n요약 결과:\t{summary}")
        
        removed = self.remover.trim_redundant_block(summary)
        print(f"중복 제거:\t{removed}")

        replaced = restore_names_from_original(body, removed)
        print(f"이름 복원:\t{replaced}")

        return replaced


class RedundancyRemover:
    def __init__(self, min_common_len=5):
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
                        max_start = start2
                        max_end = start2 + length

        # 제거할 중복 구간이 있다면 제거
        if max_len >= self.min_common_len:
            new_tokens = tokens[:max_start] + tokens[max_end:]
            return ' '.join(new_tokens)
        return text


# 🔍 예시 실행
if __name__ == "__main__":
    title = "김 의원, 장애인예술단 설립 질의"
    body1 = """
    30분간 언쟁이 오가고 김한표 당시 통합당 원내수석부대표가 회의장에 들어오고 나서야 충돌은 중단됐다. 김승희 전 통합당 의원은 2일 중앙일보와 통화에서 "저 역시 공공의료 확충에 반대하는 것은 아니었다. 하지만 지역에 공공의대를 따로 만들지 않더라도, 기존 의대 졸업생들에게 인센티브를 제공하는 방식으로 공공의료 인력을 늘릴 수도 있는데, 여당이 선거를 앞두고 밀어붙이는 게 옳지 않다고 반대했을 뿐"이라고 말했다.
김 전 의원은 정세균 총리와의 전화에 대해서도 "정확한 시점은 기억나지 않는다"며 "정 총리 전화가 강압적인 분위기는 아니었다. ‘서남대 의대 정원만큼 남원에 공공의료원을 만들테니 도와달라’는 내용의 정중한 전화였다"고 말했다. 정세균 국무총리는 전북 남원과 인접한 전북 진안·무주·장수 지역구에서 15~18대 국회의원을 지냈다.
    """
#     body2 = """
#     김현권 국회의원(더불어민주당·비례대표·사진)은 "최근 국방부의 통합신공항 부지선정 발표를 환영한다"면서 "앞으로 구미시를 신공항 배후 교통·물류·산업의 중심지로 커 나가도록 지원을 아끼지 않겠다"고 30일 밝혔다.
#     """
#     body3 = """
#     주변 도시를 잇는 교통망 확충 역시 신공항의 성패를 좌우할 핵심과제로 떠오르고 있다. 경북도에 따르면 2021년부터 전철 4곳, 고속도로 2곳 등 총 260㎞에 걸쳐 국비 6조원을 투입하는 신공항과 구미·포항·대구 등 인근 도시들을 연결하는 교통망 확충사업이 추진된다.
# 김 의원은 "구미시가 신공항배후단지로서 산업·교통·물류의 중심지로 부상하면 구미산단이나 아파트 신도시 활성화뿐만 아니라 도시와 농촌이 조화하는 지역 균형발전이 이뤄질 것"이라고 내다봤다.
#     """
    extractor = TopicExtractor()
    topic = extractor.extract_topic(title = title, body = body1)
    # topic = extractor.extract_topic(title, body2)
    # topic = extractor.extract_topic(title, body3)
