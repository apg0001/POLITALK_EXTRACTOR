from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import nltk
import re
from collections import defaultdict
from text_manager import nlp


class Summarizer:
    """텍스트 요약을 담당하는 클래스
    
    T5 기반 한국어 텍스트 요약 모델을 사용하여 긴 텍스트를 요약합니다.
    """
    
    def __init__(self, model_dir="lcw99/t5-base-korean-text-summary"):
        """Summarizer 초기화
        
        Args:
            model_dir (str): 사용할 T5 모델 경로
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        self.max_input_length = 2048
        

    def summarize(self, text, max_length=128):
        """텍스트 요약 실행"""
        inputs = self.tokenizer([text], max_length=self.max_input_length,
                                truncation=True, return_tensors="pt", padding=True)
        output = self.model.generate(
            **inputs, num_beams=16, do_sample=False, min_length=1, max_length=max_length)
        decoded = self.tokenizer.batch_decode(
            output, skip_special_tokens=True)[0]
        return nltk.sent_tokenize(decoded.strip())[0]


class TextCleaner:
    """텍스트 정리를 담당하는 클래스"""
    
    @staticmethod
    def remove_parentheses_content(text):
        """괄호 내용 제거"""
        pattern = r'[\(\[\<〈《][^)\]\>〉》]*[\)\]\>〉》]'
        cleaned = re.sub(pattern, '', text)
        return re.sub(r'\s{2,}', ' ', cleaned).strip()

    @staticmethod
    def restore_names_from_original(original, summary):
        """원문에서 이름 복원"""
        POSITION_SUFFIXES = ["의원", "장", "전", "당", "대표", "수석"]
        MAX_NAME_BLOCK = 4

        def split_words(text):
            return re.findall(r'\b[\w가-힣]+\b', text)

        def get_position_suffix(word):
            for suffix in POSITION_SUFFIXES:
                if suffix in word:
                    return suffix
            return None
        
        def ends_with_particle(text):
            return text.endswith(("은", "는", "이", "가", "와", "과", "도"))

        original = TextCleaner.remove_parentheses_content(original)
        original_words = split_words(original)
        summary_words = split_words(summary)

        original_blocks = []
        for i in range(len(original_words)):
            for size in range(2, MAX_NAME_BLOCK + 1):
                if i + size <= len(original_words):
                    block = original_words[i:i + size]
                    original_blocks.append(block)

        summary_pairs = [(summary_words[i], summary_words[i + 1])
                         for i in range(len(summary_words) - 1)]

        replacement_map = {}

        for block in original_blocks:
            if len(block) < 2:
                continue
            full_name = ' '.join(block)
            o1 = block[0]
            o2 = block[-1]

            for s1, s2 in summary_pairs:
                suffix_o = get_position_suffix(o2)
                suffix_s = get_position_suffix(s2)
                if (o1[0] == s1 and
                    (o2 == s2 or (suffix_o and suffix_o == suffix_s)) and
                    len(o1) >= 2 and len(o1) <= 3):
                    short_form = f"{s1} {s2}"
                    if (short_form not in replacement_map or
                        len(full_name) < len(replacement_map[short_form])):
                        replacement_map[short_form] = full_name

        for short, full in replacement_map.items():
            if short in full:
                continue
            if ((ends_with_particle(short) and ends_with_particle(full)) or
                (not ends_with_particle(short) and not ends_with_particle(full))):
                summary = summary.replace(short, full)

        return summary


class RedundancyRemover:
    """중복 제거를 담당하는 클래스"""
    
    def __init__(self, min_common_len=3):
        self.min_common_len = min_common_len
        self.nlp = nlp

    def tokenize(self, text):
        """텍스트를 토큰으로 분리"""
        doc = self.nlp(text)
        return [word.text for sent in doc.sentences for word in sent.words]

    def lemmatize(self, text):
        """텍스트를 원형으로 변환"""
        doc = self.nlp(text)
        return [word.lemma.split('+')[0] for sent in doc.sentences for word in sent.words]

    def trim_redundant_block(self, text):
        """중복 구간 제거"""
        tokens = self.tokenize(text)
        lemmas = self.lemmatize(text)

        lemma_map = defaultdict(list)
        for idx, lemma in enumerate(lemmas):
            lemma_map[lemma].append(idx)

        max_start, max_end = -1, -1
        max_len = 0

        for lemma, indices in lemma_map.items():
            if len(indices) < 2:
                continue
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

        if max_len >= self.min_common_len:
            new_tokens = tokens[:max_start] + tokens[max_end:]
            return ' '.join(new_tokens).replace(" .", ".")

        return text


class TopicExtractor:
    """주제 추출을 담당하는 메인 클래스
    
    발언문에서 주제와 배경을 추출하는 전체 파이프라인을 관리합니다.
    요약, 중복 제거, 이름 복원 등의 과정을 거쳐 최종 주제를 추출합니다.
    """
    
    def __init__(self):
        """TopicExtractor 초기화
        
        필요한 하위 프로세서들을 초기화합니다.
        """
        self.summarizer = Summarizer()
        self.remover = RedundancyRemover()
        self.text_cleaner = TextCleaner()

        print(f"Summurize Model device: {self.summarizer.model.device}")  # cuda:0 또는 cpu


    def extract_topic(self, title=None, body=None, purpose=None, sentences=None, name=None, prev_paragraph=None):
        """주제 추출 메인 함수
        
        Args:
            title (str): 기사 제목
            body (str): 발언문 본문
            purpose (str): 발언의 목적
            sentences (str): 발언문들
            name (str): 발언자 이름
            prev_paragraph (str): 이전 문단
            
        Returns:
            str: 추출된 주제/배경
        """
        # sentences(따옴표 발언문) 을 본문에서 제거하여
        # "발언의 배경"으로 사용할 비발언(설명/배경) 부분을 우선 추출한다.
        new_body = body or ""

        if sentences:
            # 발언문이 "  " (공백 두 개) 기준으로 이어져 넘어오는 구조 가정
            quoted_sentences = sentences.split("  ")
            for s in quoted_sentences:
                if not s:
                    continue
                new_body = new_body.replace(s, "")

        # 공백 정리
        new_body = new_body.replace("\n", " ").strip()
        prev_text = (prev_paragraph or "").replace("\n", " ").strip()

        # 유형 1, 2, 3 분기
        #
        # - 유형 1: 발언문단에 큰따옴표 문장 이외의 문장이 있는 경우
        #   case 1) 비발언(new_body)만 요약
        #   case 2) 비발언(new_body)이 너무 짧으면 앞 문단(prev_paragraph)과 합쳐서 요약
        #          → 비발언 단어 수가 9개 이하면 합침
        # - 유형 2: 발언문단에 큰따옴표 문장 이외의 문장이 없는 경우
        #   → 앞의 문단(prev_paragraph)만 요약
        # - 유형 3: 비발언도 없고 앞의 문단도 없는 경우
        #   → 요약 생략 (빈 문자열 반환)

        if new_body:  # 유형 1
            non_quote_words = new_body.split()
            if prev_text and len(non_quote_words) <= 9:
                # case 2: 비발언이 짧을 때 앞 문단과 함께 요약
                target_body = f"{prev_text} {new_body}".strip()
            else:
                # case 1: 비발언만 요약
                target_body = new_body
        else:
            if prev_text:  # 유형 2
                target_body = prev_text
            else:  # 유형 3
                return ""

        summary = self.summarizer.summarize(target_body.replace("\n", " "))

        if target_body == "" or "nan" in summary:
            return ""

        removed = self.remover.trim_redundant_block(summary)
        replaced = self.text_cleaner.restore_names_from_original(target_body, removed)

        return replaced


# 하위 호환성을 위한 함수들
def remove_parentheses_content(text):
    cleaner = TextCleaner()
    return cleaner.remove_parentheses_content(text)

def restore_names_from_original(original, summary):
    cleaner = TextCleaner()
    return cleaner.restore_names_from_original(original, summary)