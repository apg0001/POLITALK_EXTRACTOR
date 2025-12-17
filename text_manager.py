import re
import stanza
from ner_extractor import NERExtractor
from text_cleaner import TextCleaner
import torch

# 자동 디바이스 감지
device = "cuda" if torch.cuda.is_available() else "cpu"

# Stanza 한국어 모델 로드
stanza.download("ko")
nlp = stanza.Pipeline(
    "ko", 
    processors='tokenize,pos,lemma,depparse', 
    use_gpu=torch.cuda.is_available(),
    device=device
)
device_str = "CUDA" if device == "cuda" else "CPU"
print(f"stanza_ko Using {device_str}")


class TextProcessor:
    """텍스트 처리 관련 기능을 담당하는 클래스
    
    이 클래스는 발언문에서 텍스트를 추출, 정제, 분석하는 모든 기능을 제공합니다.
    NER 모델을 사용한 발언자 추출, 텍스트 정규화, 유사도 계산 등의 기능을 포함합니다.
    """
    
    def __init__(self):
        """TextProcessor 초기화
        
        NER 모델과 텍스트 처리에 필요한 딕셔너리들을 초기화합니다.
        """
        self.ner_extractor = NERExtractor()
        self.text_cleaner = TextCleaner()

    def extract_quotes(self, text, name):
        """문단에서 큰따옴표로 묶인 발언을 추출하고, 특정 인물의 이름이 포함된 것만 반환
        
        Args:
            text (str): 추출할 텍스트
            name (str): 발언자 이름
            
        Returns:
            str: 추출된 발언문들을 공백으로 연결한 문자열
        """
        return self.text_cleaner.extract_quotes(text, name)

    def simplify_purpose(self, sentence, name):
        """문장에서 대체 가능한 표현을 간소화"""
        return self.text_cleaner.simplify_purpose(sentence, name)

    def to_string(self, text):
        """None값이나 숫자 등 문자열이 아닌 값을 문자열로 변경"""
        return self.text_cleaner.to_string(text)

    def find_sequential_conjunction(self, sentence):
        """문장에서 처음 5개 단어를 확인하고 접속사 탐지"""
        return self.text_cleaner.find_sequential_conjunction(sentence)

    def add_comma_after_target_words(self, sentence):
        """특정 단어(말했지만, 하지만 등) 뒤에 ','를 추가하는 함수"""
        return self.text_cleaner.add_comma_after_target_words(sentence)

    def filter_sentences_by_name(self, sentences, keywords):
        """특정 이름이나 성이 포함된 문장 조각(컴마 기준) 또는 순접 접속사가 포함된 문장 조각만 필터링"""
        return self.text_cleaner.filter_sentences_by_name(sentences, keywords)

    def split_preserving_quotes(self, text):
        """따옴표 안의 마침표를 무시하면서 문장을 나누는 함수"""
        return self.text_cleaner.split_preserving_quotes(text)

    def normalize_spaces_inside_single_quotes(self, text):
        """작은따옴표 안의 공백을 정리하는 함수"""
        return self.text_cleaner.normalize_spaces_inside_single_quotes(text)

    def extract_and_clean_quotes(self, text):
        """텍스트에서 쌍따옴표로 묶인 문장을 추출하고 원래 문단에서 제거"""
        return self.text_cleaner.extract_and_clean_quotes(text)

    def split_sentences_by_comma(self, text):
        """쉼표를 기준으로 문장을 분리하되, 큰따옴표 및 작은따옴표 내부의 쉼표는 무시"""
        return self.text_cleaner.split_sentences_by_comma(text)

    def merge_tokens(self, ner_results):
        """BERT 토큰을 하나의 단어로 합치는 후처리 함수"""
        return self.ner_extractor.merge_tokens(ner_results)

    def extract_speaker(self, text):
        """NER을 사용하여 발언자 추출
        
        Args:
            text (str): 발언자 추출할 텍스트
            
        Returns:
            list: 추출된 발언자 정보 리스트
        """
        return self.ner_extractor.extract_speaker(text)

    def calculate_similarity(self, sentence1, sentence2, criteria):
        """두 문장을 단어 단위로 비교하여 유사도 계산"""
        return self.text_cleaner.calculate_similarity(sentence1, sentence2, criteria)

    def normalize_text(self, text):
        """문장을 비교 전에 정규화 (공백, 특수문자 제거)"""
        return self.text_cleaner.normalize_text(text)

    def is_valid_speaker_by_josa(self, speakers, sentence):
        """주어 다음 조사 판단: '은', '는'이면 발언자 인정"""
        return self.text_cleaner.is_valid_speaker_by_josa(speakers, sentence)


class Merger:
    """행 합치기 로직에 사용되는 함수들
    
    이 클래스는 연속된 발언들을 병합하기 위한 다양한 케이스를 판별하는 기능을 제공합니다.
    접속사, 발언자, 문맥 등을 고려하여 발언 병합 여부를 결정합니다.
    """
    
    # 순접 접속사 리스트 (병합 가능한 접속사들)
    SEQUENTIAL_CONJUNCTIONS = [
        "이어", "이어서", "그러고는", "그러고 나서", "그 후에", "이후", "뒤이어", "잠시 후",
        "곧", "한편", "계속해서", "마침", "잠깐 후", "이때부터", "그래서", "그러므로",
        "따라서", "그 결과", "이로 인해", "결국", "이 때문에", "그로 인해", "이와 같이",
        "그 덕분에", "그 탓에", "또한", "그리고", "더불어", "덧붙여", "나아가", "특히",
        "게다가", "더욱이", "한편", "그와 함께", "뿐만 아니라", "비슷하게", "같은 맥락에서",
        "추가적으로", "이와 동시에", "아울러", "그러면", "그렇다면", "그때", "이 조건에서",
        "그럴 경우", "그러면서", "그러자", "결과적으로", "그리하여", "이에 따라", "따라서",
        "이렇게 해서", "다음으로", "그와 동시에", "종합하면", "총괄적으로", "이를 통해",
        "이런 점에서", "결국에는", "이와 함께", "또"
    ]
    
    @staticmethod
    def split_text_by_quotes(text):
        """큰따옴표를 기준으로 텍스트를 세 부분으로 분리
        
        큰따옴표로 묶인 발언문을 기준으로:
        - part_a: 첫 번째 큰따옴표 앞 부분 (접속사, 주어 등)
        - part_c: 마지막 큰따옴표 뒤 부분 (동사, 서술어 등)
        
        Args:
            text (str): 분리할 텍스트
            
        Returns:
            tuple: (part_a, part_c) - 큰따옴표 앞부분과 뒷부분
        """
        quote_matches = list(re.finditer(r'"', text))

        # 큰따옴표가 2개 이상 있는 경우 (발언문 포함)
        if len(quote_matches) >= 2:
            first_quote_start = quote_matches[0].start()
            last_quote_end = quote_matches[-1].end()
            
            # 첫 번째 큰따옴표 앞 부분 (접속사, 주어 등)
            part_before_quote = text[:first_quote_start].strip()
            
            # 마지막 큰따옴표 뒤 부분 (동사, 서술어 등)
            part_after_quote = text[last_quote_end:].strip()
            
            # part_c가 여러 단어로 이루어진 경우, 첫 단어 제거 (보통 동사 앞 조사)
            if len(part_after_quote.split(" ")) > 1:
                part_after_quote = " ".join(part_after_quote.split(" ")[1:])
        else:
            # 큰따옴표가 없는 경우
            part_before_quote = text.strip()
            part_after_quote = ""

        return part_before_quote, part_after_quote

    @staticmethod
    def extract_sentences(paragraph):
        """문단을 문장 단위로 분리하는 함수"""
        paragraph = paragraph.replace(""", '"').replace(""", '"').replace("'", "'").replace("'", "'")
        doc = nlp(paragraph)
        return [sentence.text for sentence in doc.sentences]

    @classmethod
    def case_base(cls, paragraph, target_sentence, previous_quoted_speeches):
        """기본 병합 조건 확인: 이전 발언과 문맥적으로 연결되어 있는지 확인
        
        현재 문장의 앞 문장이 이전 발언문과 관련이 있는지 확인합니다.
        큰따옴표가 닫히지 않은 경우 이전 문장과 합쳐서 확인합니다.
        
        Args:
            paragraph (str): 현재 문단
            target_sentence (str): 확인할 대상 문장
            previous_quoted_speeches (list): 이전 발언문 리스트
            
        Returns:
            bool: 이전 발언과 문맥적으로 연결되어 있으면 True
        """
        # 문단을 문장 단위로 분리
        paragraph_sentences = cls.extract_sentences(paragraph)
        target_sentence_text = cls.extract_sentences(target_sentence)[0]
        
        # 대상 문장이 문단의 몇 번째 문장인지 찾기
        target_sentence_idx = -1
        for idx, sentence in enumerate(paragraph_sentences):
            if target_sentence_text in sentence:
                target_sentence_idx = idx
                break

        # 대상 문장을 찾지 못했거나 첫 번째 문장이면 병합 불가
        if target_sentence_idx == -1 or target_sentence_idx == 0:
            return False

        # 대상 문장의 바로 앞 문장 가져오기
        previous_sentence = paragraph_sentences[target_sentence_idx - 1]
        current_sentence = paragraph_sentences[target_sentence_idx]

        # 큰따옴표가 닫히지 않은 경우 (홀수 개) 이전 문장과 합치기
        # 큰따옴표가 문장 경계를 넘어가는 경우 처리
        while previous_sentence.count('"') % 2 != 0:
            try:
                previous_sentence = paragraph_sentences[target_sentence_idx - 2] + " " + previous_sentence
                target_sentence_idx -= 1
            except IndexError:
                break

        # 이전 발언문이 앞 문장이나 현재 문장에 포함되어 있는지 확인
        is_previous_speech_in_context = any(
            (prev_speech in previous_sentence) or (previous_sentence in prev_speech) 
            for prev_speech in previous_quoted_speeches
        )
        is_previous_speech_in_current = any(
            (prev_speech in current_sentence) or (current_sentence in prev_speech) 
            for prev_speech in previous_quoted_speeches
        )
        
        return is_previous_speech_in_context or is_previous_speech_in_current

    @classmethod
    def is_exceptional_conjunction(cls, part_before_quote):
        """예외 접속사 확인: 병합하지 않아야 하는 접속사가 있는지 확인
        
        "이에 대해" 같은 예외 접속사가 있으면 병합하지 않습니다.
        
        Args:
            part_before_quote (str): 큰따옴표 앞 부분
            
        Returns:
            bool: 예외 접속사가 없으면 True (병합 가능), 있으면 False (병합 불가)
        """
        if not part_before_quote:
            return True

        # 병합하지 않아야 하는 예외 접속사 리스트
        exceptional_conjunctions = ["이에 대해"]
        
        # 조건 1: 앞 부분이 예외 접속사와 정확히 일치하지 않음
        is_not_exact_match = part_before_quote not in exceptional_conjunctions
        
        # 조건 2: 앞 부분에 예외 접속사가 포함되지 않음
        has_no_exceptional = not any(conj in part_before_quote for conj in exceptional_conjunctions)
        
        return is_not_exact_match and has_no_exceptional

    @classmethod
    def case_same_sentence(cls, paragraph, target_sentence, previous_quoted_speeches):
        """같은 문장 내 병합 조건 확인: 현재 문장에 이전 발언이 포함되어 있는지 확인
        
        현재 문장 자체에 이전 발언문이 포함되어 있으면 병합 가능합니다.
        
        Args:
            paragraph (str): 현재 문단
            target_sentence (str): 확인할 대상 문장
            previous_quoted_speeches (list): 이전 발언문 리스트
            
        Returns:
            bool: 현재 문장에 이전 발언이 포함되어 있으면 True
        """
        # 문단을 문장 단위로 분리
        paragraph_sentences = cls.extract_sentences(paragraph)
        target_sentence_text = cls.extract_sentences(target_sentence)[0]
        
        # 대상 문장이 문단의 몇 번째 문장인지 찾기
        target_sentence_idx = -1
        for idx, sentence in enumerate(paragraph_sentences):
            if target_sentence_text in sentence:
                target_sentence_idx = idx
                break

        # 대상 문장을 찾지 못했으면 병합 불가
        if target_sentence_idx == -1:
            return False

        current_sentence = paragraph_sentences[target_sentence_idx]

        # 현재 문장에 이전 발언문이 포함되어 있는지 확인
        return any(prev_speech in current_sentence for prev_speech in previous_quoted_speeches)

    @classmethod
    def is_case_1(cls, part_before_quote, part_after_quote):
        """Case 1: 접속사만 + 단일 동사 패턴
        
        큰따옴표 앞 부분(part_before_quote)이 순접 접속사만 있고,
        큰따옴표 뒤 부분(part_after_quote)이 단일 동사(2단어 이하)로만 이루어진 경우
        
        예: "이어 말했다" -> part_before_quote="이어", part_after_quote="말했다"
        
        Args:
            part_before_quote (str): 큰따옴표 앞 부분
            part_after_quote (str): 큰따옴표 뒤 부분
            
        Returns:
            bool: Case 1 조건을 만족하면 True
        """
        if not part_before_quote or not part_after_quote:
            return False

        # 조건 1: 앞 부분이 순접 접속사와 정확히 일치
        is_sequential_conjunction = part_before_quote in cls.SEQUENTIAL_CONJUNCTIONS
        
        # 조건 2: 뒤 부분이 단일 동사로만 이루어짐 (2단어 이하)
        is_single_verb = len(part_after_quote.split(" ")) <= 2
        
        return is_sequential_conjunction and is_single_verb

    @classmethod
    def is_case_2(cls, part_before_quote, part_after_quote):
        """Case 2: 접속사 + 주어(은/는) + 단일 동사 패턴
        
        큰따옴표 앞 부분이 접속사와 주어(은/는)로 이루어지고,
        큰따옴표 뒤 부분이 단일 동사로만 이루어진 경우
        
        예: "이어서 그는 말했다" -> part_before_quote="이어서 그는", part_after_quote="말했다"
        
        Args:
            part_before_quote (str): 큰따옴표 앞 부분
            part_after_quote (str): 큰따옴표 뒤 부분
            
        Returns:
            bool: Case 2 조건을 만족하면 True
        """
        if not part_before_quote or not part_after_quote:
            return False

        part_before_words = part_before_quote.split(" ")
        
        # 조건 1: 첫 단어가 접속사이고, 마지막이 "은" 또는 "는"으로 끝남
        # 예: "이어서 그는" -> ["이어서", "그는"]
        is_conjunction_with_subject = (
            part_before_words[0] in cls.SEQUENTIAL_CONJUNCTIONS and
            (part_before_quote.endswith("은") or part_before_quote.endswith("는"))
        )
        
        # 조건 2: "은" 또는 "는"이 포함되고, 접속사가 포함되며, 5단어 이하
        # 예: "그래서 그는" -> "은" 포함, "그래서" 접속사 포함
        has_subject_particle = ("은" in part_before_quote or "는" in part_before_quote)
        has_conjunction = any(conj in part_before_quote for conj in cls.SEQUENTIAL_CONJUNCTIONS)
        is_short_phrase = len(part_before_words) <= 5
        
        is_conjunction_subject_pattern = (has_subject_particle and has_conjunction and is_short_phrase)
        
        # 조건 3: 뒤 부분이 단일 동사로만 이루어짐
        is_single_verb = len(part_after_quote.split(" ")) <= 2
        
        return (is_conjunction_with_subject or is_conjunction_subject_pattern) and is_single_verb

    @classmethod
    def is_case_3(cls, part_before_quote, part_after_quote):
        """Case 3: 대명사 주어(그는/그녀는) + 단일 동사 패턴
        
        큰따옴표 앞 부분이 "그는" 또는 "그녀는"만 있고,
        큰따옴표 뒤 부분이 단일 동사로만 이루어진 경우
        
        예: "그는 말했다" -> part_before_quote="그는", part_after_quote="말했다"
        
        Args:
            part_before_quote (str): 큰따옴표 앞 부분
            part_after_quote (str): 큰따옴표 뒤 부분
            
        Returns:
            bool: Case 3 조건을 만족하면 True
        """
        if not part_before_quote or not part_after_quote:
            return False

        # 조건 1: 앞 부분이 대명사 주어만 존재
        is_pronoun_subject_only = (part_before_quote == "그는" or part_before_quote == "그녀는")
        
        # 조건 2: 뒤 부분이 단일 동사로만 이루어짐
        is_single_verb = len(part_after_quote.split(" ")) <= 2
        
        return is_pronoun_subject_only and is_single_verb

    @classmethod
    def is_case_4(cls, part_before_quote, part_after_quote):
        """Case 4: 주어(은/는)만 + 단일 동사 패턴
        
        큰따옴표 앞 부분이 주어(은/는)로만 끝나고 (4단어 이하),
        큰따옴표 뒤 부분이 단일 동사로만 이루어진 경우
        
        예: "홍길동 의원은 말했다" -> part_before_quote="홍길동 의원은", part_after_quote="말했다"
        
        Args:
            part_before_quote (str): 큰따옴표 앞 부분
            part_after_quote (str): 큰따옴표 뒤 부분
            
        Returns:
            bool: Case 4 조건을 만족하면 True
        """
        if not part_before_quote or not part_after_quote:
            return False
        
        # 조건 1: 앞 부분이 4단어 이하이고 "은" 또는 "는"으로 끝남
        is_subject_only = (
            len(part_before_quote.split(" ")) <= 4 and
            (part_before_quote.endswith("은") or part_before_quote.endswith("는"))
        )
        
        # 조건 2: 뒤 부분이 단일 동사로만 이루어짐
        is_single_verb = len(part_after_quote.split(" ")) <= 2
        
        return is_subject_only and is_single_verb

    @classmethod
    def is_case_5(cls, part_before_quote, part_after_quote):
        """Case 5: 공백 + 단일 동사 패턴
        
        큰따옴표 앞 부분이 공백이고,
        큰따옴표 뒤 부분이 단일 동사로만 이루어진 경우
        
        예: ""말했다" -> part_before_quote="", part_after_quote="말했다"
        
        Args:
            part_before_quote (str): 큰따옴표 앞 부분
            part_after_quote (str): 큰따옴표 뒤 부분
            
        Returns:
            bool: Case 5 조건을 만족하면 True
        """
        if not part_after_quote:
            return False

        # 조건 1: 앞 부분이 공백
        is_before_empty = not part_before_quote
        
        # 조건 2: 뒤 부분이 단일 동사로만 이루어짐
        is_single_verb = len(part_after_quote.split(" ")) <= 2
        
        return is_before_empty and is_single_verb

    @classmethod
    def check_cases(cls, text, paragraph, previous_quoted_speeches):
        """발언 병합 가능 여부를 확인하는 메인 함수
        
        다음 조건들을 종합하여 병합 가능 여부를 판단합니다:
        1. 5가지 구조적 케이스 중 하나를 만족 (접속사, 주어, 동사 패턴)
        2. 이전 발언과 문맥적으로 연결되어 있음 (case_base)
        3. 예외 접속사가 아님 (is_exceptional_conjunction)
        4. 또는 같은 문장 내에 이전 발언이 포함됨 (case_same_sentence)
        
        Args:
            text (str): 확인할 텍스트 (큰따옴표 포함)
            paragraph (str): 현재 문단
            previous_quoted_speeches (list): 이전 발언문 리스트
            
        Returns:
            bool: 병합 가능하면 True, 아니면 False
        """
        # 큰따옴표를 기준으로 텍스트 분리
        part_before_quote, part_after_quote = cls.split_text_by_quotes(text)

        # 5가지 구조적 케이스 확인
        # Case 1: 접속사만 + 단일 동사
        is_case_1 = cls.is_case_1(part_before_quote, part_after_quote)
        # Case 2: 접속사 + 주어(은/는) + 단일 동사
        is_case_2 = cls.is_case_2(part_before_quote, part_after_quote)
        # Case 3: "그는"만 + 단일 동사
        is_case_3 = cls.is_case_3(part_before_quote, part_after_quote)
        # Case 4: 주어(은/는)만 + 단일 동사
        is_case_4 = cls.is_case_4(part_before_quote, part_after_quote)
        # Case 5: 공백 + 단일 동사
        is_case_5 = cls.is_case_5(part_before_quote, part_after_quote)
        
        # 문맥적 연결 확인
        is_contextually_connected = cls.case_base(paragraph, text, previous_quoted_speeches)
        
        # 예외 접속사 확인 (이에 대해 등은 병합하지 않음)
        is_not_exceptional = cls.is_exceptional_conjunction(part_before_quote)
        
        # 같은 문장 내 포함 확인
        is_in_same_sentence = cls.case_same_sentence(paragraph, text, previous_quoted_speeches)

        # 병합 가능 조건:
        # (구조적 케이스 중 하나 + 문맥적 연결 + 예외 아님) 또는 같은 문장 내 포함
        structural_case_matched = (is_case_1 or is_case_2 or is_case_3 or is_case_4 or is_case_5)
        can_merge = ((structural_case_matched and is_contextually_connected and is_not_exceptional) 
                     or is_in_same_sentence)
        
        return can_merge


# 하위 호환성을 위한 함수들
def extract_quotes(text, name):
    processor = TextProcessor()
    return processor.extract_quotes(text, name)

def simplify_purpose(sentence, name):
    processor = TextProcessor()
    return processor.simplify_purpose(sentence, name)

def to_string(text):
    processor = TextProcessor()
    return processor.to_string(text)

def find_sequential_conjunction(sentence):
    processor = TextProcessor()
    return processor.find_sequential_conjunction(sentence)

def add_comma_after_target_words(sentence):
    processor = TextProcessor()
    return processor.add_comma_after_target_words(sentence)

def filter_sentences_by_name(sentences, keywords):
    processor = TextProcessor()
    return processor.filter_sentences_by_name(sentences, keywords)

def split_preserving_quotes(text):
    processor = TextProcessor()
    return processor.split_preserving_quotes(text)

def normalize_spaces_inside_single_quotes(text):
    processor = TextProcessor()
    return processor.normalize_spaces_inside_single_quotes(text)

def extract_and_clean_quotes(text):
    processor = TextProcessor()
    return processor.extract_and_clean_quotes(text)

def split_sentences_by_comma(text):
    processor = TextProcessor()
    return processor.split_sentences_by_comma(text)

def merge_tokens(ner_results):
    processor = TextProcessor()
    return processor.merge_tokens(ner_results)

def extract_speaker(text):
    processor = TextProcessor()
    return processor.extract_speaker(text)

def calculate_similarity(sentence1, sentence2, criteria):
    processor = TextProcessor()
    return processor.calculate_similarity(sentence1, sentence2, criteria)

def normalize_text(text):
    processor = TextProcessor()
    return processor.normalize_text(text)

def is_valid_speaker_by_josa(speakers, sentence):
    processor = TextProcessor()
    return processor.is_valid_speaker_by_josa(speakers, sentence)