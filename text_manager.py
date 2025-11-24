import re
import stanza
from ner_extractor import NERExtractor
from text_cleaner import TextCleaner

# Stanza 한국어 모델 로드
stanza.download("ko")
nlp = stanza.Pipeline("ko", processors='tokenize,pos,lemma,depparse')


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
    
    @staticmethod
    def split_text_by_quotes(text):
        """가장 첫 번째 큰따옴표 앞까지를 part_a, 가장 마지막 큰따옴표 뒤부터를 part_c로 반환"""
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
        paragraph = paragraph.replace(""", '"').replace(""", '"').replace("'", "'").replace("'", "'")
        doc = nlp(paragraph)
        return [sentence.text for sentence in doc.sentences]

    @classmethod
    def case_base(cls, paragraph, target_sentence, prev):
        """주어진 문단에서 특정 문장의 바로 앞 문장에 큰따옴표가 포함되어 있는지 확인"""
        sentences = cls.extract_sentences(paragraph)
        target_sentence = cls.extract_sentences(target_sentence)[0]
        idx = -1

        for i, sentence in enumerate(sentences):
            if target_sentence in sentence:
                idx = i

        if idx == -1 or idx == 0:
            return False

        prev_sentence = sentences[idx - 1]
        cur_sentence = sentences[idx]

        while prev_sentence.count('"') % 2 != 0:
            try:
                prev_sentence = sentences[idx - 2] + " " + prev_sentence
                idx -= 1
            except IndexError:
                break

        if any((sent in prev_sentence) or (prev_sentence in sent) for sent in prev) or \
           any((sent in cur_sentence) or (cur_sentence in sent) for sent in prev):
            return True
        else:
            return False

    @classmethod
    def is_exceptional_conjunction(cls, part_a):
        """예외 접속사가 있는 확인"""
        if not part_a:
            return True

        exceptional_conjunctions = ["이에 대해"]
        condition_1 = part_a not in exceptional_conjunctions
        condition_2 = not any(conj in part_a for conj in exceptional_conjunctions)
        return condition_1 and condition_2

    @classmethod
    def case_same_sentence(cls, paragraph, target_sentence, prev):
        """주어진 문단에서 특정 문장의 바로 앞 문장에 큰따옴표가 포함되어 있는지 확인"""
        sentences = cls.extract_sentences(paragraph)
        target_sentence = cls.extract_sentences(target_sentence)[0]
        idx = -1

        for i, sentence in enumerate(sentences):
            if target_sentence in sentence:
                idx = i

        if idx == -1:
            return False

        cur_sentence = sentences[idx]

        if any(sent in cur_sentence for sent in prev):
            return True
        else:
            return False

    @classmethod
    def is_case_1(cls, part_a, part_c):
        """Case 1: A 파트에 접속사만 존재, C 파트는 단일 동사로만 이루어짐"""
        if not part_a or not part_c:
            return False

        sequential_conjunctions = [
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

        condition_1 = part_a in sequential_conjunctions
        condition_2 = len(part_c.split(" ")) <= 2
        return condition_1 and condition_2

    @classmethod
    def is_case_2(cls, part_a, part_c):
        """Case 2: A 파트에 접속사 + 'OOO 은, 는, 그는'으로 이루어짐, C 파트는 단일 동사로 이루어짐"""
        if not part_a or not part_c:
            return False

        sequential_conjunctions = [
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

        condition_1 = (part_a.split(" ")[0] in sequential_conjunctions and 
                      (part_a.endswith("은") or part_a.endswith("는")))
        condition_2 = (("은" in part_a or "는" in part_a) and 
                      any(conj in part_a for conj in sequential_conjunctions) and 
                      len(part_a.split(" ")) <= 5)
        condition_3 = len(part_c.split(" ")) <= 2
        return (condition_1 or condition_2) and condition_3

    @classmethod
    def is_case_3(cls, part_a, part_c):
        """Case 3: A 파트에 '그는'만 존재, C 파트는 단일 동사로만 이루어짐"""
        if not part_a or not part_c:
            return False

        condition_1 = part_a == "그는" or part_a == "그녀는"
        condition_2 = len(part_c.split(" ")) <= 2
        return condition_1 and condition_2

    @classmethod
    def is_case_4(cls, part_a, part_c):
        """Case 4: A 파트에 'OOO 은' 형태(세 글자 이상)만 포함, C 파트는 단일 동사로만 이루어짐"""
        if not part_a or not part_c:
            return False
        condition_1 = len(part_a.split(" ")) <= 4 and (part_a.endswith("은") or part_a.endswith("는"))
        condition_2 = len(part_c.split(" ")) <= 2
        return condition_1 and condition_2

    @classmethod
    def is_case_5(cls, part_a, part_c):
        """Case 5: A 파트가 공란이고, C 파트는 단일 동사로만 이루어짐"""
        if not part_c:
            return False

        condition_1 = not part_a
        condition_2 = len(part_c.split(" ")) <= 2
        return condition_1 and condition_2

    @classmethod
    def check_cases(cls, text, paragraph, prev):
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

        return ((case_1 or case_2 or case_3 or case_4 or case_5) and case_base and case_exceptional_conjunction) or case_same_sentence


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