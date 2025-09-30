import re
import stanza
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from collections import defaultdict

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
        self._init_ner_model()
        self.replacement_dict = self._init_replacement_dict()
        self.sequential_conjunctions = self._init_sequential_conjunctions()
        self.exceptional_conjunctions = ["이에 대해"]
    
    def _init_ner_model(self):
        """NER 모델 초기화"""
        model_name = "KPF/KPF-bert-ner"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner_pipeline = pipeline(
            "ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")
    
    def _init_replacement_dict(self):
        """동사 표현 간소화를 위한 딕셔너리 초기화"""
        return {
            "지적했다": "지적", "강조했다": "강조", "반발했다": "반발", "말했다": "발언",
            "비판했다": "비판", "글을 올렸다": "게시", "썼다": "게시", "주장했다": "주장",
            "의심했다": "의심", "캐물었다": "질문", " 물었다": " 질문", "단정했다": "단정",
            "언급한 바 있다": "언급", "촉구했다": "촉구", "설명을 덧붙였다": "설명",
            "문제를 제기했다": "제기", "우려를 나타냈다": "우려", "확인을 요청했다": "요청",
            "지적을 가했다": "지적", "언급했다": "언급", "반박했다": "반박", "반문했다": "반문",
            "입장을 밝혔다": "입장", "강력히 주장했다": "주장", "중요성을 강조했다": "강조",
            "목소리를 냈다": "의견", "찬성 의견을 밝혔다": "찬성", "반대를 표명했다": "반대",
            "찬사를 보냈다": "찬사", "동의를 표했다": "동의", "결론을 내렸다": "결론",
            "평가를 내렸다": "평가", "확실히 했다": "확정", "입장을 표명했다": "입장",
            "질문을 던졌다": "질문", "알려진 바 있다": "알려짐", "조치를 취했다": "조치",
            "약속을 지켰다": "이행", "찬성을 표명했다": "찬성", "환영을 표했다": "환영",
            "감사를 전했다": "감사", "공로를 치하했다": "치하", "지지를 보냈다": "지지",
            "격려의 말을 전했다": "격려", "승인을 전했다": "승인", "축하를 전했다": "축하",
            "호평을 전했다": "호평", "정보를 제공했다": "제공", "의견을 나눴다": "의견",
            "상황을 공유했다": "공유", "해결책을 제안했다": "제안", "문제를 설명했다": "설명",
            "진행 상황을 알렸다": "보고", "변화를 요구했다": "요구", "의미를 전달했다": "전달",
            "근거를 제시했다": "근거", "조언을 요청했다": "조언", "비난을 가했다": "비난",
            "사과를 요구했다": "사과 요구", "잘못을 지적했다": "지적", "논란을 제기했다": "논란 제기",
            "불신을 드러냈다": "불신", "비판의 목소리를 냈다": "비판", "고발을 진행했다": "고발",
            "항의를 표했다": "항의", "불만을 드러냈다": "불만", "의혹을 제기했다": "의혹",
            "결과를 발표했다": "발표", "합의를 도출했다": "합의", "대화를 요청했다": "요청",
            "호소를 전했다": "호소", "결정권을 주장했다": "주장", "합의안을 제시했다": "제안",
            "필요성을 강조했다": "강조", "중요성을 지적했다": "지적", "타협을 제안했다": "타협",
            "문제를 제시했다": "제시", "대안을 주장했다": "주장", "목표를 강조했다": "강조",
            "이점을 설명했다": "설명", "문제를 고발했다": "고발", "해결책을 주장했다": "주장",
            "우려를 표명했다": "우려", "입장을 정리했다": "정리", "입장을 조율했다": "조율",
            "목표를 제시했다": "제시", "요청을 전달했다": "요청", "마이크를 했다": "마이크를 잡고 발언",
            "마이크를 했고": "마이크를 잡고 발언", "밝혔다": "밝힘", "요청했다": "요청",
            "열기도 했다": "열기도 함", "제안했다": "제안", "언성을 높였고": "언성을 높여 발언",
            "라고도 했다": "대해 발언", "맞았다": "맞음", "제기했다": "제기", "보탰다": "보탬",
            "질문에 했다": "질문에 답변", "것에 밝혔다": "것에 대해 발언", "발언 했다": "발언",
            "관련해선": "관련해서", "많다 지적에": "많다는 지적에", "소감문을 밝혔다": "소감문을 통해 밝힘",
            " 고 발언": "발언", "호소했다": "호소", "입장문을 이라고 발언": "입장문을 통해 발언",
            "는 내용의 기자회견": "기자회견", "비난했다": "비난", "덧붙였다": "덧붙임",
            "진행 중단했다": "진행하려다가 일단 중단", " 고 했다": "발언", " 라고 했다": "발언",
            "검토에 했다": "검토에 대해 발언", "따져물었다": "발언", "평가했다": "평가",
            "말했지만": "발언", "질의했다": "질문", "역설하기도 했다": "역설함",
            "고 목소리를 높였다": "목소리를 높여 발언", " 했다": " 발언", "내용의 게시": "내용의 글을 게시",
            "하기도 했다": "했다", " 한 것": " 발언한 것", "페이스북에 시작하는 게시": "페이스북에 게시",
            "대표는 하는 곳으로 검찰을 묘사했다": "대표는 검찰을 묘사했다", "대해선": "대해"
        }
    
    def _init_sequential_conjunctions(self):
        """순접 접속사 리스트 초기화"""
        return [
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

    def extract_quotes(self, text, name):
        """문단에서 큰따옴표로 묶인 발언을 추출하고, 특정 인물의 이름이 포함된 것만 반환
        
        Args:
            text (str): 추출할 텍스트
            name (str): 발언자 이름
            
        Returns:
            str: 추출된 발언문들을 공백으로 연결한 문자열
        """
        text = text.replace(""", "\"").replace(""", "\"").replace("'", "'")
        quotes = []
        start = 0

        while start < len(text):
            start = text.find('"', start)
            if start == -1:
                break
            end = text.find('"', start + 1)
            if end == -1:
                break

            extracted_sentence = f'"{text[start + 1:end]}"'
            quotes.append(extracted_sentence)
            start = end + 1

        return "  ".join(quotes)

    def simplify_purpose(self, sentence, name):
        """문장에서 대체 가능한 표현을 간소화"""
        for key, value in sorted(self.replacement_dict.items(), key=lambda x: len(x[0]), reverse=True):
            if key in sentence:
                sentence = sentence.replace(key, value)
        
        if sentence in ["발언", "했다"]:
            sentence = f"{name}의 발언"
        elif sentence in ["물었다"]:
            sentence = f"{name}의 질문"
        return sentence

    def to_string(self, text):
        """None값이나 숫자 등 문자열이 아닌 값을 문자열로 변경"""
        if text is None:
            text = ""
        elif not isinstance(text, str):
            text = str(text)
        return text

    def find_sequential_conjunction(self, sentence):
        """문장에서 처음 5개 단어를 확인하고 접속사 탐지"""
        words = sentence.split()
        first_five_words = words[:5]

        for word in first_five_words:
            if word in self.sequential_conjunctions:
                return word
        return None

    def add_comma_after_target_words(self, sentence):
        """특정 단어(말했지만, 하지만 등) 뒤에 ','를 추가하는 함수"""
        target_words = ["말했지만", "하지만", "그러나", "그럼에도", "그렇지만", "다만", "반면에", "한편", "그렇다면"]
        pattern = r'\b(' + '|'.join(target_words) + r')\b\s*'
        modified_sentence = re.sub(pattern, r'\1, ', sentence)
        return modified_sentence

    def filter_sentences_by_name(self, sentences, keywords):
        """특정 이름이나 성이 포함된 문장 조각(컴마 기준) 또는 순접 접속사가 포함된 문장 조각만 필터링"""
        filtered_sentences = []

        for sentence in sentences:
            sentence = sentence.replace(""", '"').replace(""", '"').replace("'", "'").replace("'", " '")
            sentence = self.add_comma_after_target_words(sentence)

            pattern = r',(?=(?:[^\'"]*[\'"][^\'"]*[\'"])*[^\'"]*$)'
            sentence_parts = re.split(pattern, sentence)
            filtered_parts = []

            for part in sentence_parts:
                part = part.strip()
                starts_with_quote = part.startswith('"')
                contains_keyword = any(keyword in part for keyword in keywords)
                contains_conjunction = self.find_sequential_conjunction(part)

                if starts_with_quote or contains_keyword or contains_conjunction:
                    filtered_parts.append(part)

            if filtered_parts:
                filtered_sentences.append(', '.join(filtered_parts))

        return filtered_sentences

    def split_preserving_quotes(self, text):
        """따옴표 안의 마침표를 무시하면서 문장을 나누는 함수"""
        pattern = r'\.(?=(?:[^\'"]*["\'][^\'"]*["\'])*[^\'"]*$)'
        sentences = re.split(pattern, text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def normalize_spaces_inside_single_quotes(self, text):
        """작은따옴표 안의 공백을 정리하는 함수"""
        return re.sub(r"'(.*?)'", lambda m: "'{}'".format(re.sub(r"\s+", " ", m.group(1).strip())), text)

    def extract_and_clean_quotes(self, text):
        """텍스트에서 쌍따옴표로 묶인 문장을 추출하고 원래 문단에서 제거"""
        text = text.replace(""", "\"").replace(""", "\"").replace("'", "'").replace("'", "'").replace("'", "'")
        quotes = re.findall(r'"(.*?)"', text)
        cleaned_text = re.sub(r'"(.*?)"', '""', text).strip()
        return quotes, cleaned_text

    def split_sentences_by_comma(self, text):
        """쉼표를 기준으로 문장을 분리하되, 큰따옴표 및 작은따옴표 내부의 쉼표는 무시"""
        text = text.replace(""", '"').replace(""", '"').replace("'", "'").replace("'", "'")

        result = []
        current_sentence = []
        in_single_quote = False
        in_double_quote = False

        for char in text:
            if char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            elif char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote

            if char == ',' and not in_single_quote and not in_double_quote:
                result.append("".join(current_sentence).strip())
                current_sentence = []
            else:
                current_sentence.append(char)

        if current_sentence:
            result.append("".join(current_sentence).strip())

        return result

    def merge_tokens(self, ner_results):
        """BERT 토큰을 하나의 단어로 합치는 후처리 함수"""
        merged = []
        current_word = ""

        for entity in ner_results:
            word = entity["word"]
            if word.startswith("##"):
                current_word += word[2:]
            elif entity['entity_group'] == "LABEL_35":
                current_word += (" " + word)
            else:
                if current_word:
                    merged.append(current_word)
                current_word = word

        if current_word:
            merged.append(current_word)

        return merged

    def extract_speaker(self, text):
        """NER을 사용하여 발언자 추출
        
        Args:
            text (str): 발언자 추출할 텍스트
            
        Returns:
            list: 추출된 발언자 정보 리스트
        """
        ner_results = self.ner_pipeline(text)
        speakers = [entity for entity in ner_results if entity['entity_group'] in [
            "LABEL_96", "LABEL_185", "LABEL_187", "LABEL_246"]]

        for i, entity in enumerate(ner_results[:-1]):
            if (entity['word'] == "이" and ner_results[i + 1]['entity_group'] == "LABEL_35"):
                speakers.append({
                    "word": "이",
                    "entity_group": "LABEL_96",
                    "start": entity['start'],
                    "end": entity['end']
                })

        return speakers

    def calculate_similarity(self, sentence1, sentence2, criteria):
        """두 문장을 단어 단위로 비교하여 유사도 계산"""
        words1 = set(sentence1.split())
        words2 = set(sentence2.split())
        if not words1 or not words2:
            return False

        common_words = words1 & words2

        if criteria == "max":
            total_words = max(len(words1), len(words2))
            sim_thresh = 0.7
        else:
            total_words = min(len(words1), len(words2))
            sim_thresh = 0.8

        similarity = len(common_words) / total_words if total_words > 0 else 0
        return similarity >= sim_thresh

    def normalize_text(self, text):
        """문장을 비교 전에 정규화 (공백, 특수문자 제거)"""
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def is_valid_speaker_by_josa(self, speakers, sentence):
        """주어 다음 조사 판단: '은', '는'이면 발언자 인정"""
        for speaker in speakers:
            if speaker in sentence:
                idx = sentence.find(speaker) + len(speaker)
                next_char = sentence[idx:idx+1]
                if next_char in ["은", "는", "도", "또한"]:
                    return True
                try:
                    next_word = sentence[idx:].split()[0] if len(sentence[idx:].split()) > 0 else ""
                    if next_word.endswith(("은", "는", "도", "또한")):
                        return True
                    next_word = sentence[idx:].split()[1] if len(sentence[idx:].split()) > 0 else ""
                    if next_word.endswith(("은", "는", "도", "또한")):
                        return True
                except:
                    return False
        return False


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