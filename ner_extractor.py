"""NER(개체명 인식) 관련 기능을 담당하는 클래스"""

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


class NERExtractor:
    """NER(개체명 인식) 관련 기능을 담당하는 클래스"""
    
    def __init__(self):
        """NERExtractor 초기화"""
        self._init_ner_model()
    
    def _init_ner_model(self):
        """NER 모델 초기화"""
        model_name = "KPF/KPF-bert-ner"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner_pipeline = pipeline(
            "ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")
    
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
    
    def merge_tokens(self, ner_results):
        """BERT 토큰을 하나의 단어로 합치는 후처리 함수
        
        Args:
            ner_results (list): NER 결과 리스트
            
        Returns:
            list: 병합된 토큰 리스트
        """
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