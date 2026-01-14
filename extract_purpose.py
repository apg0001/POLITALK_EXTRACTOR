# import re
# from text_manager import nlp
# from transformers import pipeline


# class PurposeExtractor:
#     """발언의 목적/배경/취지 추출을 담당하는 클래스
    
#     이 클래스는 발언문에서 불필요한 요소를 제거하고 핵심 목적을 추출합니다.
#     접속사 제거, 동사 표현 간소화, 언론사 언급 정리 등의 기능을 제공합니다.
#     """
    
#     def __init__(self):
#         """PurposeExtractor 초기화
        
#         텍스트 처리에 필요한 딕셔너리들과 정리 도구들을 초기화합니다.
#         """
#         self.replace_dict = self._init_replace_dict()
#         self.unimportant_conjunctions = self._init_unimportant_conjunctions()
#         self.replacement_dict = self._init_replacement_dict()
#         self.media_cleaner = MediaMentionCleaner()
    
#     def _init_replace_dict(self):
#         """토씨 또는 어미의 조정을 위한 딕셔너리"""
#         return {
#             r'(\S+)에선': r'\1에서',
#             r'(\S+)에서는': r'\1에서',
#             r'(\S+을) 두곤': r'\1 두고',
#             r'(\S+을) 두고는': r'\1 두고',
#             r'(\S+에) 대해선': r'\1 대해',
#             r'(\S+에) 대해서는': r'\1 대해',
#         }
    
#     def _init_unimportant_conjunctions(self):
#         """발언의 목적배경취지에 쓰지 않을 접속사들"""
#         return [
#             "이에 대해", "이에", "이같이 말하며", "반면", "이를 두고",
#             "이어", "그러므로", "또", "그러나", "그러자", "그러면서",
#             "이어서", "다만", "아울러"
#         ]
    
#     def _init_replacement_dict(self):
#         """동사 표현 간소화를 위한 딕셔너리"""
#         return {
#             "지적했다": "지적", "강조했다": "강조", "반발했다": "반발", "말했다": "발언",
#             "비판했다": "비판", "글을 올렸다": "게시", "썼다": "게시", "주장했다": "주장",
#             "의심했다": "의심", "캐물었다": "질문", " 물었다": " 질문", "단정했다": "단정",
#             "언급한 바 있다": "언급", "촉구했다": "촉구", "설명을 덧붙였다": "설명",
#             "문제를 제기했다": "제기", "우려를 나타냈다": "우려", "확인을 요청했다": "요청",
#             "지적을 가했다": "지적", "언급했다": "언급", "반박했다": "반박", "반문했다": "반문",
#             "입장을 밝혔다": "입장", "강력히 주장했다": "주장", "중요성을 강조했다": "강조",
#             "목소리를 냈다": "의견", "찬성 의견을 밝혔다": "찬성", "반대를 표명했다": "반대",
#             "찬사를 보냈다": "찬사", "동의를 표했다": "동의", "결론을 내렸다": "결론",
#             "평가를 내렸다": "평가", "확실히 했다": "확정", "입장을 표명했다": "입장",
#             "질문을 던졌다": "질문", "알려진 바 있다": "알려짐", "조치를 취했다": "조치",
#             "약속을 지켰다": "이행", "찬성을 표명했다": "찬성", "환영을 표했다": "환영",
#             "감사를 전했다": "감사", "공로를 치하했다": "치하", "지지를 보냈다": "지지",
#             "격려의 말을 전했다": "격려", "승인을 전했다": "승인", "축하를 전했다": "축하",
#             "호평을 전했다": "호평", "정보를 제공했다": "제공", "의견을 나눴다": "의견",
#             "상황을 공유했다": "공유", "해결책을 제안했다": "제안", "문제를 설명했다": "설명",
#             "진행 상황을 알렸다": "보고", "변화를 요구했다": "요구", "의미를 전달했다": "전달",
#             "근거를 제시했다": "근거", "조언을 요청했다": "조언", "비난을 가했다": "비난",
#             "사과를 요구했다": "사과 요구", "잘못을 지적했다": "지적", "논란을 제기했다": "논란 제기",
#             "불신을 드러냈다": "불신", "비판의 목소리를 냈다": "비판", "고발을 진행했다": "고발",
#             "항의를 표했다": "항의", "불만을 드러냈다": "불만", "의혹을 제기했다": "의혹",
#             "결과를 발표했다": "발표", "합의를 도출했다": "합의", "대화를 요청했다": "요청",
#             "호소를 전했다": "호소", "결정권을 주장했다": "주장", "합의안을 제시했다": "제안",
#             "필요성을 강조했다": "강조", "중요성을 지적했다": "지적", "타협을 제안했다": "타협",
#             "문제를 제시했다": "제시", "대안을 주장했다": "주장", "목표를 강조했다": "강조",
#             "이점을 설명했다": "설명", "문제를 고발했다": "고발", "해결책을 주장했다": "주장",
#             "우려를 표명했다": "우려", "입장을 정리했다": "정리", "입장을 조율했다": "조율",
#             "목표를 제시했다": "제시", "요청을 전달했다": "요청", "마이크를 했다": "마이크를 잡고 발언",
#             "마이크를 했고": "마이크를 잡고 발언", "밝혔다": "밝힘", "요청했다": "요청",
#             "열기도 했다": "열기도 함", "제안했다": "제안", "언성을 높였고": "언성을 높여 발언",
#             "라고도 했다": "대해 발언", "맞았다": "맞음", "제기했다": "제기", "보탰다": "보탬",
#             "질문에 했다": "질문에 답변", "것에 밝혔다": "것에 대해 발언", "발언 했다": "발언",
#             "관련해선": "관련해서", "많다 지적에": "많다는 지적에", "소감문을 밝혔다": "소감문을 통해 밝힘",
#             " 고 발언": "발언", "호소했다": "호소", "입장문을 이라고 발언": "입장문을 통해 발언",
#             "는 내용의 기자회견": "기자회견", "비난했다": "비난", "덧붙였다": "덧붙임",
#             "진행 중단했다": "진행하려다가 일단 중단", " 고 했다": "발언", " 라고 했다": "발언",
#             "검토에 했다": "검토에 대해 발언", "따져물었다": "발언", "평가했다": "평가",
#             "말했지만": "발언", "질의했다": "질문", "역설하기도 했다": "역설함",
#             "고 목소리를 높였다": "목소리를 높여 발언", " 했다": " 발언", "내용의 게시": "내용의 글을 게시",
#             "하기도 했다": "했다", " 한 것": " 발언한 것", "페이스북에 시작하는 게시": "페이스북에 게시",
#             "대표는 하는 곳으로 검찰을 묘사했다": "대표는 검찰을 묘사했다", "대해선": "대해"
#         }

#     def remove_quotes(self, text):
#         """큰따옴표 안 내용 + 바로 뒤에 붙은 한 단어까지 제거"""
#         text = text.replace(""", "\"").replace(""", "\"").replace("'", "'").replace("'", "'")
#         text = text.replace("\" \"", "\", \"")
        
#         pattern = r'"[^"]*"(?:\s*\S+)?'
#         cleaned_text = re.sub(pattern, '', text)
#         cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
#         return cleaned_text

#     def adjust_particles_and_endings(self, text):
#         """토씨 또는 어미의 조정"""
#         doc = nlp(text)
#         words = []

#         for sent in doc.sentences:
#             size = 0
#             try:
#                 size = len(doc.sent.words)
#             except:
#                 size = 0
            
#             for i, word in enumerate(sent.words):
#                 new_word = list(word.text)

#                 if word.upos == 'NOUN' and word.xpos in ["ncn+jxt", "ncn+ncn+jxt"] and word.text.endswith(("은", "는")):
#                     if word.id < size - 3 and sent.words[word.id].text.startswith(("입장", "취지")):
#                         new_word[-1] = ""
#                     else:
#                         if word.text.endswith("은"):
#                             new_word[-1] = "이"
#                         else:
#                             new_word[-1] = '가'

#                 new_word = "".join(new_word)
#                 words.append(new_word)

#         adjusted_text = ' '.join(words)

#         for pattern, repl in self.replace_dict.items():
#             adjusted_text = re.sub(pattern, repl, adjusted_text)

#         adjusted_text = adjusted_text.replace(" .", ".").replace(" ,", ",")
#         adjusted_text = re.sub(r'\s+', ' ', adjusted_text).strip()

#         return adjusted_text

#     def exclude_conjunctions(self, text):
#         """발언문장 서두의 불필요한 접속사 제거"""
#         for conjunction in self.unimportant_conjunctions:
#             if text.startswith(conjunction):
#                 text = text.replace(conjunction, "", 1)

#         return re.sub(r'\s+', ' ', text).strip()

#     def simplify_purpose(self, sentence, name):
#         """문장에서 대체 가능한 표현을 간소화"""
#         for key, value in sorted(self.replacement_dict.items(), key=lambda x: len(x[0]), reverse=True):
#             if key in sentence:
#                 sentence = sentence.replace(key, value)
        
#         if sentence in ["발언", "했다", "그는 발언"]:
#             sentence = f"{name}의 발언"
#         elif sentence in ["물었다"]:
#             sentence = f"{name}의 질문"

#         return sentence

#     def restore_speaker(self, text, name):
#         """발언자 복원"""
#         POSITION_SUFFIXES = ["의원은", "대표는", "장관은", "총장은", "위원장은"]
#         if not name:
#             return text

#         target_surname = name[0]
#         position_pattern = "|".join(map(re.escape, POSITION_SUFFIXES))
#         full_pattern = rf'([\w가-힣]+ ({position_pattern}))'

#         matches = list(re.finditer(full_pattern, text))
#         if len(matches) < 2:
#             return text

#         first, second = matches[0], matches[1]
#         between = text[first.end():second.start()]
#         is_contiguous = re.fullmatch(r'[\s\W]*', between)

#         if not is_contiguous:
#             return text

#         first_name = first.group(1).split()[0]
#         second_name = second.group(1).split()[0]

#         if first_name[0] != target_surname:
#             start, end = first.start(), first.end()
#             return (text[:start] + text[end:]).strip()

#         if second_name[0] != target_surname:
#             start, end = second.start(), second.end()
#             return (text[:start] + text[end:]).strip()

#         return text

#     def _filter_text_for_speaker(self, text, name):
#         """쉼표 등으로 분절 후, (1) 입력된 이름(전체) 또는 성+직책이 name과 일치하는 분절, (2) 발언자 패턴이 전혀 없는 분절(설명문)만 남기고 나머지는 삭제"""
#         if not name or not text:
#             return text

#         text = text.strip()
#         try:
#             segments = re.split(r'[，,;；·]\s*', text)
#         except Exception:
#             segments = [text]

#         surname = name[0]
#         position_keywords = ["의원", "대표", "장관", "위원장", "차관", "국장", "총장", "전"]
#         pos_rx = r'(?:' + '|'.join(map(re.escape, position_keywords)) + r')'

#         def is_target_segment(seg):
#             # 전체 이름이 정확히 포함
#             if name in seg:
#                 return True
#             # 성+직책 패턴이 정확히 포함
#             if re.search(rf'{re.escape(name)}\s*{pos_rx}', seg):
#                 return True
#             if re.search(rf'{re.escape(surname)}[가-힣]{{1,2}}\s*{pos_rx}', seg):
#                 # 예: 윤 의원, 윤건영 의원 등
#                 return True
#             return False

#         def has_any_speaker_pattern(seg):
#             # 전체 이름 또는 성+직책 패턴이 있으면 True
#             if name in seg:
#                 return True
#             if re.search(rf'{re.escape(name)}\s*{pos_rx}', seg):
#                 return True
#             if re.search(rf'{re.escape(surname)}[가-힣]{{1,2}}\s*{pos_rx}', seg):
#                 return True
#             return False

#         result_segments = []
#         for seg in segments:
#             seg = seg.strip()
#             if not seg:
#                 continue
#             if is_target_segment(seg):
#                 result_segments.append(seg)
#             elif not has_any_speaker_pattern(seg):
#                 # 발언자 패턴이 전혀 없는 일반 설명문도 남김
#                 result_segments.append(seg)
#             # else: 다른 사람 발언자 패턴이 있는 분절은 삭제

#         # 만약 아무것도 남지 않으면 원본 반환(방어적)
#         if not result_segments:
#             return text
#         return ' '.join(result_segments)

#     def extract_purpose(self, name=None, title=None, body1=None, body2=None, prev=None):
#         """발언의 목적/배경/취지 추출 메인 함수
        
#         Args:
#             name (str): 발언자 이름
#             title (str): 기사 제목
#             body1 (str): 발언문 본문
#             body2 (str): 추가 본문 (사용하지 않음)
#             prev (str): 이전 발언 (사용하지 않음)
            
#         Returns:
#             str: 추출된 발언의 목적/배경/취지
#         """
#         cleaned_text = self.remove_quotes(body1)

#         # 다수 발언자가 한 문장에 있는 경우, 먼저 해당 발언자(name)에 해당하는 분절만 남긴다.
#         filtered_text = self._filter_text_for_speaker(cleaned_text, name)

#         # 발언자 관련 불필요한 앞뒤 문구 제거
#         restored_speaker_text = self.restore_speaker(filtered_text, name)
#         adjusted_text = self.adjust_particles_and_endings(restored_speaker_text)
#         excluded_text = self.exclude_conjunctions(adjusted_text)
#         cleaned = self.media_cleaner.clean(excluded_text)
#         simplified_text = self.simplify_purpose(cleaned, name)

#         return simplified_text


# class MediaMentionCleaner:
#     """언론사 및 방송 프로그램 언급 정리 클래스
    
#     발언문에서 언론사나 방송 프로그램 언급을 제거하여 핵심 내용만 남깁니다.
#     """
    
#     def __init__(self):
#         self.media_names = [
#             "본지", "중앙일보", "조선일보", "동아일보", "세계일보",
#             "MBC", "KBS", "SBS", "JTBC", "채널A", "TV조선", "연합뉴스", "CBS"
#         ]
#         self.programs = [
#             "김현정의 뉴스쇼", "배성규의 정치펀치", "조선일보 유튜브" "김종배의 시선집중"
#         ]

#         # 좀 더 포괄적으로 언론/프로그램 + 동사(출연/통화/인터뷰 등)를 제거하도록 패턴 확장
#         prog = '|'.join(map(re.escape, self.programs))
#         medias = '|'.join(map(re.escape, self.media_names))
#         self.patterns = [
#             rf"['\"]?\s*(?:{prog})\s*['\"]?\s*(?:인터뷰|출연|통화)?\s*(?:에서|에)?",
#             rf"(?:{medias})\s*(?:라디오|TV|유튜브)?\s*(?:에서|에)?\s*(?:출연해|출연했다|출연하여|출연|통화|인터뷰|방송)?",
#             rf"(?:{medias})"
#         ]

#     def clean(self, text):
#         """언론사 및 방송 프로그램 언급 제거"""
#         # 대소문자 구분 없이 제거
#         for pattern in self.patterns:
#             try:
#                 text = re.sub(pattern, "", text, flags=re.IGNORECASE)
#             except re.error:
#                 # 안전성: 잘못된 정규식이면 건너뜀
#                 continue

#         # 추가적으로 '...에 출연해'와 같은 동사구를 남아있다면 제거
#         try:
#             text = re.sub(r'\b(출연해|출연했다|출연하여|출연|통화했다|인터뷰했다|출연하여)\b', '', text, flags=re.IGNORECASE)
#         except re.error:
#             pass

#         # 문법적으로 불필요한 조사나 전치 표현 정리
#         text = re.sub(r"\s+(에서|에|와의|와|의)\s+", " ", text)
#         text = re.sub(r"\s{2,}", " ", text)
#         return text.strip()


# # 하위 호환성을 위한 함수
# def extract_purpose(name=None, title=None, body1=None, body2=None, prev=None):
#     extractor = PurposeExtractor()
#     return extractor.extract_purpose(name, title, body1, body2, prev)