import re
from text_manager import nlp
from transformers import pipeline

# 2. 토씨 또는 어미의 조정
replace_dict = {
    r'(\S+)에선': r'\1에서',
    r'(\S+)에서는': r'\1에서',
    r'(\S+을) 두곤': r'\1 두고',
    r'(\S+을) 두고는': r'\1 두고',
    r'(\S+에) 대해선': r'\1 대해',
    r'(\S+에) 대해서는': r'\1 대해',
    # 추가 패턴 계속 작성 가능
}

# 3. 발언문장 서두에 나오는 [이에, 이에 대해, 이같이 말하며, 반면] 등의 문구는 발언의 목적배경취지에 쓰지 않음.
# 4. 발언문장 서두에 나오는 [이어, 그러므로, 또] 등의 행 합치기 대상인 접속사는 발언의 목적배경취지에 쓰지 않음.
# 9. 문장 서두의 [이에, 이에 대해] 는 1안) 쓰지 않거나,  2안) 앞 문장에서 '이에 대해' 에 해당되는 내용을 찾아서 씀
unimportant_conjunctions = [
    # 3
    "이에 대해",
    "이에",
    "이같이 말하며",
    "반면",
    "이를 두고",
    

    # 4
    "이어",
    "그러므로",
    "또",
    
    
    # 발언 정리 진행방향 제안
    "그러나",
    "그러자",
    "그러면서",
    "이어서",
    "다만",
    "아울러"
    ]

# 6. OOO 의원은 "….." 고 했다, 말했다  --> OOO 의원의 발언
# 7. OOO 의원은 "….." 고 비판, 주장, 비난, 반박했다 등  --> OOO 의원의 비판, 주장, 비난, 반박 등

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
    "반문했다": "반문",

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
    "내용의 게시": "내용의 글을 게시",

    # 샘플개선 4 검토의견 5. 발언의 목적 배경 취지
    "하기도 했다": "했다",
    " 한 것": " 발언한 것",
    
    # 샘플개선 4 3. 발언의 목적배경 취지
    "페이스북에 시작하는 게시": "페이스북에 게시",
    "대표는 하는 곳으로 검찰을 묘사했다": "대표는 검찰을 묘사했다",
    "대해선": "대해"
    # "했다": "함",
    
}

# 1. 기본적으로 큰따옴표의 전 후에 있는 문구를 그대로 옮겨 쓰되 아래의 조정이 필요
# 1. 발언자는  성+직책 or 성명+직책 or 성+호칭 or 성멍+호칭 or 대명사 등 발언문장에 표현된 그대로 옮겨 씀
# 5. 큰따옴표 바로 뒤에 붙어있는 [며, 라며, 이라며, 고, 라고, 이라고] 등의 단어는 발언의 목적배경취지에 쓰지 않음
# 추가. 두 개의 큰따옴표 문장 입력 시 하나의 문구만 채택


def remove_quotes(text):
    
    text = text.replace("“", "\"")
    text = text.replace("”", "\"")
    text = text.replace("‘", "\'")
    text = text.replace("’", "\'")
    text = text.replace("\" \"", "\", \"") # 쌍따옴표 문장 두 개가 연속한 경우 처리

    # 쌍따옴표 안 내용 + 바로 뒤에 붙은 한 단어(띄어쓰기 포함해서 최대 2글자 정도)까지 제거
    pattern = r'"[^"]*"(?:\s*\S+)?'

    cleaned_text = re.sub(pattern, '', text)
    # 여러 공백을 하나로, 앞뒤 공백 제거
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

# 2. 토씨 또는 어미의 조정
# 11. [ …...  "……..." 는+명사 …......]  의 문장 구조 : 큰따옴표 문장이 바로 뒤에 나오는 명사를 수식하는 문구가 되어있는 형태


def adjust_particles_and_endings(text):
    doc = nlp(text)
    # print(doc)
    words = []

    # 주어 + 은/는 → 주어 + 이/가 조정
    for sent in doc.sentences:
        size = 0
        try:
            size = len(doc.sent.words)
            # print(size)
        except:
            size = 0
        for i, word in enumerate(sent.words):
            # 기본 형태 저장
            new_word = list(word.text)

            # 조사 '은' or '는'인 경우
            if word.upos == 'NOUN' and word.xpos in ["ncn+jxt", "ncn+ncn+jxt"] and word.text.endswith(("은", "는")):
                if word.id < size - 3 and sent.words[word.id].text.startswith(("입장", "취지")):
                    new_word[-1] = ""
                else:
                    # '은' → '이', '는' → '가'
                    if word.text.endswith("은"):
                        new_word[-1] = "이"
                    else:
                        new_word[-1] = '가'

            new_word = "".join(new_word)  # 리스트 → 문자열
            words.append(new_word)

    adjusted_text = ' '.join(words)

    # 이후 어미 교정: 정규표현식으로 치환
    for pattern, repl in replace_dict.items():
        adjusted_text = re.sub(pattern, repl, adjusted_text)

    # 불필요한 공백 처리
    adjusted_text = adjusted_text.replace(" .", ".")
    adjusted_text = adjusted_text.replace(" ,", ",")
    adjusted_text = re.sub(r'\s+', ' ', adjusted_text).strip()

    return adjusted_text

# 3. 발언문장 서두에 나오는 [이에, 이에 대해, 이같이 말하며, 반면] 등의 문구는 발언의 목적배경취지에 쓰지 않음.
# 4. 발언문장 서두에 나오는 [이어, 그러므로, 또] 등의 행 합치기 대상인 접속사는 발언의 목적배경취지에 쓰지 않음.
# 9. 문장 서두의 [이에, 이에 대해] 는 1안) 쓰지 않거나,  2안) 앞 문장에서 '이에 대해' 에 해당되는 내용을 찾아서 씀


def exclude_conjunctions(text):
    for conjunction in unimportant_conjunctions:
        if text.startswith(conjunction):
            # print(conjunction + " 치환")
            text = text.replace(conjunction, "", 1)

    return re.sub(r'\s+', ' ', text).strip()

# 6. OOO 의원은 "….." 고 했다, 말했다  --> OOO 의원의 발언
# 7. OOO 의원은 "….." 고 비판, 주장, 비난, 반박했다 등  --> OOO 의원의 비판, 주장, 비난, 반박 등


def simplify_purpose(sentence, name):
    """
    문장에서 대체 가능한 표현을 간소화.
    """
    
    for key, value in replacement_dict.items():
        if key in sentence:
            sentence = sentence.replace(key, value)
            # print(f"{key} -> {value} : {sentence}")
    if sentence in ["발언", "했다", "그는 발언"]:
        sentence = f"{name}의 발언"
    elif sentence in ["물었다"]:
        sentence = f"{name}의 질문"
        
    return sentence

class MediaMentionCleaner:
    def __init__(self):
        self.media_names = [
            "본지", "중앙일보", "조선일보", "동아일보", "세계일보",
            "MBC", "KBS", "SBS", "JTBC", "채널A", "TV조선", "연합뉴스", "CBS"
        ]
        self.programs = [
            "김현정의 뉴스쇼", "배성규의 정치펀치", "조선일보 유튜브"
        ]

        self.patterns = [
            # 특정 방송 프로그램명 + 인터뷰/출연
            rf"[‘\"']?\s*({'|'.join(self.programs)})\s*[’\"']?\s*(인터뷰|출연)?\s*(에서|에)?",
            # 언론사 + 연결어 + 맥락 표현
            rf"({'|'.join(self.media_names)})\s*(유튜브)?\s*([와과]의|[와과]|의)?\s*[\w\s]*?(통화|인터뷰|라디오|방송|만남|출연|강조|만난)?\s*(에서|에|당시)?",
            # 단독 언론사 이름만 나올 경우
            rf"({'|'.join(self.media_names)})"
        ]

    def clean(self, text: str) -> str:
        for pattern in self.patterns:
            text = re.sub(pattern, "", text)
        # 조사 혼자 남은 것 제거
        text = re.sub(r"\s+(에서|에|와의|와|의)\s+", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()
    

def restore_speaker(text: str, name: str) -> str:
    POSITION_SUFFIXES = ["의원은", "대표는", "장관은", "총장은", "위원장은"]
    if not name:
        return text

    target_surname = name[0]

    # 직책 표현 패턴
    position_pattern = "|".join(map(re.escape, POSITION_SUFFIXES))
    full_pattern = rf'([\w가-힣]+ ({position_pattern}))'

    matches = list(re.finditer(full_pattern, text))
    if len(matches) < 2:
        return text

    first, second = matches[0], matches[1]
    between = text[first.end():second.start()]
    is_contiguous = re.fullmatch(r'[\s\W]*', between)

    if not is_contiguous:
        return text  # 연속 아님

    first_name = first.group(1).split()[0]
    second_name = second.group(1).split()[0]
    
    if first_name[0] != target_surname:
        start, end = first.start(), first.end()
        return (text[:start] + text[end:]).strip()

    if second_name[0] != target_surname:
        start, end = second.start(), second.end()
        return (text[:start] + text[end:]).strip()

    return text


def extract_purpose(name=None, title=None, body1=None, body2=None, prev=None):
    # 1. 기본적으로 큰따옴표의 전 후에 있는 문구를 그대로 옮겨 쓰되 아래의 조정이 필요
    # 1. 발언자는  성+직책 or 성명+직책 or 성+호칭 or 성멍+호칭 or 대명사 등 발언문장에 표현된 그대로 옮겨 씀
    # 5. 큰따옴표 바로 뒤에 붙어있는 [며, 라며, 이라며, 고, 라고, 이라고] 등의 단어는 발언의 목적배경취지에 쓰지 않음
    cleaned_text = remove_quotes(body1)
    # print("1단계: " + cleaned_text)
    
    # 두 개의 주어가 연속해서 나오는 경우 처리
    restored_speaker_text = restore_speaker(cleaned_text, name)

    # 2. 토씨 또는 어미의 조정
    # 11. [ …...  "……..." 는+명사 …......]  의 문장 구조 : 큰따옴표 문장이 바로 뒤에 나오는 명사를 수식하는 문구가 되어있는 형태
    adjusted_text = adjust_particles_and_endings(restored_speaker_text)
    # print("2단계: " + adjusted_text)

    # 3. 발언문장 서두에 나오는 [이에, 이에 대해, 이같이 말하며, 반면] 등의 문구는 발언의 목적배경취지에 쓰지 않음.
    # 4. 발언문장 서두에 나오는 [이어, 그러므로, 또] 등의 행 합치기 대상인 접속사는 발언의 목적배경취지에 쓰지 않음.
    # 9. 문장 서두의 [이에, 이에 대해] 는 1안) 쓰지 않거나,  2안) 앞 문장에서 '이에 대해' 에 해당되는 내용을 찾아서 씀
    excluded_text = exclude_conjunctions(adjusted_text)
    # print("3단계: " + excluded_text)
    
    # cleaner = MediaMentionCleaner()
    # cleaned = cleaner.clean(excluded_text)
    cleaned = excluded_text

    # 6. OOO 의원은 "….." 고 했다, 말했다  --> OOO 의원의 발언
    # 7. OOO 의원은 "….." 고 비판, 주장, 비난, 반박했다 등  --> OOO 의원의 비판, 주장, 비난, 반박 등
    simplified_text = simplify_purpose(cleaned, name)
    # print("목적배경취지: " + simplified_text)

    return simplified_text


if __name__ == "__main__":
    name = "김현정"
    # text = '이에 대해 박기찬 의원은 "밥이 맛없다"라는 발언에 대해선 "밥 좀 맛있게 해라"고 지적했고, 김동욱 전 회장은 "그게 무슨소리냐"고 반문했다.'
    # text = "박기찬 의원은 말했다"
    text = """
    고민정 최고위원은 이날 오전 CBS 라디오 ‘김현정의 뉴스쇼’ 인터뷰에서 "불체포 특권을 포기하자는 혁신위 제안과 이재명 대표의 발언을 번복하자는 의미냐고 묻고 싶다"며 "한 번 내뱉은 말에 대해서는 당연히 약속을 지키는 게 정치 아닌가"라고 말했다.
    """
    # print(text.startswith("이에 대해"))
    print(extract_purpose(name=name, body1=text))
  