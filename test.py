import stanza

# 한국어 파이프라인 초기화 (tokenize, pos, lemma, depparse 포함)
nlp = stanza.Pipeline(
    "ko", 
    processors='tokenize,pos,lemma,depparse', 
)

# 분석할 문장
text = """
이에 민주당 윤건영 의원은 "최대치를 봐준 이유가 무엇이냐"고 물었고 한 장관은 "과징금심의위원회가 관련 기준에 따라 했으며 특정기업에 대한 특혜는 없었다"고 답했다.
"""

# 파싱 실행
doc = nlp(text)

# 결과 출력: 각 단어의 ID, 단어, head ID, head 단어, deprel
for sent in doc.sentences:
    print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' 
           for word in sent.words], sep='\n')