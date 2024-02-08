from openai import OpenAI
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os, re, io
import requests
from bs4 import BeautifulSoup

client = OpenAI(
    # This is the default and can be omitted
    api_key=st.secrets(["OPENAI_API_KEY"])
)

def download_and_store_df(file_id, key):
    """파일을 다운로드하고 세션 상태에 저장하는 함수"""
    if key not in st.session_state:
        download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
        response = requests.get(download_url)
        df = pd.read_pickle(io.BytesIO(response.content))
        st.session_state[key] = df

# with st.spinner('총선 챗봇이 파일을 다운로드합니다..'):
#         # 파일 ID를 사용한 다운로드 URL 생성
#         file_id_선거구 = '1KJw12xawMoOd1RWOSo0ib6PkydLdwYar'
#         file_id_선거법 = '1Wc_lP14JjbOUxuUgZPwdkXDB79ccVai3'
#         download_url_선거구 = f'https://drive.google.com/uc?export=download&id={file_id_선거구}'
#         download_url_선거법 = f'https://drive.google.com/uc?export=download&id={file_id_선거법}'

#         # 파일 다운로드 및 로드 (선거구)
#         response_선거구 = requests.get(download_url_선거구)
#         df_선거구 = pd.read_pickle(io.BytesIO(response_선거구.content))

#         # 파일 다운로드 및 로드 (선거법)
#         response_선거법 = requests.get(download_url_선거법)
#         df_선거법 = pd.read_pickle(io.BytesIO(response_선거법.content))

# Pickle 파일 로딩에 with 구문 사용
# with open('2020small_district_embeddings.pkl', 'rb') as file:
#     df_선거구 = pd.read_pickle(file)

# Pickle 파일 로딩에 with 구문 사용
# with open('election_info_embeddings.pkl', 'rb') as file:
#     df_선거법 = pd.read_pickle(file)

with st.sidebar:
    st.markdown("🤖<strong>“이렇게 질문해보세요!”</strong>", unsafe_allow_html=True)
    st.markdown("""
    - 임종석은 어디서 출마했니?
    - 김은혜 후보의 출마 정보는?
    - 원희룡의 후보 정보 알려줘
    - 서울 마포갑 후보들은 누구?
    - 전북 군산에서 나온 후보들은?
    - 대전 동구의 후보 등록 상황은?
    - 서울 송파구에 어떤 선거구 있나?
    - 사전투표 날짜는 언제야?
    - 투표할 때 필요한 준비물은?
    - 투표 시 신분증을 잃어버렸다면?
    - 선거법이 정하는 국회의원 후보 자격은?
    - 선거운동 기간 후보의 금품수수는 어떤 처벌 받나?
    - 선거에서 딥페이크 영상을 사용하면?
    - 선거여론조사를 공표할 수 없는 경우는? 
    """)

# 후보자 정보를 보다 이해하기 쉬운 텍스트 포맷으로 변환
def format_candidate_info(response_json):
    items = response_json['response']['body']['items']['item']
    formatted_text = ""
    for item in items:
        formatted_text += f"이름: {item['name']} ({item['gender']}, {item['age']}세)\n"
        formatted_text += f"선거구: {item['sdName']} {item['sggName']}\n"
        formatted_text += f"정당: {item['jdName']}\n"
        formatted_text += f"직업: {item['job']}\n"
        formatted_text += f"교육: {item['edu']}\n"
        formatted_text += f"주요 경력: {item['career1']}, {item['career2']}\n"
        formatted_text += f"후보 등록일: {item['regdate']}\n"
        formatted_text += f"후보 등록상태: {item['status']}\n\n"
    return formatted_text

def get_candidate_info(sggName, sdName):
    # API 요청 주소
    url = "http://apis.data.go.kr/9760000/PofelcddInfoInqireService/getPoelpcddRegistSttusInfoInqire"
    
    # 파라미터 설정
    params = {
        "serviceKey": "klnPMx6BcJRvXEqdyooFTB4iLH1XwVLiIQPlcXxK2BGGGx7zR/R37T5SYr9a9GG3okt5Wpg63CnIJrsD6nG07g==",
        "pageNo": "1",  # 페이지 번호
        "numOfRows": "10",  # 한 페이지 결과 수
        "resultType": "json",
        "sgId": "20240410",  # 선거ID
        "sgTypecode": "2",  # 선거종류코드
        "sggName": sggName,  # 함수 입력으로 받은 선거구명
        "sdName": sdName  # 함수 입력으로 받은 시도명
    }
    
    # API 호출
    response = requests.get(url, params=params)
    
    # 응답 결과 확인
    if response.status_code == 200:
        # JSON 포맷으로 파싱
        data = format_candidate_info(response.json())
        return data
    else:
        return f"API 호출에 실패했습니다. 상태 코드: {response.status_code}"
    
def find_candidate_info(후보자_이름):
    url = f"http://info.nec.go.kr/search/searchCandidate.xhtml?searchKeyword={후보자_이름}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # "data-election-name"="20240410"을 포함하는 모든 태그 찾기
    election_infos = soup.find_all(attrs={"data-election-name": "20240410"})
    
    # 각 <div> 태그의 텍스트 내용을 리스트로 수집
    texts = [div.get_text(strip=True) for div in election_infos]
    
    # 리스트의 모든 텍스트를 하나의 문자열로 결합
    final_text = ' '.join(texts)

    # "선거정보"라는 단어가 있는 인덱스 찾기
    index_of_election_info = final_text.find("바로가기")

    # "바로가기"라는 단어 이전의 텍스트만을 남기도록 슬라이싱
    if index_of_election_info != -1:  # "바로가기"라는 단어가 존재하면
        final_text = final_text[:index_of_election_info]
    else:
        # "바로가기"라는 단어가 없으면 final_text는 변경하지 않음
        pass

    # 선거구 이름을 추출하기 위한 로직
    for election_info in election_infos:
        # 해당 태그 내에서 class="t"를 가진 태그 찾기
        # 여기서는 예시로 election_info의 형제 태그들 중에서 class="t"를 찾습니다.
        # 실제 구조에 따라 next_sibling, find_parent, find 등을 적절히 사용해야 할 수 있습니다.
        target_div = election_info.find(class_='t')
        
        if target_div:
            # 원하는 텍스트 추출
            text_parts = target_div.get_text(strip=True).split('/')
            if text_parts:
                # 마지막 부분이 원하는 텍스트
                target_text = text_parts[-1].strip()
            else:
                return "선거구 이름을 찾을 수 없습니다."

        sibling = election_info.find_next_sibling(class_='list')
            
        while sibling:
            if 'list' in sibling.get('class', []):
                list_text = sibling.get_text(strip=True)
                # "자세히보기"라는 단어 이전의 텍스트만을 남기도록 슬라이싱
                index_of_shortcut = list_text.find("자세히보기")
                if index_of_shortcut != -1:
                    list_text = list_text[:index_of_shortcut]
                    final_text += f"{list_text} "
                # 다음 형제 요소로 이동
                sibling = sibling.find_next_sibling(class_='list')
            else:
                # 'list' 클래스가 아니면 반복 중지
                break
    if final_text:
        #final_text += f"4월 총선 출마하는 선거구: {target_text}"
        return final_text.strip()
    else:
        return "지정된 선거 날짜에 해당하는 후보자 정보를 찾을 수 없습니다."    

def complete_prompt(question):
    prompt = f"""
    
    Complete the user's prompt by adding words to make it an Korean interrogative sentence that very clearly reveals the user's intent.

    Prompt: 김은혜 후보는 어디서 출마?
    Complete sentence: 김은혜 후보는 이번 총선에서 어느 선거구에서 출마했나요?

    Prompt: 서울 마포의 후보들은? 
    Complete sentence: 서울 마포에서 출마한 후보들은 누구인가요?

    Prompt: 원희룡 전 장관의 이력은?
    Complete sentence: 원희룡 전 장관의 후보 등록 이력은 무엇인가요? 

    Prompt: 사전투표는 언제?
    Complete sentence: 사전투표의 날짜는 언제인가요?

    Prompt: 투표할 때 준비물?
    Complete sentence: 선거에서 투표를 할 때 필요한 준비물은 무엇인가요?

    Prompt:  {question}
    Complete sentence:
    """

    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=40,
        temperature=1.0
    )

    # 응답에서 의도와 엔티티 추출
    response_text = response.choices[0].text.strip().strip('"')
    cleaned_question = response_text.replace("Complete", "").replace("sentence", "").replace(":", "").strip()
    response_lines = cleaned_question.split('\n')
    full_question = response_lines[0] if len(response_lines) > 0 else "질문 없음"
    
    return full_question

def classify_intent_and_entities(question):
    # 사용자 질문 분석을 위한 프롬프트 엔지니어링
    prompt = f"""
    다음은 국회의원선거(총선)에 관해 사용자들이 질문할 수 있는 질문에서 추출한 의도와 엔티티의 예시들입니다:

    질문: 임종석은 어디서 출마했니?
    의도: 후보자 정보 조회
    엔티티: 임종석

    질문: 김은혜 후보의 출마 정보는?
    의도: 후보자 정보 조회
    엔티티: 김은혜

    질문: 원희룡 후보의 공약은?
    의도: 후보자 공약 조회
    엔티티: 원희룡
    
    질문: 서울 송파에 어떤 선거구가 있나요?
    의도: 선거구 조회
    엔티티: 서울 송파

    질문: 부산 해운대 후보는?
    의도: 후보자 정보 조회
    엔티티 부산 해운대

    질문: 서울 마포에서 출마한 후보자들은 누구?
    의도: 후보자 정보 조회
    엔티티: 서울 마포

    질문: 이번 총선에 어떤 정당들이 참여하나요?
    의도: 정당 정보 조회
    엔티티: 없음

    질문: 더불어민주당의 주요 공약은 무엇인가요?
    의도: 공약 및 정책 조회
    엔티티: 더불어민주당

    질문: 투표는 어떻게 하나요?
    의도: 선거정보 및 투표 방법 안내
    엔티티: 없음

    질문: 사전투표는 언제인가요?
    의도: 선거정보 및 투표 일정 조회
    엔티티: 사전투표

    질문: 투표할 때 필요한 준비물은?
    의도: 선거정보 및 투표 규정 안내
    엔티티: 투표, 준비물

    질문: 선거법에 따른 후보자 자격은 무엇인가요?
    의도: 선거법 및 규정 안내
    엔티티: 후보자 자격

    질문: 후보가 돈 받으면 어떤 처벌을 받나요?
    의도: 부정행위 및 처벌 조회
    엔티티: 후보, 돈, 처벌

    예시를 참고해서 아래 사용자의 질문으로부터 의도와 엔티티를 분류해주세요:

    사용자의 질문: "{question}"
    의도:
    엔티티:
    """

    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=100,
        temperature=1.0
    )

    # 응답에서 의도와 엔티티 추출
    response_text = response.choices[0].text.strip().strip('"')
    lines = response_text.split('\n')
    if len(lines) > 0 and ": " in lines[0]:
        intent = lines[0].split(": ")[1]
    else:
        intent = "없음"
    if len(lines) > 1 and ": " in lines[1]:
        entity = lines[1].split(": ")[1]
    else:
        entity = "없음"
    total_tokens = response.usage.total_tokens    
    return intent, entity, total_tokens 

def choose_tool_with_gpt_detailed(question, intent):
    # 의도를 바탕으로 적절한 도구 선택을 위한 상세 설명이 포함된 프롬프트
    prompt = f"""
    The following tools, such as API calls and DB access, can help you better answer user questions:

    - 후보개인API: This is an API that allows users to search for information of requested candidate(including preliminary candidates).
    - 후보공약API: This is an API that allows users to query election pledge information of requested candidates.
    - 동네후보API: This is an API that allows users to search for candidates in the constituency or requested region.
    - 선거구DB: This is a database containing information on the names of electoral districts and administrative districts. This is essential if the user asks for a certain region.
    - 선거정보DB: This is a database that contains all information related to general elections, elections, voting, early voting, and vote counting. It also includes election laws and prohibitions.

    Depending on the user's questions and intentions, determine whether each tool is necessary or not. Categorize only those tools that you have judged as 'yes'.

    Questions: {question}
    Intent: {intent}
    Tools needed:
    """

    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=120,
        temperature=1.0
    )
    
    # 응답에서 선택된 도구 추출
    tools = response.choices[0].text.strip().strip('"')
    cleand_tools = tools.replace("Tools ", "").replace("needed", "").replace(":", "").strip()
    total_tokens = response.usage.total_tokens
    return cleand_tools, total_tokens

def process_based_on_chosen_tool(choosed_tool, user_entity, prompt):
    
    final_return = ""
    df_선거구 = st.session_state.get("df_선거구")
    df_선거법 = st.session_state.get("df_선거법")

    if "후보개인" in choosed_tool:
        # 쉼표나 빈칸으로 구분
        words = re.split(r',\s*|\s+', user_entity)
        # 각 단어를 find_candidate_info() 입력 파라미터에 넣어 호출
        results = [find_candidate_info(word.strip()) for word in words]
        final_return += f"선거관리위원회에서 {user_entity}의 예비 후보자 등록 여부를 검색한 결과입니다.\n\n"
        # 각각의 words 와 results 를 final_return 스트링에 쌍으로 append
        for word, result in zip(words, results):
            final_return += f"{word.strip()}: {result}\n\n"
    
    if "후보공약" in choosed_tool:
        # 실제 API 호출 및 데이터 처리 코드
        final_return += "현재 예비후보자 등록 기간입니다. 선거관리위원회에 후보자 공약 정보가 등록돼있지 않습니다.\n\n"
    
    if "동네후보" in choosed_tool:
        user_entity_modified = user_entity.replace(",", " ")
        user_question_embedding = client.embeddings.create(input=user_entity_modified, model='text-embedding-3-small').data[0].embedding
        user_question_embedding_2d = np.array(user_question_embedding).reshape(1, -1)

        # DataFrame의 각 문서 임베딩에 대해 유사도 계산
        df_선거구['similarities'] = df_선거구['embeddings'].apply(lambda x: cosine_similarity([x], user_question_embedding_2d)[0][0])
        most_similar_texts = df_선거구.sort_values('similarities', ascending=False).iloc[0]
        final_return += f"{most_similar_texts['시도명']} {most_similar_texts['선거구명']}에서 출마한 예비 후보자들의 정보입니다:\n\n"
        final_return += get_candidate_info(most_similar_texts['선거구명'], most_similar_texts['시도명'])

    if "투표소" in choosed_tool:
        # 실제 API 호출 및 데이터 처리 코드
        final_return += "아직 선거관리위원회가 투표소에 관한 정보를 제공하고 있지 않습니다.\n\n"
    
    if "선거구" in choosed_tool:

        # prompt_for_district = f"{user_entity}에 대해서 시도명이 반드시 들어간 공식 행정명칭으로 완성해줘."

        # response_completion = client.completions.create(
        #     model="gpt-3.5-turbo-instruct",
        #     prompt=prompt_for_district,
        #     max_tokens=30,
        #     temperature=0
        # )
        # response_text = response_completion.choices[0].text.strip().strip('"')
        user_entity_modified = user_entity.replace(",", " ")
        # 사용자 질문에 대한 임베딩을 2D 배열로 변환
        user_question_embedding = client.embeddings.create(input=user_entity_modified, model='text-embedding-3-small').data[0].embedding
        user_question_embedding_2d = np.array(user_question_embedding).reshape(1, -1)

        # DataFrame의 각 문서 임베딩에 대해 유사도 계산
        df_선거구['similarities'] = df_선거구['embeddings'].apply(lambda x: cosine_similarity([x], user_question_embedding_2d)[0][0])
            
        # 유사도가 가장 높은 상위 5개 텍스트 찾기
        top_5_similar_texts = df_선거구.sort_values('similarities', ascending=False).head(5)
        final_return += "선거구와 그것에 속한 행정구역 정보들입니다. 오는 4월 10일 열리는 제22대 국회의원 선거의 선거구는 아직 확정되지 않았습니다.\n\n"
        for index, row in top_5_similar_texts.iterrows():
            final_return += f"선거구: {row['선거구명']}\n행정구역: {row['시도명']} {row['시군구명']} {row['읍면동명']}\n\n"
        
    if "정당정보" in choosed_tool:
        # 실제 DB 조회 및 데이터 처리 코드
        final_return += "아직 선거관리위원회가 선거 참여 정당들에 관한 정보를 제공하고 있지 않습니다.\n\n"
    
    if "선거정보" in choosed_tool:
        # 사용자 질문에 대한 임베딩을 2D 배열로 변환
        user_question_embedding = client.embeddings.create(input=prompt, model='text-embedding-3-small').data[0].embedding
        user_question_embedding_2d = np.array(user_question_embedding).reshape(1, -1)

        # DataFrame의 각 문서 임베딩에 대해 유사도 계산
        df_선거법['similarities'] = df_선거법['embeddings'].apply(lambda x: cosine_similarity([x], user_question_embedding_2d)[0][0])
            
        # 유사도가 가장 높은 상위 5개 텍스트 찾기
        top_5_similar_texts = df_선거법.sort_values('similarities', ascending=False).head(2)
        final_return_선거법 = ""
        for index, row in top_5_similar_texts.iterrows():
            final_return_선거법 += f"{row['text']}\n\n"
        final_return +=  final_return_선거법
    return final_return

st.title("🧐4월 총선을 알려줘!🗳️")

# 세션 상태에 'messages'가 없으면 초기 메시지를 설정합니다.
인사말 = """
안녕하세요, 저는 총선 챗봇 '땡땡이'입니다. 
언론사에서 챗GPT 기술이 어떻게 활용될 수 있을지 SBS 구성원들에게 예시로 보여드리기 위한 목적으로 개발됐습니다.
자, 그럼 총선에 대해 무엇을 알고 싶으신가요? 제가 엉뚱하게 대답했을 땐 재차 질문해주세요!👀
"""

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": 인사말}]
    
# 세션 상태에 저장된 메시지들을 채팅 메시지 형식으로 표시합니다.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자가 채팅 입력을 했는지 확인합니다.
if prompt := st.chat_input():
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 파일 다운로드가 필요한 경우만 다운로드 진행
    if "file_downloaded" not in st.session_state:
        with st.spinner('챗봇이 선거DB를 조회합니다..'):
            download_and_store_df('1KJw12xawMoOd1RWOSo0ib6PkydLdwYar', 'df_선거구')
            download_and_store_df('1Wc_lP14JjbOUxuUgZPwdkXDB79ccVai3', 'df_선거법')
            st.session_state["file_downloaded"] = True  # 파일 다운로드 상태 표시

    with st.spinner('챗봇이 관련 자료를 찾고 있습니다..'):
        # 사용자의 질문을 분류합니다.
        full_question= complete_prompt(prompt)
        response_content = f"원문: {prompt}  완성: {full_question}\n\n"

        user_intent, user_entity, c_total_tokens = classify_intent_and_entities(prompt)
        response_content += f"1차 분류 :: {user_intent}, {user_entity}, {c_total_tokens} 토큰\n\n"

        choosed_tool, d_total_tokens = choose_tool_with_gpt_detailed(full_question, user_intent)
        response_content += f"2차 분류 :: {choosed_tool}, {d_total_tokens} 토큰\n\n"

        result_prompt_command = process_based_on_chosen_tool(choosed_tool, user_entity, full_question)
        response_content += f"{result_prompt_command}\n\n"

        st.write(response_content)

        final_prompt_command=f"{result_prompt_command}\n\n사용자의 질문: {full_question}"

    with st.chat_message("assistant"):
        # 최종 프롬프트 명령을 기반으로 챗 모델을 사용하여 응답 생성
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Answer only with the information mentioned in the prompt. Please summarize all the answers. If you don't have the information you need to answer, reply '관련 정보가 없어서 답변해드릴 수 없습니다.'."},
                {"role": "user", "content": final_prompt_command}
            ],
            temperature=0.0,
            stream=True,
        )
        response = st.write_stream(stream)
    
    # 분류된 카테고리를 메시지 형태로 변환합니다.
    msg = {"role": "assistant", "content": response}
    # 응답 메시지를 세션 상태의 메시지 목록에 추가하고 채팅 메시지로 표시
    st.session_state.messages.append(msg)