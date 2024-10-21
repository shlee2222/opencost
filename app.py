import streamlit as st
from openai import OpenAI
import PyPDF2

# OpenAI API 키 가져오기
api_key = st.secrets["OPENAI_API_KEY"]

# OpenAI API 클라이언트 초기화
client = OpenAI(api_key=api_key)
st.title('당직서는 AI 평택이')

# 파일 업로드
uploaded_file = st.file_uploader('문서를 업로드하세요 (txt, pdf)')

if uploaded_file is not None:
    # 파일 확장자 확인
    if uploaded_file.type == 'application/pdf':
        # PDF 내용 추출
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        content = ''
        for page in pdf_reader.pages:
            content += page.extract_text()
    elif uploaded_file.type == 'text/plain':
        # 텍스트 파일 내용 읽기
        content = uploaded_file.read().decode('utf-8')
    else:
        st.error('지원하지 않는 파일 형식입니다.')
        st.stop()

    # 이하 코드는 동일
    question = st.text_input('질문을 입력하세요')

    if st.button('질문하기'):
        with st.spinner('평택시 ai가 답변을 생성하고 있습니다...'):
            prompt = f'다음 문서를 기반으로 질문에 답해주세요:\n\n문서 내용:\n{content}\n\n질문:\n{question}\n\n답변:'

            response = client.chat.completions.create(
                model='gpt-4o',
                messages=[
                    {"role": "system", "content": "당신은 당직상활실 근무자에게 도움을 주는 챗봇입니다. 당직매뉴얼을 기반으로 답변하나, 만약 내용이 없으면 인터넷을 검색해서 알려주세요."},
                    {"role": "user", "content": prompt}
                ],
                
            )

            answer = response.choices[0].message.content
            st.success('답변:')
            st.write(answer)