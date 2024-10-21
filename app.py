import streamlit as st
from openai import OpenAI
import os
from PyPDF2 import PdfReader
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv

# OpenAI API 키 가져오기
api_key = st.secrets["OPENAI_API_KEY"]

# OpenAI API 클라이언트 초기화
client = OpenAI(api_key=api_key)

def summarize_files(files):
    contents = ""
    for file in files:
        # 파일 유형 확인 및 내용 추출
        if file.name.lower().endswith('.pdf'):
            reader = PdfReader(file)
            for page in reader.pages:
                contents += page.extract_text() + "\n"
        else:
            # 텍스트 파일 처리
            contents += file.read().decode('utf-8') + "\n"

    # 내용이 너무 길면 잘라냅니다 (API 제한을 고려)
    max_content_length = 40000  # 적절한 값으로 조정하세요
    if len(contents) > max_content_length:
        contents = contents[:max_content_length] + "..."

    # GPT API를 사용하여 요약 생성
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "당신은 행정업무를 도와주는 챗봇입니다."},
            {"role": "user", "content": f"다음 입력된 데이터를 바탕으로 데이터시트로 정리해줘:\n\n{contents}"},
            {"role": "user", "content": "사용자 / 사용일시 / 사용 장소 / 집행목적 / 인원 / 금액 / 방법 / 비목"},
            {"role": "user", "content": "사용자는 제목이 시책추진업무추진비 이면 과장이고 기관운영업무추진비 이면 국장이야. 그리고 부서운영추진비 데이터시트를 만들지마. 방법은 카드 나 현금이야."},
            {"role": "user", "content": "사용장소는 상호명만 적어줘. 집행목적도 간단하게 10자 정도로 정리해."},
            {"role": "user", "content": "데이터시트를 markdown 테이블로 만들어줘."},
        ],
    )

    # 응답에서 테이블 추출
    content = response.choices[0].message.content

    # Markdown 테이블을 DataFrame으로 변환
    df = parse_markdown_table(content)

    return df

def parse_markdown_table(content):
    import re
    table_match = re.search(r'\|.*\|\n(\|[-\s]+\|\n)?(?:\|.*\|\n?)+', content, re.MULTILINE)
    if table_match:
        markdown_table = table_match.group(0)
    else:
        raise ValueError("응답에서 테이블을 찾을 수 없습니다.")

    import pandas as pd
    from io import StringIO

    # 구분선 제거
    lines = markdown_table.strip().split('\n')
    lines = [line for line in lines if not set(line.strip()) == {'|', '-', ' '}]

    # 문자열로 결합
    table = '\n'.join(lines)

    # DataFrame으로 읽기
    df = pd.read_csv(StringIO(table), sep='|', engine='python', skipinitialspace=True)

    # 불필요한 열 제거
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # 공백 제거
    df.columns = df.columns.str.strip()
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    return df

# Streamlit 앱
def main():
    st.set_page_config(page_title="업무추진비 공개 자료 작성", layout="wide")
    st.title('업무추진비 공개 자료 작성')
    st.write('업로드한 파일을 기반으로 데이터시트를 생성하고 엑셀 파일로 다운로드할 수 있습니다.')

    files = st.file_uploader('파일을 업로드하세요 (PDF 또는 텍스트 파일, 여러 개 선택 가능)', type=['txt', 'pdf'], accept_multiple_files=True)

    if files:
        if st.button('데이터시트 생성'):
            with st.spinner('처리 중입니다... 잠시만 기다려주세요.'):
                try:
                    df = summarize_files(files)
                    st.success('데이터시트가 성공적으로 생성되었습니다.')
                    st.dataframe(df)

                    # DataFrame을 엑셀로 저장
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False)
                    output.seek(0)

                    # 다운로드 버튼 제공
                    st.download_button(
                        label="📥 엑셀 파일 다운로드",
                        data=output,
                        file_name='데이터시트.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
                except Exception as e:
                    st.error(f'에러가 발생했습니다: {e}')

if __name__ == '__main__':
    main()
