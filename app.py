# TODO: 발표 장표 만들기
# TODO: README 파일 작성(환경 변수(예: Azure 키), 설치 방법, 실행 가이드 등)
# TODO: 코드 제출 준비(README, 환경변수 예시(.env.sample), 배포 가이드, 사용 데이터 인덱스)

import streamlit as st
import plotly.graph_objs as go
import textwrap
import copy

from collections import defaultdict
from datetime import datetime, timedelta
from rtm_pipeline import process_document, count_recent_blob, get_openai_response, extract_text_from_file, load_latest_rtm_excel, count_blob
from streamlit_plotly_events import plotly_events

# ───────────────────────────────
# 0. 페이지 컬럼 레이아웃
# ───────────────────────────────
st.set_page_config(page_title="RTM Dashboard", layout="wide")
left, center, right = st.columns([1, 2, 1])

# ───────────────────────────────
# 1. 세션 변수 초기화
# ───────────────────────────────
def initialize_session_state():
    if "uploaded_file_ready" not in st.session_state:
        st.session_state.uploaded_file_ready = False
    if "file_uploaded" not in st.session_state:
        st.session_state.file_uploaded = False
    if "last_uploaded_filename" not in st.session_state:
        st.session_state.last_uploaded_filename = None
    if ("rtm_df" not in st.session_state or
        "blob_url" not in st.session_state or
        "uploaded_filename" not in st.session_state or
        "most_relevant_note" not in st.session_state):
        st.session_state.rtm_df, st.session_state.blob_url, st.session_state.uploaded_filename, st.session_state.most_relevant_note = load_latest_rtm_excel()
    if "total_count" not in st.session_state:
        st.session_state.total_count = count_blob(".txt")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{
            "role": "assistant",
            "content": "궁금한 사항을 질문해주세요!"
        }]

with st.spinner("로드 중입니다..."):
    initialize_session_state()

# ───────────────────────────────
# 2. CSS 주입: 레이아웃
# ───────────────────────────────
st.markdown(
    """
    <style>
    .ver{ color:#909090; margin-top: auto; text-align: right; font-size: 0.9rem; }
    .download-button {
        width: 100%;
        text-align: center;
        display: inline-block;
        padding: 0.55em 1.3em;
        border: 2px solid #1f77b4;
        border-radius: 8px;
        font-weight: 600;
        color: #1f77b4;
        background: #fff;
        transition: all 0.3s ease;
    }
    .download-button:hover {
        background-color: #1f77b4;
        color: white;
    }
    .st-emotion-cache-zy6yx3 {
        padding-top: 4rem !important;
        padding-bottom: 0rem !important;
    }
    
    div[data-testid="stHorizontalBlock"] > div.stColumn:nth-child(3) {
        display: flex;
        flex-direction: column;
        height: 88vh;          /* ↔ 필요시 조정 */
        overflow-y: auto;      /* 컬럼 전체 스크롤 막기 */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ───────────────────────────────
# 3. 다운로드 버튼 헬퍼
# ───────────────────────────────
def download_button(label="Download RTM"):
    if st.session_state.blob_url == "":
        st.markdown(
        textwrap.dedent(
            f"""
            <a
               class="download-button"
               style="
                text-decoration: none;
                pointer-events: none;
               ">
              📥 {label}
            </a>
            """
        ),
        unsafe_allow_html=True,
        )
    else:
        st.markdown(
            textwrap.dedent(
                f"""
                <a href="{st.session_state.blob_url}"
                download
                class="download-button"
                style="
                    text-decoration: none;
                ">
                📥 {label}
                </a>
                """
            ),
            unsafe_allow_html=True,
        )

# ───────────────────────────────
# 5‑A. 좌측: 문서 관리
# ───────────────────────────────
with left:
    # 제목 + 버튼 UI
    col1, col2 = st.columns([9, 5])
    with col1:
        st.markdown("### 📁 문서 관리")
    with col2:
        btn_clicked = st.button("RTM 분석 업로드", disabled=not st.session_state.uploaded_file_ready)

    # 파일 업로드 UI
    uploaded_file = st.file_uploader("회의록 / 요구사항 파일 업로드", type=["docx", "txt"], accept_multiple_files=False)

    # 파일이 새로 업로드 되었거나 삭제되었을 때 상태 초기화
    if uploaded_file:
        if uploaded_file.name != st.session_state.get("last_uploaded_filename", None):
            st.session_state.uploaded_file_ready = True
            st.session_state.last_uploaded_filename = uploaded_file.name
            st.session_state.file_uploaded = False   # 업로드 전 상태로 초기화
            st.rerun()
    else:
        if st.session_state.get("uploaded_file_ready", False) or st.session_state.get("file_uploaded", False):
            st.session_state.uploaded_file_ready = False
            st.session_state.file_uploaded = False
            st.session_state.last_uploaded_filename = None
            st.rerun()

    # 버튼 클릭 시 처리
    if btn_clicked:
        with st.spinner("요구사항 분석 중입니다..."):
            bytes_data = extract_text_from_file(uploaded_file)
            
            now_rtm_df, blob_url, total_count, most_relevant_note = process_document(bytes_data, uploaded_file.name)
            st.session_state.rtm_df = now_rtm_df
            st.session_state.blob_url = blob_url
            st.session_state.total_count = total_count
            st.session_state.most_relevant_note = most_relevant_note
            st.session_state.uploaded_filename = uploaded_file.name
            st.session_state.file_uploaded = True
            st.session_state.uploaded_file_ready = False
            st.rerun()

    # 완료 메시지
    if st.session_state.get("file_uploaded", False):
        st.success("성공적으로 분석 완료되었습니다!")

    col1, col2 = st.columns(2)

    CARD_STYLE = """
        background-color: #f0f2f6;
        padding: 16px;
        border-radius: 12px;
        min-height: 120px;
    """

    with col1:
        st.markdown(f"""
            <div style="{CARD_STYLE}">
                <h6 style="margin: 0;">총 문서 수</h6>
                <p style="font-size: 24px; margin: 0;">{st.session_state.total_count}</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div style="{CARD_STYLE}">
                <h6 style="margin: 0;">가장 관련있는 문서</h6>
                <p style="font-size: 16px; margin: 0; word-break: break-word;">
                    {st.session_state.most_relevant_note}
                </p>
            </div>
        """, unsafe_allow_html=True)
    st.divider()

    date_counts = defaultdict(int)
    today = datetime.now().date()
    dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]
    x = dates
    # 더미 데이터
    y = [1, 0, 2, 4, 5, 3, count_recent_blob(".txt")[-1]]
    fig = go.Figure(
        data=[
            go.Bar(
                x=x,
                y=y,
                marker_color="#A1D99B",
                text=y,
                textposition="inside"
            )
        ]
    )

    fig.update_layout(
        title="최근 7일간 파일 업로드 현황",
        xaxis_title="날짜",
        yaxis_title="업로드 수",
        xaxis=dict(
            tickangle=-45,
            tickmode='array',
            tickvals=x,
            ticktext=[d[5:] for d in x],  # 'YYYY-MM-DD' -> 'MM-DD'만 표시
        ),
        height=400,
        margin=dict(t=40, b=40, l=40, r=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# ───────────────────────────────
# 5‑B. 중앙: 차트 + 표
# ───────────────────────────────
with center:
    col1, col2 = st.columns([6, 2])
    with col1:
        st.markdown("### 요구사항 분석 결과")
    with col2:
        nowFile = st.session_state.uploaded_filename
        st.markdown(f'<div class="ver">ver: {nowFile}</div>', unsafe_allow_html=True)

    status_order = ["양호", "중복", "충돌"]

    # 상태별 집계
    pie_df = (
        st.session_state.rtm_df["상태"].str.strip()
              .value_counts()
              .reindex(status_order, fill_value=0)
              .rename_axis("상태")
              .reset_index(name="Count")
    )

    labels = pie_df["상태"].tolist()
    values = pie_df["Count"].tolist()

    # 총 건 수
    total_count = sum(values)

    # 직접 go.Figure 사용해 명시적 정의
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(
                    colors=[
                        "#72CC5D",  # 양호
                        "#FFA300",  # 중복
                        "#D52D00",  # 충돌
                    ]
                ),
                textinfo="percent+label",
                textfont=dict(size=16, color="black", family="Arial"),
                hovertemplate=(
                "상태 : %{label}<br>" +
                "건수 : %{value}건<br>" +
                "비율 : %{percent}<extra></extra>"
                )
            )
        ]
    )
    fig.update_layout(
        height=450,
        margin=dict(t=30, b=30, l=0, r=0),
        annotations=[
        dict(
            text=f"{total_count}건",
            x=0.5,
            y=0.5,
            font=dict(size=22, color="black"),
            showarrow=False
        )
    ]
    )

    # 클릭 이벤트
    click = plotly_events(
        copy.deepcopy(fig),
        click_event=True,
        select_event=False,
        override_height=450,
        key="pie",
    )

    # 클릭 결과
    sel_label = labels[click[0]["pointNumber"]] if click else "양호"

    st.caption(f"**{sel_label}** 상태의 요구사항 목록")

    filtered_df = st.session_state.rtm_df[st.session_state.rtm_df["상태"].str.strip() == sel_label].reset_index(drop=True)
    # 'Status' 컬럼 제외
    filtered_df = filtered_df.drop(columns=["상태"])
    filtered_df.index = filtered_df.index + 1  # 인덱스를 1부터 시작
    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=240,
    )

# ───────────────────────────────
# 5‑C. 우측: AI Chat
# ───────────────────────────────
with right:
    download_button()
    st.subheader("💬 AI Chat")
    user_input = st.chat_input("Enter your question: ")

    for message in st.session_state.chat_history:
        st.chat_message(message["role"]).write(message["content"])

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        with st.spinner("응답을 기다리는 중입니다..."):
            response = get_openai_response(st.session_state.chat_history)
            st.chat_message("assistant").write(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})