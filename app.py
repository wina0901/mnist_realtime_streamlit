from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from mnist_onnx import load_session, predict_digit, preprocess_canvas_image


st.set_page_config(
    page_title="실시간 손글씨 미션",
    page_icon="",
    layout="wide",
)

st.title("MNIST ONNX 손글씨")
st.caption("Streamlit + drawable canvas + ONNX Runtime")
st.divider()

try:
    load_session()
    st.sidebar.success("모델 로딩 완료")
except Exception as exc:  # noqa: BLE001
    st.sidebar.error("모델 로딩 실패")
    st.sidebar.exception(exc)

with st.sidebar:
    st.header("입력 설정")
    stroke_width = st.slider("펜 두께", min_value=5, max_value=35, value=18)
    realtime_update = st.toggle("그리는 중 실시간 반영", value=True)
    st.info("검은 배경 위에 흰색으로 숫자 0~9 중 하나를 크게 그려주세요.")

canvas_col, preview_col, result_col = st.columns([1.25, 0.85, 1.25])

with canvas_col:
    st.subheader("1. 입력 캔버스")
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        update_streamlit=realtime_update,
        key="mnist_canvas",
    )
    run_button = st.button("🔍 숫자 예측하기", type="primary", use_container_width=True)

prediction = None
preview_image = None

if canvas_result.image_data is not None:
    try:
        input_tensor, preview_image = preprocess_canvas_image(canvas_result.image_data)
        if run_button or realtime_update:
            prediction = predict_digit(input_tensor)
    except Exception as exc:  # noqa: BLE001
        st.error(f"전처리 또는 추론 중 오류가 발생했습니다: {exc}")

with preview_col:
    st.subheader("2. 전처리 이미지")
    if preview_image is not None:
        st.image(
            preview_image.resize((168, 168), Image.Resampling.NEAREST),
            caption="28×28 모델 입력 이미지",
        )
    else:
        st.info("캔버스에 숫자를 입력하면 전처리 결과가 표시됩니다.")

with result_col:
    st.subheader("3. 모델 추론 결과")
    if prediction is not None:
        label = prediction["label"]
        confidence = prediction["confidence"]
        probabilities = prediction["probabilities"]

        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("예측 숫자", label)
        metric_col2.metric("신뢰도", f"{confidence:.2%}")
        st.progress(confidence)

        df = pd.DataFrame(
            {
                "digit": [str(i) for i in range(10)],
                "probability": probabilities,
            }
        )
        chart = (
            alt.Chart(df)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                x=alt.X("probability:Q", title="예측 확률", axis=alt.Axis(format="%")),
                y=alt.Y("digit:N", sort="-x", title="숫자"),
                tooltip=["digit", alt.Tooltip("probability:Q", format=".2%")],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)
        st.success(f"이 손글씨는 {confidence:.2%}의 확률로 숫자 [{label}]로 예측되었습니다.")
    else:
        st.info("숫자를 그린 뒤 예측 버튼을 누르면 결과가 표시됩니다.")
