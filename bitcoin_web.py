# bitcoin_web.py
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import mplfinance as mpf
import numpy as np

from upbit_utils import get_upbit_tickers, upbit_ohlcv
from analysis_core import (
    compute_indicators,
    compute_signals,
    find_support_resistance,
    linear_regression_trend,
    ai_recommend,
    ai_price_targets
)

# ======================================================
# 📌 Plotly Interactive Chart Function (HTS 스타일)
# ======================================================
import plotly.graph_objects as go

def plot_interactive_chart(df, ticker_name):
    fig = go.Figure()

    # ---------------------------
    # 📌 캔들 차트
    # ---------------------------
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Candles"
    ))

    # ---------------------------
    # 📌 거래량
    # ---------------------------
    fig.add_trace(go.Bar(
        x=df.index,
        y=df["Volume"],
        name="Volume",
        marker_color="rgba(0,150,255,0.4)",
        yaxis="y2"
    ))

    # ---------------------------
    # 📌 BUY SIGNAL
    # ---------------------------
    buys = df.index[df["buy_signal"]]
    fig.add_trace(go.Scatter(
        x=buys,
        y=df["Close"][df["buy_signal"]],
        mode="markers",
        marker=dict(color="lime", size=12, symbol="triangle-up"),
        name="Buy Signal"
    ))

    # ---------------------------
    # 📌 SELL SIGNAL
    # ---------------------------
    sells = df.index[df["sell_signal"]]
    fig.add_trace(go.Scatter(
        x=sells,
        y=df["Close"][df["sell_signal"]],
        mode="markers",
        marker=dict(color="red", size=12, symbol="triangle-down"),
        name="Sell Signal"
    ))

    # ---------------------------
    # 📌 눌림목 신호 표시
    # ---------------------------
    pulls = df.index[df["pullback_signal"]]
    fig.add_trace(go.Scatter(
        x=pulls,
        y=df["Close"][df["pullback_signal"]],
        mode="markers",
        marker=dict(color="yellow", size=14, symbol="star"),
        name="Pullback Buy"
    ))

    # ---------------------------
    # 📌 지지/저항선
    # ---------------------------
    from analysis_core import find_support_resistance, linear_regression_trend

    supports, resistances = find_support_resistance(df)

    for t, s in supports:
        fig.add_hline(y=s, line_dash="dot", line_color="green", opacity=0.4)

    for t, r in resistances:
        fig.add_hline(y=r, line_dash="dot", line_color="red", opacity=0.4)

    # ---------------------------
    # 📌 추세선 + 미래 예측
    # ---------------------------
    trend, future = linear_regression_trend(df)

    if len(trend) > 0:
        recent_idx = df.index[-len(trend):]
        fig.add_trace(go.Scatter(
            x=recent_idx,
            y=trend,
            mode="lines",
            line=dict(color="yellow", width=2),
            name="Trend Line"
        ))

    # Plotly Layout
    fig.update_layout(
        title=f"{ticker_name} - Interactive Chart",
        xaxis=dict(
            rangeslider=dict(visible=False),
            showgrid=False
        ),
        yaxis=dict(
            title="Price",
            side="right"
        ),
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="left",
            position=0.05,
            showgrid=False
        ),
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="white"),
        height=750,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


# -------------------------
# 가격 형식 통일 함수
# -------------------------
def format_price(value):
    if value >= 100:
        return f"{value:.2f}"
    elif value >= 1:
        return f"{value:.4f}"
    else:
        return f"{value:.8f}"

# Streamlit 설정
plt.style.use("dark_background")
st.set_page_config(layout="wide")
st.title("📈 업비트 코인 자동 분석 시스템 (단일 분석 + 시장 스캐너)")

# ------------------------------
# 업비트 티커 로드
# ------------------------------
tickers, names = get_upbit_tickers("KRW")

# ------------------------------
# 모드 선택
# ------------------------------
mode = st.sidebar.radio(
    "모드를 선택하세요",
    ["단일 코인 분석", "시장 스캐너"]
)

# =====================================================================
# 1) 단일 코인 분석
# =====================================================================
if mode == "단일 코인 분석":

    st.subheader("🔍 단일 코인 기술적 분석")

    ticker = st.selectbox("코인을 선택하세요", tickers, format_func=lambda x: names[x])

    tf = st.selectbox(
        "📌 시간 프레임 선택",
        ["1분봉", "3분봉", "5분봉", "10분봉", "15분봉", "30분봉", "60분봉", "일봉", "주봉"]
    )

    count = st.slider("가져올 캔들 개수", 200, 2000, 500)

    run = st.button("분석 시작", key="analysis_button")

    if run:

        # ------------------------
        # 데이터 불러오기
        # ------------------------
        df = upbit_ohlcv(ticker, tf, count)

        if df.empty:
            st.error("데이터 로드 실패")
            st.stop()

        df = compute_indicators(df)
        df = compute_signals(df)

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last

        change_pct = (last["Close"] - prev["Close"]) / prev["Close"] * 100 if len(df) > 1 else 0

        # ------------------------
        # 가격/RSI/거래량 요약
        # ------------------------
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("현재가", format_price(last["Close"]))
            st.metric("등락률(%)", f"{change_pct:.2f}%")
        with col2:
            st.metric("RSI(14)", f"{last['RSI']:.1f}")
            st.metric("CCI(20)", f"{last['CCI']:.1f}")
        with col3:
            st.metric("거래량", f"{last['Volume']:.0f}")
            st.metric("20일 평균 거래량", f"{last['vol_ma20']:.0f}")

        # ---------------------------------------------------------
        # 📌 트레이딩뷰 스타일 캔들 차트
        # ---------------------------------------------------------
        st.subheader("📉 트레이딩뷰 스타일 캔들 차트")

        modern_style = mpf.make_mpf_style(
            base_mpf_style='nightclouds',
            marketcolors=mpf.make_marketcolors(
                up="#26A69A",
                down="#EF5350",
                wick="white",
                edge="inherit",
                volume={"up": "#26A69A", "down": "#EF5350"}
            ),
            gridcolor="#444444",
            gridstyle="-",
            facecolor="#0D0D0D",
            figcolor="#0D0D0D"
        )

        candle_data = df[["Open", "High", "Low", "Close", "Volume"]].copy()

        fig = plot_interactive_chart(df, names[ticker])
        st.plotly_chart(fig, use_container_width=True)


        # ---------------------------------------------------------
        # 📊 지표 대시보드
        # ---------------------------------------------------------
        st.markdown("### 📊 기술적 지표 대시보드")

        fig2, axes = plt.subplots(3, 2, figsize=(15, 10))

        axes[0, 0].plot(df.index, df["Close"], label="Close")
        axes[0, 0].plot(df.index, df["ma20"], label="MA20")
        axes[0, 0].plot(df.index, df["ma60"], label="MA60")
        axes[0, 0].set_title("가격 & 이동평균")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.2)

        axes[0, 1].plot(df.index, df["RSI"], color="purple")
        axes[0, 1].axhline(70, color="red", linestyle="--")
        axes[0, 1].axhline(30, color="green", linestyle="--")
        axes[0, 1].set_title("RSI (14)")
        axes[0, 1].grid(alpha=0.2)

        axes[1, 0].plot(df.index, df["UpperBB"], color="red")
        axes[1, 0].plot(df.index, df["MA20"], color="orange")
        axes[1, 0].plot(df.index, df["LowerBB"], color="green")
        axes[1, 0].fill_between(df.index, df["UpperBB"], df["LowerBB"], alpha=0.1)
        axes[1, 0].set_title("Bollinger Bands")
        axes[1, 0].grid(alpha=0.2)

        axes[1, 1].plot(df.index, df["MACD"], color="blue")
        axes[1, 1].plot(df.index, df["Signal"], color="red")
        axes[1, 1].set_title("MACD")
        axes[1, 1].grid(alpha=0.2)

        axes[2, 0].plot(df.index, df["%K"], color="blue")
        axes[2, 0].plot(df.index, df["%D"], color="orange")
        axes[2, 0].set_title("Stochastic")
        axes[2, 0].grid(alpha=0.2)

        axes[2, 1].plot(df.index, df["CCI"], color="brown")
        axes[2, 1].set_title("CCI")
        axes[2, 1].grid(alpha=0.2)

        st.pyplot(fig2)

        # ---------------------------------------------------------
        # 📌 추천의견
        # ---------------------------------------------------------
        pull = bool(df["pullback_signal"].iloc[-1])
        hot = bool(df["hot_signal"].iloc[-1])

        st.subheader("📌 추천 의견")

        if pull:
            st.success("🔥 눌림목 매수 신호 감지")
        if hot:
            st.warning("🚀 거래량 급등(급등주) 신호 발생")
        if not pull and not hot:
            st.info("특별한 신호 없음")

        # ---------------------------------------------------------
        # 🤖 AI 매매 판단
        # ---------------------------------------------------------
        verdict, score, reasons = ai_recommend(df)
        buy_price, stop_loss, take_profit = ai_price_targets(df)

        st.subheader(f"🤖 AI 판단: {verdict} (점수: {score})")

        st.write("### 📌 분석 근거")
        for r in reasons:
            st.write("- " + r)

        st.write("### 🎯 추천 매매 가격")
        st.metric("매수 추천가", format_price(buy_price))
        st.metric("손절가", format_price(stop_loss))
        st.metric("익절 목표가", format_price(take_profit))



# =====================================================================
# 2) 시장 스캐너
# =====================================================================
else:
    st.subheader("📡 업비트 전체 코인 스캐너")

    scan_count = st.sidebar.slider("스캔 시 사용할 일봉 개수", 60, 400, 120)
    min_vol = st.sidebar.number_input("최소 거래량 필터", value=0.0)
    only_hot = st.sidebar.checkbox("🚀 급등 신호만 표시")
    only_pullback = st.sidebar.checkbox("🔥 눌림목 신호만 표시")

    run_scan = st.button("📡 시장 스캔 실행", key="scan_run")

    if run_scan:

        rows = []
        progress = st.progress(0.0)

        total = len(tickers)

        for i, t in enumerate(tickers):
            progress.progress((i + 1) / total)

            try:
                df = upbit_ohlcv(t, "일봉", scan_count)

                if len(df) < 40:
                    continue

                df = compute_indicators(df)
                df = compute_signals(df)

                last = df.iloc[-1]
                prev = df.iloc[-2]

                row = {
                    "티커": t,
                    "이름": names.get(t, t),
                    "현재가": float(last["Close"]),
                    "등락률(%)": float((last["Close"] - prev["Close"]) / prev["Close"] * 100),
                    "RSI": float(last["RSI"]),
                    "거래량": float(last["Volume"]),
                    "평균 거래량(20)": float(last["vol_ma20"]),
                    "급등신호": bool(last["hot_signal"]),
                    "눌림목": bool(last["pullback_signal"]),
                    "매수신호": bool(last["buy_signal"]),
                    "매도신호": bool(last["sell_signal"]),
                }

                if min_vol > 0 and row["거래량"] < min_vol:
                    continue

                rows.append(row)

            except:
                continue

        if not rows:
            st.warning("조건에 맞는 코인이 없습니다.")
        else:
            df_scan = pd.DataFrame(rows)

            if only_hot:
                df_scan = df_scan[df_scan["급등신호"]]
            if only_pullback:
                df_scan = df_scan[df_scan["눌림목"]]

            df_scan = df_scan.sort_values("거래량", ascending=False).reset_index(drop=True)

            st.dataframe(df_scan, use_container_width=True)
