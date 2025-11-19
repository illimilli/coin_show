# bitcoin_web.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from upbit_utils import get_upbit_tickers, upbit_ohlcv
from analysis_core import (
    compute_indicators,
    compute_signals,
    find_support_resistance,
    linear_regression_trend,
    ai_recommend,
    ai_price_targets,
)

# ---------------------------------------------------------
# 🔧 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="업비트 코인 자동 분석 시스템",
    layout="wide"
)

# ---------------------------------------------------------
# 🎨 테마 선택 (다크 / 라이트)
# ---------------------------------------------------------
theme_choice = st.sidebar.radio(
    "테마 선택",
    ["다크 모드", "라이트 모드"],
    index=0
)

def get_theme_colors(choice: str):
    if choice == "라이트 모드":
        return {
            "bg": "#FFFFFF",
            "fg": "#000000",
            "plot_bg": "#FFFFFF",
            "paper_bg": "#FFFFFF",
        }
    else:  # 다크 모드
        return {
            "bg": "#111111",
            "fg": "#FFFFFF",
            "plot_bg": "#111111",
            "paper_bg": "#111111",
        }

colors = get_theme_colors(theme_choice)

# 전역 Matplotlib 스타일
if theme_choice == "다크 모드":
    plt.style.use("dark_background")
else:
    plt.style.use("default")

# ---------------------------------------------------------
# CSS (모바일 + 전체 스타일)
# ---------------------------------------------------------
st.markdown(f"""
<style>
body {{
    background-color: {colors["bg"]} !important;
    color: {colors["fg"]} !important;
}}
@media (max-width: 768px) {{
    .block-container {{
        padding-top: 0.5rem !important;
        padding-left: 0.4rem !important;
        padding-right: 0.4rem !important;
    }}
    h1, h2, h3 {{
        font-size: 1.2rem !important;
        text-align: center;
    }}
}}
h1, h2, h3 {{
    color: {colors["fg"]} !important;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 가격 포맷 함수
# ---------------------------------------------------------
def format_price(value: float) -> str:
    if value >= 100:
        return f"{value:.2f}"
    elif value >= 1:
        return f"{value:.4f}"
    else:
        return f"{value:.8f}"

# ---------------------------------------------------------
# 📊 Plotly 반응형 캔들 차트
# ---------------------------------------------------------
def plot_interactive_chart(df: pd.DataFrame, name: str) -> go.Figure:
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price",
        increasing_line_color="#26A69A",
        decreasing_line_color="#EF5350"
    ))

    # Volume
    vol_color = "rgba(0,150,255,0.4)" if theme_choice == "다크 모드" else "rgba(0,80,200,0.4)"
    fig.add_trace(go.Bar(
        x=df.index,
        y=df["Volume"],
        name="Volume",
        yaxis="y2",
        marker_color=vol_color
    ))

    # Buy / Sell / Pullback
    buys = df[df["buy_signal"]]
    sells = df[df["sell_signal"]]
    pulls = df[df["pullback_signal"]]

    fig.add_trace(go.Scatter(
        x=buys.index,
        y=buys["Close"],
        mode="markers",
        marker=dict(color="lime", symbol="triangle-up", size=12),
        name="Buy"
    ))

    fig.add_trace(go.Scatter(
        x=sells.index,
        y=sells["Close"],
        mode="markers",
        marker=dict(color="red", symbol="triangle-down", size=12),
        name="Sell"
    ))

    fig.add_trace(go.Scatter(
        x=pulls.index,
        y=pulls["Close"],
        mode="markers",
        marker=dict(color="yellow", symbol="star", size=14),
        name="Pullback"
    ))

    # Support / Resistance
    supports, resistances = find_support_resistance(df)
    for t, s in supports:
        fig.add_hline(y=s, line_dash="dot", line_color="green", opacity=0.4)
    for t, r in resistances:
        fig.add_hline(y=r, line_dash="dot", line_color="red", opacity=0.4)

    # Trend Line
    trend, future = linear_regression_trend(df)
    if len(trend) > 0:
        recent_idx = df.index[-len(trend):]
        fig.add_trace(go.Scatter(
            x=recent_idx,
            y=trend,
            mode="lines",
            line=dict(color="yellow", width=2),
            name="Trend"
        ))

    # Layout
    fig.update_layout(
        title=dict(
            text=f"{name} 차트",
            x=0.5,
            xanchor="center",
            font=dict(size=16, color=colors["fg"])
        ),
        xaxis=dict(
            rangeslider=dict(visible=False),
            showgrid=False
        ),
        yaxis=dict(
            title="Price",
            side="right",
            showgrid=False
        ),
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="left",
            position=0.05,
            showgrid=False,
            range=[0, df["Volume"].max() * 4]
        ),
        autosize=True,
        height=600,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor=colors["plot_bg"],
        paper_bgcolor=colors["paper_bg"],
        font=dict(color=colors["fg"]),
        legend=dict(
            orientation="h",
            x=0.5,
            y=1.12,
            xanchor="center",
            font=dict(size=10)
        )
    )

    return fig

# ---------------------------------------------------------
# 🔽 메인 타이틀
# ---------------------------------------------------------
st.title("📈 업비트 코인 자동 분석 시스템")

# 업비트 티커 불러오기
tickers, names = get_upbit_tickers("KRW")

# 모드 선택
mode = st.sidebar.radio(
    "모드를 선택하세요",
    ["단일 코인 분석", "시장 스캐너"]
)

# =====================================================================
# 1️⃣ 단일 코인 분석 모드
# =====================================================================
if mode == "단일 코인 분석":

    st.subheader("🔍 단일 코인 기술적 분석")

    ticker = st.selectbox("코인 선택", tickers, format_func=lambda x: names[x])

    tf = st.selectbox(
        "시간 프레임",
        ["1분봉", "5분봉", "15분봉", "30분봉", "60분봉", "일봉", "주봉"]
    )

    count = st.slider("가져올 캔들 개수", 200, 2000, 500)

    run = st.button("분석 시작")

    if run:
        # 데이터 로드
        df = upbit_ohlcv(ticker, tf, count)
        if df.empty:
            st.error("데이터를 불러오지 못했습니다.")
            st.stop()

        df = compute_indicators(df)
        df = compute_signals(df)

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last

        change_pct = (last["Close"] - prev["Close"]) / prev["Close"] * 100 if len(df) > 1 else 0

        # -------------------------------
        # 📌 요약 정보
        # -------------------------------
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("현재가", format_price(last["Close"]))
            st.metric("등락률(%)", f"{change_pct:.2f}%")
        with col2:
            st.metric("RSI(14)", f"{last['RSI']:.1f}")
            st.metric("CCI(20)", f"{last['CCI']:.1f}")
        with col3:
            st.metric("현재 거래량", f"{last['Volume']:.0f}")
            st.metric("20일 평균 거래량", f"{last['vol_ma20']:.0f}")

        # -------------------------------
        # 📉 메인 캔들 차트
        # -------------------------------
        fig = plot_interactive_chart(df, names[ticker])
        st.plotly_chart(fig, use_container_width=True)

        # =========================================================
        # 📊 하단 기술적 지표 차트 (6개 패널)
        # =========================================================
        st.subheader("📊 기술적 지표 차트")

        fig2, axes = plt.subplots(3, 2, figsize=(15, 10))
        if theme_choice == "다크 모드":
            fig2.patch.set_facecolor("#111111")

        # 1) 가격 & 이동평균
        axes[0, 0].plot(df.index, df["Close"], label="Close", color="white" if theme_choice=="다크 모드" else "black")
        axes[0, 0].plot(df.index, df["ma20"], label="MA20", color="yellow")
        axes[0, 0].plot(df.index, df["ma60"], label="MA60", color="cyan")
        axes[0, 0].set_title("가격 & 이동평균")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.2)

        # 2) RSI
        axes[0, 1].plot(df.index, df["RSI"], color="violet")
        axes[0, 1].axhline(70, color="red", linestyle="--")
        axes[0, 1].axhline(30, color="green", linestyle="--")
        axes[0, 1].set_title("RSI")
        axes[0, 1].grid(alpha=0.2)

        # 3) Bollinger Bands
        axes[1, 0].plot(df.index, df["UpperBB"], color="red", label="Upper")
        axes[1, 0].plot(df.index, df["MA20"], color="yellow", label="MA20")
        axes[1, 0].plot(df.index, df["LowerBB"], color="green", label="Lower")
        axes[1, 0].fill_between(df.index, df["UpperBB"], df["LowerBB"], alpha=0.1)
        axes[1, 0].set_title("Bollinger Bands")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.2)

        # 4) MACD
        axes[1, 1].plot(df.index, df["MACD"], color="cyan", label="MACD")
        axes[1, 1].plot(df.index, df["Signal"], color="magenta", label="Signal")
        axes[1, 1].set_title("MACD")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.2)

        # 5) Stochastic
        axes[2, 0].plot(df.index, df["%K"], color="blue", label="%K")
        axes[2, 0].plot(df.index, df["%D"], color="orange", label="%D")
        axes[2, 0].axhline(80, color="red", linestyle="--")
        axes[2, 0].axhline(20, color="green", linestyle="--")
        axes[2, 0].set_title("Stochastic Slow")
        axes[2, 0].legend()
        axes[2, 0].grid(alpha=0.2)

        # 6) CCI
        axes[2, 1].plot(df.index, df["CCI"], color="brown")
        axes[2, 1].axhline(100, color="red", linestyle="--")
        axes[2, 1].axhline(-100, color="green", linestyle="--")
        axes[2, 1].set_title("CCI (20)")
        axes[2, 1].grid(alpha=0.2)

        plt.tight_layout()
        st.pyplot(fig2)

        # =========================================================
        # 🧠 지표 자동 해석
        # =========================================================
        st.subheader("🧠 기술적 지표 자동 해석")

        rsi = float(last["RSI"])
        macd = float(last["MACD"])
        sig = float(last["Signal"])
        cci = float(last["CCI"])
        k = float(last["%K"])
        d = float(last["%D"])
        price = float(last["Close"])
        upper = float(last["UpperBB"])
        lower = float(last["LowerBB"])

        # RSI 해석
        if rsi > 70:
            st.write("🔴 **RSI 과매수(>70)** → 과열 구간, 단기 조정 가능성 높음")
        elif rsi < 30:
            st.write("🟢 **RSI 과매도(<30)** → 반등 구간 진입 가능성 높음")
        else:
            st.write("⚪ RSI 중립 구간 → 뚜렷한 과열/과매도 아님")

        # MACD 해석
        if macd > sig:
            st.write("🟢 **MACD 골든크로스** → 상승 모멘텀 우위")
        else:
            st.write("🔴 **MACD 데드크로스** → 하락 압력 우위")

        # Bollinger Bands 해석
        if price >= upper:
            st.write("🔴 **가격이 볼밴 상단에 근접/돌파** → 단기 과열, 조정 위험")
        elif price <= lower:
            st.write("🟢 **가격이 볼밴 하단 근처** → 과매도, 기술적 반등 기대")
        else:
            st.write("⚪ 볼린저 밴드 중앙 부근 → 박스권/중립 흐름")

        # Stochastic 해석
        if k < 20 and d < 20:
            st.write("🟢 **Stochastic 과매도 영역(<20)** → 단기 반등 가능성")
        elif k > 80 and d > 80:
            st.write("🔴 **Stochastic 과매수 영역(>80)** → 단기 조정 가능성")
        elif k > d:
            st.write("🟢 %K가 %D를 상향 돌파 → 단기 상승 전환 신호")
        else:
            st.write("🔴 %K가 %D를 하향 돌파 → 단기 약세 전환 신호")

        # CCI 해석
        if cci > 100:
            st.write("🟢 **CCI +100 이상** → 강한 상승 추세 구간")
        elif cci < -100:
            st.write("🔴 **CCI -100 이하** → 강한 하락 추세 구간")
        else:
            st.write("⚪ CCI 중립 → 뚜렷한 추세보단 완만한 흐름")

        # =========================================================
        # 📌 추천 의견 + AI 매매 추천
        # =========================================================
        st.subheader("📌 종합 추천 의견")

        pull = bool(last["pullback_signal"])
        hot = bool(last["hot_signal"])

        if pull:
            st.success("🔥 눌림목 매수 패턴 포착 → 상승 추세 내 조정 구간 매수 기회")
        if hot:
            st.warning("🚀 거래량 급등(급등주) 패턴 포착 → 변동성 매우 큼, 추격매수 주의")
        if not pull and not hot:
            st.info("특별한 눌림목/급등 패턴 없음 → 지표 기반 일반적인 매매 판단 권장")

        verdict, score, reasons = ai_recommend(df)
        buy_price, stop_loss, take_profit = ai_price_targets(df)

        st.subheader(f"🤖 AI 매매 판단: {verdict} (점수: {score})")

        st.write("### 🔍 분석 근거")
        for r in reasons:
            st.write("- " + r)

        st.write("### 🎯 추천 매매 가격")
        c1, c2, c3 = st.columns(3)
        c1.metric("매수 추천가", format_price(buy_price))
        c2.metric("손절가", format_price(stop_loss))
        c3.metric("익절 목표가", format_price(take_profit))


# =====================================================================
# 2️⃣ 시장 스캐너 모드
# =====================================================================
else:
    st.subheader("📡 업비트 전체 코인 스캐너")

    scan_count = st.sidebar.slider("스캔에 사용할 일봉 개수", 60, 400, 120)
    run_scan = st.button("📡 스캔 실행")

    if run_scan:
        rows = []
        progress = st.progress(0.0)
        total = len(tickers)

        for i, t in enumerate(tickers):
            progress.progress((i + 1) / total)

            try:
                df = upbit_ohlcv(t, "일봉", scan_count)
                if df.empty or len(df) < 30:
                    continue

                df = compute_indicators(df)
                df = compute_signals(df)

                last = df.iloc[-1]
                prev = df.iloc[-2]

                change_pct = (last["Close"] - prev["Close"]) / prev["Close"] * 100

                rows.append({
                    "티커": t,
                    "이름": names.get(t, t),
                    "현재가": float(last["Close"]),
                    "등락률(%)": float(change_pct),
                    "RSI": float(last["RSI"]),
                    "거래량": float(last["Volume"]),
                    "급등신호": bool(last["hot_signal"]),
                    "눌림목": bool(last["pullback_signal"]),
                    "매수신호": bool(last["buy_signal"]),
                    "매도신호": bool(last["sell_signal"]),
                })
            except Exception:
                continue

        if not rows:
            st.warning("조건에 맞는 코인이 없습니다.")
        else:
            scan_df = pd.DataFrame(rows)
            scan_df = scan_df.sort_values("거래량", ascending=False).reset_index(drop=True)
            st.dataframe(scan_df, use_container_width=True)
