# analysis_core.py
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


# -----------------------------
# 기술적 지표 계산
# -----------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]

    # 이동평균
    df["ma20"] = close.rolling(20).mean()
    df["ma60"] = close.rolling(60).mean()

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger
    df["MA20"] = close.rolling(20).mean()
    df["STD20"] = close.rolling(20).std()
    df["UpperBB"] = df["MA20"] + 2 * df["STD20"]
    df["LowerBB"] = df["MA20"] - 2 * df["STD20"]

    # MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Stochastic
    low14 = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["%K"] = (close - low14) / (high14 - low14) * 100
    df["%D"] = df["%K"].rolling(3).mean()

    # CCI
    tp = (df["High"] + df["Low"] + close) / 3
    sma = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    df["CCI"] = (tp - sma) / (0.015 * mad)

        # -----------------------------
    # 회귀선 기울기 계산
    # -----------------------------
    trend, future = linear_regression_trend(df)
    if len(trend) > 1:
        slope = trend[-1] - trend[0]
        df["reg_slope"] = 0
        df.loc[df.index[-1], "reg_slope"] = slope
    else:
        df["reg_slope"] = 0


    # 거래량 평균
    df["vol_ma20"] = df["Volume"].rolling(20).mean()

    return df


# -----------------------------
# 매수/매도 & 눌림목 & 급등신호
# -----------------------------
def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]

    # 볼밴 + RSI 기반
    df["buy_signal"] = (close < df["LowerBB"]) & (df["RSI"] < 30)
    df["sell_signal"] = (close > df["UpperBB"]) & (df["RSI"] > 70)

    # 눌림목 (상승추세 + ma20 아래로 눌림 + RSI 30~55 + MACD 시그널 상향 돌파)
    trend_up = df["ma20"] > df["ma60"]
    pullback = close < df["ma20"]
    rsi_ok = (df["RSI"] > 30) & (df["RSI"] < 55)

    macd_prev = (df["MACD"] - df["Signal"]).shift(1)
    macd_now = (df["MACD"] - df["Signal"])
    macd_cross_up = (macd_prev < 0) & (macd_now >= 0)

    df["pullback_signal"] = trend_up & pullback & rsi_ok & macd_cross_up

    # 급등: 거래량 3배 이상 + 직전 종가 대비 5% 이상 상승
    df["hot_signal"] = False
    if len(df) > 21:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        cond_vol = last["Volume"] > last["vol_ma20"] * 3
        cond_price = (last["Close"] - prev["Close"]) / prev["Close"] * 100 >= 5
        df.loc[df.index[-1], "hot_signal"] = bool(cond_vol and cond_price)

    return df


# -----------------------------
# 지지/저항 자동 탐지
# -----------------------------
def find_support_resistance(df: pd.DataFrame, window: int = 10):
    supports = []
    resistances = []
    if len(df) < 2 * window + 1:
        return supports, resistances

    closes = df["Close"].values
    idx = df.index

    for i in range(window, len(df) - window):
        segment = closes[i - window: i + window + 1]
        if closes[i] == segment.min():
            supports.append((idx[i], closes[i]))
        if closes[i] == segment.max():
            resistances.append((idx[i], closes[i]))

    return supports, resistances


# -----------------------------
# 단순 회귀 추세선 + 미래 예측
# -----------------------------
def linear_regression_trend(df: pd.DataFrame, days: int = 60, future_days: int = 14):
    if len(df) < 10:
        return np.array([]), np.array([])

    recent = df.tail(days)
    y = recent["Close"].values
    X = np.arange(len(y)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    trend = model.predict(X)

    X_future = np.arange(len(y), len(y) + future_days).reshape(-1, 1)
    future = model.predict(X_future)

    return trend, future

def ai_recommend(df):

    last = df.iloc[-1]

    score = 0
    reasons = []

    # ================================
    # 1) RSI 신호
    # ================================
    if last["RSI"] < 30:
        score += 15
        reasons.append("RSI 과매도 → 매수 우세")
    elif last["RSI"] > 70:
        score -= 15
        reasons.append("RSI 과매수 → 매도 우세")

    # ================================
    # 2) MACD 신호
    # ================================
    if last["MACD"] > last["Signal"]:
        score += 10
        reasons.append("MACD 골든크로스 → 상승 흐름")
    else:
        score -= 10
        reasons.append("MACD 데드크로스 → 하락 흐름")

    # ================================
    # 3) 볼린저밴드 위치
    # ================================
    if last["Close"] < last["LowerBB"]:
        score += 10
        reasons.append("볼밴 하단 이탈 → 반등 구간")
    elif last["Close"] > last["UpperBB"]:
        score -= 10
        reasons.append("볼밴 상단 돌파 → 과열 구간")

    # ================================
    # 4) 눌림목 신호
    # ================================
    if last.get("pullback_signal", False):
        score += 20
        reasons.append("눌림목 매수 패턴 → 강한 매수")

    # ================================
    # 5) 급등/수급 신호
    # ================================
    if last.get("hot_signal", False):
        score -= 10
        reasons.append("급등 후 과열 가능성 있음")

    # ================================
    # 6) 추세 점수 (회귀선 기울기)
    # ================================
    if "reg_slope" in df.columns:
        slope = df["reg_slope"].iloc[-1]
        if slope > 0:
            score += 10
            reasons.append("상승 추세 유지")
        else:
            score -= 10
            reasons.append("하락 추세 유지")

    # FINAL SCORE
    if score >= 40:
        verdict = "🔥 강한 매수 (Strong Buy)"
    elif score >= 20:
        verdict = "📈 매수 (Buy)"
    elif score >= -10:
        verdict = "➖ 중립 (Neutral)"
    elif score >= -30:
        verdict = "📉 매도 (Sell)"
    else:
        verdict = "💀 강한 매도 (Strong Sell)"

    return verdict, score, reasons

def ai_price_targets(df):

    last = df.iloc[-1]
    close = last["Close"]

    atr = (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]

    buy_price = close - atr * 0.5
    stop_loss = close - atr * 1.5
    take_profit = close + atr * 2.0

    return buy_price, stop_loss, take_profit
