"""
港股量化交易 — 数据获取入门程序
================================
支持两种数据来源：
  1. yfinance  — 免费历史数据（无需账号，直接用）
  2. 富途 futu-api — 实时行情 + 实盘交易（需要安装 OpenD 客户端）

使用方式：
  python3 hk_quant_starter.py
"""

import sys
import pandas as pd
from datetime import datetime, timedelta

# ── 颜色输出工具 ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"

def ok(msg):   print(f"{GREEN}[OK]{RESET} {msg}")
def warn(msg): print(f"{YELLOW}[提示]{RESET} {msg}")
def err(msg):  print(f"{RED}[错误]{RESET} {msg}")
def info(msg): print(f"{CYAN}[信息]{RESET} {msg}")

# ── 1. 检查依赖库 ─────────────────────────────────────────────────────────────
def check_dependencies():
    print("\n" + "="*50)
    print("  检查依赖库")
    print("="*50)

    libs = {
        "yfinance":    "免费历史数据",
        "pandas":      "数据处理",
        "mplfinance":  "K线图绘制",
        "futu":        "富途实盘接口（可选）",
    }

    missing = []
    for lib, desc in libs.items():
        try:
            __import__(lib)
            ok(f"{lib:15s} — {desc}")
        except ImportError:
            if lib == "futu":
                warn(f"{lib:15s} — {desc}（未安装，仅影响实盘功能）")
            else:
                err(f"{lib:15s} — {desc}（未安装！）")
                missing.append(lib)

    if missing:
        print(f"\n{RED}请先安装缺失的库：{RESET}")
        print(f"  pip3 install {' '.join(missing)}")
        sys.exit(1)

    ok("所有必要依赖已就绪")

# ── 2. yfinance 获取港股历史数据 ──────────────────────────────────────────────
def fetch_hk_data_yfinance(ticker: str = "0700.HK", days: int = 90) -> pd.DataFrame:
    """
    用 yfinance 免费获取港股历史 K 线数据。

    参数：
        ticker : 港股代码，格式为 数字.HK，不足4位前面补0
                 例如：腾讯=0700.HK  阿里=9988.HK  美团=3690.HK
        days   : 获取多少天的历史数据
    """
    import yfinance as yf

    print("\n" + "="*50)
    print(f"  yfinance — 获取 {ticker} 历史数据")
    print("="*50)

    end_date   = datetime.today()
    start_date = end_date - timedelta(days=days)

    info(f"股票代码：{ticker}")
    info(f"时间范围：{start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    info("正在下载数据（需要网络连接）...")

    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

        if df.empty:
            err("未获取到数据，请检查股票代码是否正确")
            return pd.DataFrame()

        # 整理列名
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.index.name = "日期"

        ok(f"成功获取 {len(df)} 条K线数据")
        print(f"\n{CYAN}--- 最新5条数据 ---{RESET}")
        display = df[["Open", "High", "Low", "Close", "Volume"]].tail(5).copy()
        display.columns = ["开盘", "最高", "最低", "收盘", "成交量"]
        display.index = display.index.strftime("%Y-%m-%d")
        print(display.to_string())

        return df

    except Exception as e:
        err(f"数据获取失败：{e}")
        return pd.DataFrame()


# ── 3. 计算基础技术指标 ───────────────────────────────────────────────────────
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算常用技术指标：
      - MA5 / MA20 / MA60  移动平均线
      - RSI(14)            相对强弱指数
      - 涨跌幅             当日收益率
    """
    if df.empty:
        return df

    print("\n" + "="*50)
    print("  计算技术指标")
    print("="*50)

    close = df["Close"]

    # 移动平均线
    df["MA5"]  = close.rolling(5).mean().round(2)
    df["MA20"] = close.rolling(20).mean().round(2)
    df["MA60"] = close.rolling(60).mean().round(2)
    ok("MA5 / MA20 / MA60 计算完成")

    # RSI(14)
    delta     = close.diff()
    gain      = delta.clip(lower=0).rolling(14).mean()
    loss      = (-delta.clip(upper=0)).rolling(14).mean()
    rs        = gain / loss
    df["RSI"] = (100 - 100 / (1 + rs)).round(2)
    ok("RSI(14) 计算完成")

    # 日涨跌幅（%）
    df["涨跌幅"] = (close.pct_change() * 100).round(2)
    ok("日涨跌幅 计算完成")

    # 显示最新指标
    latest = df[["Close","MA5","MA20","RSI","涨跌幅"]].dropna().tail(3)
    latest.columns = ["收盘价", "MA5", "MA20", "RSI", "涨跌幅(%)"]
    latest.index = latest.index.strftime("%Y-%m-%d")
    print(f"\n{CYAN}--- 最新指标 ---{RESET}")
    print(latest.to_string())

    # 简单信号判断
    print(f"\n{CYAN}--- 当前信号 ---{RESET}")
    last = df.dropna().iloc[-1]
    rsi_val = last["RSI"]

    if rsi_val < 30:
        warn(f"RSI = {rsi_val:.1f}  →  超卖区域，可能存在反弹机会")
    elif rsi_val > 70:
        warn(f"RSI = {rsi_val:.1f}  →  超买区域，注意回调风险")
    else:
        info(f"RSI = {rsi_val:.1f}  →  中性区域")

    if last["MA5"] > last["MA20"]:
        ok("MA5 > MA20  →  短期趋势向上（多头排列）")
    else:
        warn("MA5 < MA20  →  短期趋势向下（空头排列）")

    return df


# ── 4. 绘制 K 线图 ────────────────────────────────────────────────────────────
def plot_kline(df: pd.DataFrame, ticker: str = "0700.HK"):
    """绘制带均线的 K 线图，保存为 PNG 文件"""
    if df.empty:
        return

    print("\n" + "="*50)
    print("  绘制 K 线图")
    print("="*50)

    try:
        import mplfinance as mpf

        plot_df = df[["Open","High","Low","Close","Volume"]].dropna().tail(60)
        # mplfinance 需要列名为英文
        plot_df.columns = ["Open","High","Low","Close","Volume"]

        add_plots = []
        for col, color, label in [
            ("MA5",  "#2196F3", "MA5"),
            ("MA20", "#FF9800", "MA20"),
            ("MA60", "#9C27B0", "MA60"),
        ]:
            if col in df.columns:
                series = df[col].dropna().reindex(plot_df.index)
                add_plots.append(mpf.make_addplot(series, color=color, width=1, label=label))

        filename = f"{ticker.replace('.','_')}_kline.png"
        mpf.plot(
            plot_df,
            type="candle",
            style="charles",
            title=f"{ticker} K线图（近60日）",
            ylabel="价格 (HKD)",
            volume=True,
            addplot=add_plots if add_plots else [],
            savefig=filename,
            figsize=(12, 7),
        )
        ok(f"K 线图已保存：{filename}")

    except Exception as e:
        warn(f"K 线图绘制失败（不影响数据功能）：{e}")


# ── 5. 富途 API 连接测试 ──────────────────────────────────────────────────────
def test_futu_connection():
    """
    测试富途 OpenD 连接。
    前提：需要先在电脑上安装并运行 OpenD 客户端
    下载地址：https://www.futunn.com/download/OpenD
    """
    print("\n" + "="*50)
    print("  富途 OpenD 连接测试（实盘接口）")
    print("="*50)

    try:
        import futu as ft

        HOST = "127.0.0.1"   # OpenD 默认本地地址
        PORT = 11111          # OpenD 默认端口

        info(f"正在连接 OpenD：{HOST}:{PORT}")
        info("（请确保已安装并登录 OpenD 客户端）")

        quote_ctx = ft.OpenQuoteContext(host=HOST, port=PORT)
        ret, data = quote_ctx.get_market_state(["HK.00700"])
        quote_ctx.close()

        if ret == ft.RET_OK:
            ok("富途 OpenD 连接成功！")
            print(data)
        else:
            err(f"查询失败：{data}")

    except ImportError:
        warn("futu-api 未安装，跳过连接测试")
        warn("安装方式：pip3 install futu-api")
        warn("OpenD 下载：https://www.futunn.com/download/OpenD")
    except Exception as e:
        warn(f"OpenD 未运行或连接失败：{e}")
        warn("请先下载安装 OpenD 并登录：https://www.futunn.com/download/OpenD")


# ── 主程序 ────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{CYAN}{'='*50}")
    print("  港股量化交易 — 数据获取入门程序")
    print(f"{'='*50}{RESET}")

    # 你想查哪只股票？修改这里即可
    # 常见港股代码：
    #   腾讯  0700.HK    阿里巴巴 9988.HK
    #   美团  3690.HK    小米    1810.HK
    #   恒生指数 ^HSI
    TICKER = "0700.HK"
    DAYS   = 120          # 获取最近多少天的数据

    # 步骤1：检查依赖
    check_dependencies()

    # 步骤2：获取数据
    df = fetch_hk_data_yfinance(TICKER, DAYS)

    # 步骤3：计算指标
    if not df.empty:
        df = calc_indicators(df)

    # 步骤4：画K线图
    if not df.empty:
        plot_kline(df, TICKER)

    # 步骤5：测试富途连接（如果已装 OpenD 可取消注释）
    # test_futu_connection()

    print(f"\n{GREEN}{'='*50}")
    print("  程序运行完毕")
    print(f"{'='*50}{RESET}")
    print("\n下一步：")
    print("  · 修改 TICKER 变量换其他股票")
    print("  · 安装 OpenD 后取消最后一行注释，测试实盘连接")
    print("  · 告诉 Claude：「帮我写均值回归策略」")


if __name__ == "__main__":
    main()
