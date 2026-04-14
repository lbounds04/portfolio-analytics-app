import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
from scipy.stats import norm, probplot
from scipy.optimize import minimize

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Interactive Portfolio Analytics App",
    layout="wide"
)

# -----------------------------
# CONSTANTS
# -----------------------------
TRADING_DAYS = 252
BENCHMARK_TICKER = "^GSPC"

# -----------------------------
# SESSION STATE
# -----------------------------
if "analysis_ready" not in st.session_state:
    st.session_state.analysis_ready = False

if "ticker_text" not in st.session_state:
    st.session_state.ticker_text = "AAPL, MSFT, NVDA"

if "start_date" not in st.session_state:
    st.session_state.start_date = date.today() - timedelta(days=365 * 5)

if "end_date" not in st.session_state:
    st.session_state.end_date = date.today()

if "risk_free_rate" not in st.session_state:
    st.session_state.risk_free_rate = 2.0

# -----------------------------
# PAGE HEADER
# -----------------------------
st.title("Interactive Portfolio Analytics Application")
st.markdown(
    """
    Build and analyze stock portfolios in real time using **Python, Streamlit, and financial analytics**.

    This app lets you:
    - download and clean market data
    - study return distributions and risk
    - compare correlation and covariance relationships
    - evaluate equal-weight and optimized portfolios
    - test how optimization changes across different estimation windows
    """
)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def parse_tickers(ticker_text):
    tickers = [ticker.strip().upper() for ticker in ticker_text.split(",")]
    tickers = [ticker for ticker in tickers if ticker != ""]
    return list(dict.fromkeys(tickers))


def minimum_two_years(start_date, end_date):
    return (end_date - start_date).days >= 730


@st.cache_data(ttl=3600)
def download_price_data(tickers, start_date, end_date):
    all_tickers = tickers + [BENCHMARK_TICKER]
    good_data = {}
    bad_tickers = []

    for ticker in all_tickers:
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False
            )

            if df.empty:
                bad_tickers.append(ticker)
                continue

            if isinstance(df.columns, pd.MultiIndex):
                if ("Adj Close", ticker) in df.columns:
                    series = df[("Adj Close", ticker)].dropna()
                elif ("Close", ticker) in df.columns:
                    series = df[("Close", ticker)].dropna()
                else:
                    bad_tickers.append(ticker)
                    continue
            else:
                if "Adj Close" in df.columns:
                    series = df["Adj Close"].dropna()
                elif "Close" in df.columns:
                    series = df["Close"].dropna()
                else:
                    bad_tickers.append(ticker)
                    continue

            if not isinstance(series, pd.Series):
                series = pd.Series(series.squeeze(), index=df.index).dropna()

            if len(series) < 100:
                bad_tickers.append(ticker)
                continue

            series.name = ticker
            good_data[ticker] = series

        except Exception:
            bad_tickers.append(ticker)

    if not good_data:
        return pd.DataFrame(), bad_tickers

    prices = pd.concat(good_data.values(), axis=1)
    return prices, bad_tickers


def align_and_clean_prices(prices):
    if prices.empty:
        return prices, []

    dropped = []

    for col in prices.columns:
        missing_pct = prices[col].isna().mean()
        if missing_pct > 0.05 and col != BENCHMARK_TICKER:
            dropped.append(col)

    cleaned = prices.drop(columns=dropped, errors="ignore")
    cleaned = cleaned.dropna()

    return cleaned, dropped


@st.cache_data
def compute_returns(prices):
    return prices.pct_change().dropna()


@st.cache_data
def compute_summary_stats(returns):
    stats_df = pd.DataFrame(index=returns.columns)
    stats_df["Annualized Mean Return"] = returns.mean() * TRADING_DAYS
    stats_df["Annualized Volatility"] = returns.std() * np.sqrt(TRADING_DAYS)
    stats_df["Skewness"] = returns.skew()
    stats_df["Kurtosis"] = returns.kurtosis()
    stats_df["Min Daily Return"] = returns.min()
    stats_df["Max Daily Return"] = returns.max()
    return stats_df


@st.cache_data
def compute_risk_metrics(returns, annual_rf_rate):
    daily_rf = annual_rf_rate / 100 / TRADING_DAYS
    annual_rf_decimal = annual_rf_rate / 100

    metrics = {}

    for col in returns.columns:
        asset_returns = returns[col].dropna()

        mean_return_annual = asset_returns.mean() * TRADING_DAYS
        asset_std = asset_returns.std()

        if asset_std == 0 or pd.isna(asset_std):
            sharpe = np.nan
        else:
            sharpe = ((asset_returns.mean() - daily_rf) / asset_std) * np.sqrt(TRADING_DAYS)

        downside_returns = asset_returns[asset_returns < daily_rf]
        downside_dev_annual = downside_returns.std() * np.sqrt(TRADING_DAYS)

        if pd.isna(downside_dev_annual) or downside_dev_annual == 0:
            sortino = np.nan
        else:
            sortino = (mean_return_annual - annual_rf_decimal) / downside_dev_annual

        metrics[col] = {
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino
        }

    return pd.DataFrame(metrics).T


def compute_drawdown(return_series):
    wealth_index = (1 + return_series).cumprod()
    running_peak = wealth_index.cummax()
    drawdown = (wealth_index - running_peak) / running_peak
    return drawdown


def compute_rolling_volatility(returns, window):
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def compute_rolling_correlation(returns, stock1, stock2, window):
    return returns[stock1].rolling(window).corr(returns[stock2])


# -----------------------------
# PORTFOLIO OPTIMIZATION HELPERS
# -----------------------------
def portfolio_return(weights, mean_returns):
    return np.sum(weights * mean_returns) * TRADING_DAYS


def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(TRADING_DAYS)


def portfolio_variance(weights, cov_matrix):
    return float(weights.T @ cov_matrix @ weights)


def negative_sharpe_ratio(weights, mean_returns, cov_matrix, annual_rf_rate):
    port_return = portfolio_return(weights, mean_returns)
    port_vol = portfolio_volatility(weights, cov_matrix)
    rf_decimal = annual_rf_rate / 100

    if port_vol == 0:
        return 1e6

    sharpe = (port_return - rf_decimal) / port_vol
    return -sharpe


def optimize_gmv(mean_returns, cov_matrix):
    """
    Find the Global Minimum Variance portfolio with no short selling.
    Uses multiple starting guesses to avoid getting stuck at equal weights.
    """
    n = len(mean_returns)
    bounds = tuple((0, 1) for _ in range(n))
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    initial_guesses = [np.repeat(1 / n, n)]

    for i in range(n):
        guess = np.zeros(n)
        guess[i] = 1.0
        initial_guesses.append(guess)

    best_result = None
    best_value = np.inf

    for guess in initial_guesses:
        result = minimize(
            fun=portfolio_variance,
            x0=guess,
            args=(cov_matrix,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-12, "disp": False}
        )

        if result.success:
            value = portfolio_variance(result.x, cov_matrix)
            if value < best_value:
                best_value = value
                best_result = result

    if best_result is not None:
        return best_result.x

    return None


def optimize_tangency(mean_returns, cov_matrix, annual_rf_rate):
    n = len(mean_returns)
    initial_weights = np.repeat(1 / n, n)
    bounds = tuple((0, 1) for _ in range(n))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(mean_returns, cov_matrix, annual_rf_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12}
    )

    if result.success:
        return result.x
    return None


def compute_portfolio_metrics(weights, returns_df, annual_rf_rate):
    portfolio_daily_returns = returns_df.dot(weights)

    ann_return = portfolio_daily_returns.mean() * TRADING_DAYS
    ann_vol = portfolio_daily_returns.std() * np.sqrt(TRADING_DAYS)

    daily_rf = annual_rf_rate / 100 / TRADING_DAYS
    annual_rf_decimal = annual_rf_rate / 100

    if portfolio_daily_returns.std() == 0:
        sharpe = np.nan
    else:
        sharpe = ((portfolio_daily_returns.mean() - daily_rf) / portfolio_daily_returns.std()) * np.sqrt(TRADING_DAYS)

    downside_returns = portfolio_daily_returns[portfolio_daily_returns < daily_rf]
    downside_dev_annual = downside_returns.std() * np.sqrt(TRADING_DAYS)

    if pd.isna(downside_dev_annual) or downside_dev_annual == 0:
        sortino = np.nan
    else:
        sortino = (ann_return - annual_rf_decimal) / downside_dev_annual

    drawdown = compute_drawdown(portfolio_daily_returns)
    max_drawdown = drawdown.min()

    return {
        "Annualized Return": ann_return,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Maximum Drawdown": max_drawdown
    }


def build_weights_table(weights, tickers, name):
    return pd.DataFrame({
        "Ticker": tickers,
        f"{name} Weight": weights
    })


def get_available_lookback_options(returns_df):
    total_years = len(returns_df) / TRADING_DAYS
    options = []

    if total_years >= 1:
        options.append(1)
    if total_years >= 3:
        options.append(3)
    if total_years >= 5:
        options.append(5)

    options.append("Full Sample")
    return options


def get_lookback_subset(stock_returns, window):
    if window == "Full Sample":
        return stock_returns.copy()

    num_days = int(window * TRADING_DAYS)
    return stock_returns.tail(num_days).copy()


# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
with st.sidebar.form("input_form"):
    st.header("Portfolio Inputs")

    ticker_text = st.text_input(
        "Enter 3 to 10 stock tickers (comma-separated)",
        value=st.session_state.ticker_text
    )

    start_date = st.date_input(
        "Start date",
        value=st.session_state.start_date
    )

    end_date = st.date_input(
        "End date",
        value=st.session_state.end_date
    )

    risk_free_rate = st.number_input(
        "Annual risk-free rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=float(st.session_state.risk_free_rate),
        step=0.1
    )

    run_button = st.form_submit_button("Run Analysis")

if run_button:
    st.session_state.analysis_ready = True
    st.session_state.ticker_text = ticker_text
    st.session_state.start_date = start_date
    st.session_state.end_date = end_date
    st.session_state.risk_free_rate = risk_free_rate

with st.sidebar.expander("About / Methodology"):
    st.markdown(
        """
        **Data source:** yfinance adjusted close prices  
        **Benchmark:** S&P 500 (`^GSPC`)  
        **Returns used:** simple daily returns  
        **Annualized return:** mean daily return × 252  
        **Annualized volatility:** daily standard deviation × √252  
        **Risk-free rate:** annualized user input, converted to daily where needed  
        **Optimization constraints:** no short selling, weights between 0 and 1, weights sum to 1  
        **Missing data rule:** stocks with more than 5% missing values are dropped  
        """
    )

# -----------------------------
# MAIN APP LOGIC
# -----------------------------
if st.session_state.analysis_ready:
    ticker_text = st.session_state.ticker_text
    start_date = st.session_state.start_date
    end_date = st.session_state.end_date
    risk_free_rate = st.session_state.risk_free_rate

    tickers = parse_tickers(ticker_text)

    if len(tickers) < 3:
        st.error("Please enter at least 3 ticker symbols.")
        st.stop()

    if len(tickers) > 10:
        st.error("Please enter no more than 10 ticker symbols.")
        st.stop()

    if start_date >= end_date:
        st.error("Start date must be earlier than end date.")
        st.stop()

    if not minimum_two_years(start_date, end_date):
        st.error("Please select a date range of at least 2 years.")
        st.stop()

    with st.spinner("Downloading data and preparing analysis..."):
        prices, bad_tickers = download_price_data(tickers, start_date, end_date)

        bad_user_tickers = [t for t in bad_tickers if t != BENCHMARK_TICKER]

        if bad_user_tickers:
            st.error(
                f"These ticker(s) failed to download or returned insufficient data: {', '.join(bad_user_tickers)}"
            )

        if BENCHMARK_TICKER in bad_tickers:
            st.error("The S&P 500 benchmark (^GSPC) failed to download.")
            st.stop()

        valid_user_tickers = [t for t in tickers if t not in bad_user_tickers]

        if len(valid_user_tickers) < 3:
            st.error("After removing invalid tickers, fewer than 3 valid tickers remain.")
            st.stop()

        desired_columns = [col for col in valid_user_tickers if col in prices.columns]
        if BENCHMARK_TICKER in prices.columns:
            desired_columns.append(BENCHMARK_TICKER)

        prices = prices[desired_columns]

        prices, dropped = align_and_clean_prices(prices)

        if dropped:
            st.warning(
                f"These ticker(s) were dropped because they had more than 5% missing values: {', '.join(dropped)}"
            )

        remaining_user_tickers = [col for col in prices.columns if col != BENCHMARK_TICKER]

        if len(remaining_user_tickers) < 3:
            st.error("After cleaning missing data, fewer than 3 valid tickers remain.")
            st.stop()

        returns = compute_returns(prices)
        stock_returns = returns[remaining_user_tickers]

    # -----------------------------
    # TABS
    # -----------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Welcome",
        "Exploratory Analysis",
        "Risk & Correlation",
        "Portfolio Analysis",
        "Data Overview"
    ])

    # -----------------------------
    # TAB 1: WELCOME
    # -----------------------------
    with tab1:
        st.subheader("Project Dashboard")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Stocks Selected", len(remaining_user_tickers))
        col2.metric("Assets Including Benchmark", len(prices.columns))
        col3.metric("Observations", len(prices))
        col4.metric("Risk-Free Rate", f"{risk_free_rate:.2f}%")

        st.markdown(
            """
            Use the tabs above to explore:
            - **Exploratory Analysis** for summary statistics, wealth growth, and distributions
            - **Risk & Correlation** for rolling volatility, drawdowns, Sharpe/Sortino, and correlations
            - **Portfolio Analysis** for equal-weight, GMV, tangency, and estimation window sensitivity
            - **Data Overview** for raw price and return previews
            """
        )

        wealth_all = (1 + returns).cumprod() * 10000
        intro_fig = px.line(
            wealth_all,
            x=wealth_all.index,
            y=wealth_all.columns,
            title="Growth of $10,000 Across Selected Assets and Benchmark",
            labels={"value": "Portfolio Value ($)", "index": "Date"}
        )
        intro_fig.update_layout(legend_title_text="Assets")
        st.plotly_chart(intro_fig, use_container_width=True)

    # -----------------------------
    # TAB 2: EXPLORATORY ANALYSIS
    # -----------------------------
    with tab2:
        st.subheader("Summary Statistics")

        summary_stats = compute_summary_stats(returns)
        st.dataframe(summary_stats.style.format("{:.4f}"), use_container_width=True)

        st.subheader("Cumulative Wealth Index")

        selected_assets = st.multiselect(
            "Select assets to display",
            options=list(returns.columns),
            default=list(returns.columns),
            key="selected_assets"
        )

        if selected_assets:
            wealth = (1 + returns[selected_assets]).cumprod() * 10000

            wealth_fig = px.line(
                wealth,
                x=wealth.index,
                y=wealth.columns,
                title="Growth of $10,000",
                labels={"value": "Portfolio Value ($)", "index": "Date"}
            )
            wealth_fig.update_layout(legend_title_text="Assets")
            st.plotly_chart(wealth_fig, use_container_width=True)
        else:
            st.warning("Please select at least one asset to display.")

        st.subheader("Return Distribution Analysis")

        selected_stock = st.selectbox(
            "Choose an asset for distribution analysis",
            options=list(returns.columns),
            key="distribution_stock"
        )

        distribution_view = st.radio(
            "Choose plot type",
            ["Histogram + Normal Curve", "Q-Q Plot"],
            horizontal=True,
            key="distribution_view"
        )

        stock_returns_selected = returns[selected_stock].dropna()

        if distribution_view == "Histogram + Normal Curve":
            hist_fig = go.Figure()

            hist_fig.add_trace(
                go.Histogram(
                    x=stock_returns_selected,
                    nbinsx=50,
                    histnorm="probability density",
                    name="Daily Returns"
                )
            )

            x_vals = np.linspace(stock_returns_selected.min(), stock_returns_selected.max(), 200)
            y_vals = norm.pdf(x_vals, stock_returns_selected.mean(), stock_returns_selected.std())

            hist_fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines",
                    name="Fitted Normal Curve"
                )
            )

            hist_fig.update_layout(
                title=f"Histogram of Daily Returns: {selected_stock}",
                xaxis_title="Daily Return",
                yaxis_title="Density"
            )

            st.plotly_chart(hist_fig, use_container_width=True)

        else:
            qq_data = probplot(stock_returns_selected, dist="norm")
            theoretical_quantiles = qq_data[0][0]
            sample_quantiles = qq_data[0][1]
            slope, intercept = qq_data[1][0], qq_data[1][1]

            qq_fig = go.Figure()

            qq_fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sample_quantiles,
                    mode="markers",
                    name="Sample Quantiles"
                )
            )

            qq_fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=slope * theoretical_quantiles + intercept,
                    mode="lines",
                    name="Reference Line"
                )
            )

            qq_fig.update_layout(
                title=f"Q-Q Plot: {selected_stock}",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles"
            )

            st.plotly_chart(qq_fig, use_container_width=True)

    # -----------------------------
    # TAB 3: RISK & CORRELATION
    # -----------------------------
    with tab3:
        st.subheader("Rolling Volatility")

        vol_window = st.selectbox(
            "Select rolling volatility window",
            [30, 60, 90, 120],
            index=1,
            key="vol_window"
        )

        rolling_vol = compute_rolling_volatility(returns, vol_window)

        vol_fig = px.line(
            rolling_vol,
            x=rolling_vol.index,
            y=rolling_vol.columns,
            title=f"Rolling Annualized Volatility ({vol_window}-Day Window)",
            labels={"value": "Annualized Volatility", "index": "Date"}
        )
        st.plotly_chart(vol_fig, use_container_width=True)

        st.subheader("Drawdown Analysis")

        drawdown_stock = st.selectbox(
            "Select an asset for drawdown analysis",
            options=list(returns.columns),
            key="drawdown_stock"
        )

        drawdown_series = compute_drawdown(returns[drawdown_stock])
        max_drawdown = drawdown_series.min()

        st.metric("Maximum Drawdown", f"{max_drawdown:.2%}")

        drawdown_fig = px.line(
            x=drawdown_series.index,
            y=drawdown_series.values,
            title=f"Drawdown Chart: {drawdown_stock}",
            labels={"x": "Date", "y": "Drawdown"}
        )
        st.plotly_chart(drawdown_fig, use_container_width=True)

        st.subheader("Risk-Adjusted Metrics")

        risk_metrics = compute_risk_metrics(returns, risk_free_rate)
        st.dataframe(risk_metrics.style.format("{:.4f}"), use_container_width=True)

        st.subheader("Correlation Heatmap")

        corr_matrix = returns.corr()

        heatmap_fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlation Matrix of Daily Returns"
        )
        heatmap_fig.update_layout(
            xaxis_title="Assets",
            yaxis_title="Assets"
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)

        st.subheader("Rolling Correlation")

        col1, col2 = st.columns(2)

        with col1:
            stock1 = st.selectbox(
                "Select first asset",
                options=list(returns.columns),
                key="stock1"
            )

        with col2:
            stock2_options = [col for col in returns.columns if col != stock1]
            stock2 = st.selectbox(
                "Select second asset",
                options=stock2_options,
                key="stock2"
            )

        corr_window = st.selectbox(
            "Select rolling correlation window",
            [30, 60, 90, 120],
            index=1,
            key="corr_window"
        )

        rolling_corr = compute_rolling_correlation(returns, stock1, stock2, corr_window)

        rolling_corr_fig = px.line(
            x=rolling_corr.index,
            y=rolling_corr.values,
            title=f"Rolling Correlation: {stock1} vs {stock2} ({corr_window}-Day Window)",
            labels={"x": "Date", "y": "Correlation"}
        )
        st.plotly_chart(rolling_corr_fig, use_container_width=True)

        with st.expander("Covariance Matrix of Daily Returns"):
            cov_matrix_display = returns.cov()
            st.dataframe(cov_matrix_display.style.format("{:.6f}"), use_container_width=True)

    # -----------------------------
    # TAB 4: PORTFOLIO ANALYSIS
    # -----------------------------
    with tab4:
        st.subheader("Equal-Weight Portfolio")

        n_assets = len(remaining_user_tickers)
        equal_weights = np.array([1 / n_assets] * n_assets)
        equal_metrics = compute_portfolio_metrics(equal_weights, stock_returns, risk_free_rate)

        metric_cols = st.columns(5)
        metric_cols[0].metric("Annualized Return", f"{equal_metrics['Annualized Return']:.2%}")
        metric_cols[1].metric("Annualized Volatility", f"{equal_metrics['Annualized Volatility']:.2%}")
        metric_cols[2].metric("Sharpe Ratio", f"{equal_metrics['Sharpe Ratio']:.3f}")
        metric_cols[3].metric(
            "Sortino Ratio",
            f"{equal_metrics['Sortino Ratio']:.3f}" if pd.notna(equal_metrics["Sortino Ratio"]) else "N/A"
        )
        metric_cols[4].metric("Maximum Drawdown", f"{equal_metrics['Maximum Drawdown']:.2%}")

        eq_weight_df = build_weights_table(equal_weights, remaining_user_tickers, "Equal")
        eq_fig = px.bar(
            eq_weight_df,
            x="Ticker",
            y="Equal Weight",
            title="Equal-Weight Portfolio Weights"
        )
        st.plotly_chart(eq_fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Optimized Portfolios")

        mean_returns_full = stock_returns.mean().values
        cov_matrix_full = stock_returns.cov().values

        gmv_weights = optimize_gmv(mean_returns_full, cov_matrix_full)
        tangency_weights = optimize_tangency(mean_returns_full, cov_matrix_full, risk_free_rate)

        opt_col1, opt_col2 = st.columns(2)

        with opt_col1:
            st.markdown("#### Global Minimum Variance (GMV)")
            if gmv_weights is None:
                st.error("GMV optimization failed.")
            else:
                gmv_metrics = compute_portfolio_metrics(gmv_weights, stock_returns, risk_free_rate)

                st.dataframe(
                    build_weights_table(gmv_weights, remaining_user_tickers, "GMV").style.format(
                        {"GMV Weight": "{:.6%}"}
                    ),
                    use_container_width=True
                )

                st.metric("Annualized Return", f"{gmv_metrics['Annualized Return']:.2%}")
                st.metric("Annualized Volatility", f"{gmv_metrics['Annualized Volatility']:.2%}")

                equal_var = portfolio_variance(equal_weights, cov_matrix_full)
                gmv_var = portfolio_variance(gmv_weights, cov_matrix_full)

                st.caption(f"Equal-weight variance: {equal_var:.10f}")
                st.caption(f"GMV variance: {gmv_var:.10f}")
                st.caption(
                    "If GMV still looks close to equal weight, compare the variance values above. "
                    "If GMV variance is lower, the optimizer is working."
                )
                st.caption(
                    "The GMV portfolio can sometimes look similar to equal weight when the selected assets "
                    "have similar volatility and correlation structures."
                )

                with st.expander("Debug: Covariance Matrix Used for GMV"):
                    st.dataframe(
                        pd.DataFrame(
                            cov_matrix_full,
                            index=remaining_user_tickers,
                            columns=remaining_user_tickers
                        ).style.format("{:.8f}"),
                        use_container_width=True
                    )

        with opt_col2:
            st.markdown("#### Tangency Portfolio")
            if tangency_weights is None:
                st.error("Tangency optimization failed.")
            else:
                tangency_metrics = compute_portfolio_metrics(tangency_weights, stock_returns, risk_free_rate)
                st.dataframe(
                    build_weights_table(tangency_weights, remaining_user_tickers, "Tangency").style.format(
                        {"Tangency Weight": "{:.4%}"}
                    ),
                    use_container_width=True
                )
                st.metric("Annualized Return", f"{tangency_metrics['Annualized Return']:.2%}")
                st.metric("Annualized Volatility", f"{tangency_metrics['Annualized Volatility']:.2%}")
                st.metric("Sharpe Ratio", f"{tangency_metrics['Sharpe Ratio']:.3f}")

        st.markdown("---")
        st.subheader("Estimation Window Sensitivity Analysis")

        st.info(
            "Mean-variance optimization is sensitive to its inputs. Small changes in the historical "
            "sample used to estimate expected returns and covariances can produce very different "
            "portfolio weights. This matters because optimized portfolios are only as stable as the "
            "data used to create them."
        )

        lookback_options = get_available_lookback_options(stock_returns)

        selected_windows = st.multiselect(
            "Select lookback windows for comparison",
            options=lookback_options,
            default=lookback_options,
            key="selected_lookback_windows"
        )

        if len(selected_windows) == 0:
            st.warning("Please select at least one lookback window.")
        else:
            sensitivity_rows = []
            weight_chart_rows = []

            for window in selected_windows:
                subset_returns = get_lookback_subset(stock_returns, window)

                if len(subset_returns) < 50:
                    continue

                mu_subset = subset_returns.mean().values
                cov_subset = subset_returns.cov().values

                gmv_w = optimize_gmv(mu_subset, cov_subset)
                tan_w = optimize_tangency(mu_subset, cov_subset, risk_free_rate)

                window_label = "Full Sample" if window == "Full Sample" else f"{window}-Year"

                if gmv_w is not None:
                    gmv_metrics_window = compute_portfolio_metrics(gmv_w, subset_returns, risk_free_rate)

                    row = {
                        "Lookback Window": window_label,
                        "Portfolio": "GMV",
                        "Annualized Return": gmv_metrics_window["Annualized Return"],
                        "Annualized Volatility": gmv_metrics_window["Annualized Volatility"],
                        "Sharpe Ratio": np.nan
                    }

                    for i, ticker in enumerate(remaining_user_tickers):
                        row[f"Weight: {ticker}"] = gmv_w[i]
                        weight_chart_rows.append({
                            "Lookback Window": window_label,
                            "Portfolio": "GMV",
                            "Ticker": ticker,
                            "Weight": gmv_w[i]
                        })

                    sensitivity_rows.append(row)

                if tan_w is not None:
                    tan_metrics_window = compute_portfolio_metrics(tan_w, subset_returns, risk_free_rate)

                    row = {
                        "Lookback Window": window_label,
                        "Portfolio": "Tangency",
                        "Annualized Return": tan_metrics_window["Annualized Return"],
                        "Annualized Volatility": tan_metrics_window["Annualized Volatility"],
                        "Sharpe Ratio": tan_metrics_window["Sharpe Ratio"]
                    }

                    for i, ticker in enumerate(remaining_user_tickers):
                        row[f"Weight: {ticker}"] = tan_w[i]
                        weight_chart_rows.append({
                            "Lookback Window": window_label,
                            "Portfolio": "Tangency",
                            "Ticker": ticker,
                            "Weight": tan_w[i]
                        })

                    sensitivity_rows.append(row)

            if len(sensitivity_rows) == 0:
                st.error("Optimization failed for the selected lookback windows.")
            else:
                sensitivity_df = pd.DataFrame(sensitivity_rows)

                format_dict = {
                    "Annualized Return": "{:.2%}",
                    "Annualized Volatility": "{:.2%}",
                    "Sharpe Ratio": "{:.3f}"
                }

                for col in sensitivity_df.columns:
                    if col.startswith("Weight: "):
                        format_dict[col] = "{:.2%}"

                st.dataframe(
                    sensitivity_df.style.format(format_dict),
                    use_container_width=True
                )

                chart_df = pd.DataFrame(weight_chart_rows)

                if not chart_df.empty:
                    selected_portfolio_for_chart = st.radio(
                        "Choose optimized portfolio to visualize across windows",
                        ["GMV", "Tangency"],
                        horizontal=True,
                        key="sensitivity_chart_portfolio"
                    )

                    filtered_chart_df = chart_df[
                        chart_df["Portfolio"] == selected_portfolio_for_chart
                    ]

                    grouped_fig = px.bar(
                        filtered_chart_df,
                        x="Ticker",
                        y="Weight",
                        color="Lookback Window",
                        barmode="group",
                        title=f"{selected_portfolio_for_chart} Weights Across Lookback Windows"
                    )
                    st.plotly_chart(grouped_fig, use_container_width=True)

    # -----------------------------
    # TAB 5: DATA OVERVIEW
    # -----------------------------
    with tab5:
        st.subheader("Price Data Preview")
        st.dataframe(prices.tail(), use_container_width=True)

        st.subheader("Return Data Preview")
        st.dataframe(returns.tail(), use_container_width=True)

else:
    st.info("Enter your inputs in the sidebar and click **Run Analysis**.")

# Use "streamlit run "portfolio-app/app.py"" to run