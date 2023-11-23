# 基礎データ分析
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# 応用データ分析
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
import statsmodels.api as sm
from arch import arch_model

# インタラクション用ライブラリ
import ipywidgets as widgets
from ipywidgets import interact, IntSlider
from IPython.display import HTML, display, clear_output




'''
【関数の使い方】
❶ plot_stock_prices(title, dataframe, selected_columns, xlabel="Date", ylabel="Price", figsize=(20, 6))
    〈引数解説〉
        title: 文字列
        dataframe: データフレーム
        selected_columns: リストでグラフ化するカラムを指定(最低1つ)
        xlabel: xラベルのタイトル
        ylabel: yラベルのタイトル
        x軸: インデックスが自動的に適用される
    〈使用例〉
    plot_stock_prices(title, df1, ['StockName'], xlabel="Date", ylabel="Price", figsize=(20, 6))

❷ display_dataframes_with_buttons(dataframes_dict)
    〈解説〉
        dataframes_dict: dict, キーにデータフレームの名前、値にデータフレームオブジェクトを持つ辞書。
    〈使用例〉
        display_dataframes_with_buttons({
        'df_base_topix_da': df_base_topix_da,
        'df_base_topix_we': df_base_topix_we,
        'df_base_topix_mo': df_base_topix_mo})

❸ display_images_with_buttons(functions_dict)
    〈解説〉
        複数の画像(グラフなど)を表示させる関数をボタン切り替えで表示
    〈使用例〉
        display_images_with_buttons({
            'ボタンタイトル1': lambda: グラフ表示関数1(引数1, 引数2, 引数3),
            'ボタンタイトル2': lambda: グラフ表示関数2(引数1, 引数2, 引数3, 引数4),
            'ボタンタイトル3': lambda: グラフ表示関数3(引数1, 引数2),
            })

❸ plot_histogram(dataframe, title, xlabel, ylabel, column_name):
    〈引数解説〉
        dataframe: データフレーム
        title: 文字列
        xlabel: 文字列
        ylabel: 文字列
        column_name: 文字列
    〈使用例〉
        display_histogram(df, 'ヒストグラムのタイトル', 'X軸のタイトル', 'Y軸のタイトル', 'カラム名')

❸ perform_adf_test_on_dataframe(df, alpha=0.05)
    〈引数解説〉
        ADF検定をします
    〈使用例〉
        df_results = perform_adf_test_on_dataframe(df_stock_mo)
        display(df_results)

❸ remove_trend_log(dataframe, ['Column1', 'Column2'])
    〈引数解説〉
        対数差分処理を行います
    〈使用例〉
        df_log_diff = remove_trend_log(dataframe, ['Column1', 'Column2'])

❸ display_order_selection(model, maxlags)
    〈引数解説〉
        VARの次数を探す
    〈使用例〉
        model = sm.tsa.VAR(your_dataframe).fit(maxlags=your_maxlags)
        display_order_selection_with_interact(model, 20)

❸ # garch_volatility_df = calculate_garch_volatility(your_dataframe)
    〈引数解説〉
        各列にGARCHモデルを適用し、条件付きボラティリティを計算する関数。
        :param dataframe: データフレーム（各列が時系列データ）
        :return: 条件付きボラティリティのデータフレーム
    〈使用例〉
        garch_volatility_df = calculate_garch_volatility(your_dataframe)



'''

def plot_stock_prices(title, dataframe, selected_columns, xlabel="Date", ylabel="Price", figsize=(20, 6)):
    """
    Given a dataframe, plot the specified stock price columns using the dataframe's index as x-axis and allows setting custom x-axis and y-axis labels.

    Parameters:
    - title: str, the title of the plot.
    - dataframe: pd.DataFrame, containing stock prices with a date index.
    - selected_columns: list of str, names of stock prices to be plotted.
    - xlabel: str, the label for the x-axis.
    - ylabel: str, the label for the y-axis.
    - figsize: tuple, size of the figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for column in selected_columns:
        ax.plot(dataframe.index, dataframe[column], label=column)

    ax.set_title(title) #タイトルの設定
    ax.set_xlabel(xlabel) #xラベルタイトルの設定
    ax.set_ylabel(ylabel) #yラベルタイトルの設定
    ax.legend() # 凡例の追加
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=12)) # x軸のメジャー数の上限を12個に設定
    plt.tight_layout()
    plt.show()

def display_dataframes_with_buttons(dataframes_dict):
    """
    複数のデータフレームを表示するためのボタンを作成し、
    ボタンをクリックすると対応するデータフレームが表示される関数。

    引数:
    dataframes_dict: dict, キーにデータフレームの名前、値にデータフレームオブジェクトを持つ辞書。
    """

    buttons = []
    rows = []  # ボタンの行を保持するリスト

    # 各データフレームに対応するボタンを生成（横幅を30%に設定）
    for name in dataframes_dict.keys():
        button = widgets.Button(description=name, layout=widgets.Layout(width='40%', margin='5px'))
        buttons.append(button)

    # 1行当たり2つのボタンを中央揃えで配置
    for i in range(0, len(buttons), 2):
        row = widgets.HBox(buttons[i:i+2], layout=widgets.Layout(justify_content='center'))
        rows.append(row)

    # ボタンのイベントハンドラー
    def on_button_clicked(b):
        clear_output(wait=True)
        for row in rows:
            display(row)
        display(dataframes_dict[b.description])

    # 各ボタンにイベントハンドラーを割り当てる
    for button in buttons:
        button.on_click(on_button_clicked)

    # すべてのボタンを表示
    for row in rows:
        display(row)

def display_images_with_buttons(functions_dict):
    """
    複数のグラフ表示関数を実行するためのボタンを作成し、
    ボタンをクリックすると対応する関数が実行される関数。

    引数:
    functions_dict: dict, キーにボタンのタイトル、値に実行するグラフ表示関数を持つ辞書。
    """

    buttons = []
    rows = []  # ボタンの行を保持するリスト

    # 各関数に対応するボタンを生成（横幅を40%に設定）
    for title in functions_dict.keys():
        button = widgets.Button(description=title, layout=widgets.Layout(width='40%', margin='5px'))
        buttons.append(button)

    # 1行当たり2つのボタンを中央揃えで配置
    for i in range(0, len(buttons), 2):
        row = widgets.HBox(buttons[i:i+2], layout=widgets.Layout(justify_content='center'))
        rows.append(row)

    # ボタンのイベントハンドラー
    def on_button_clicked(b):
        clear_output(wait=True)
        for row in rows:
            display(row)
        functions_dict[b.description]()

    # 各ボタンにイベントハンドラーを割り当てる
    for button in buttons:
        button.on_click(on_button_clicked)

    # すべてのボタンを表示
    for row in rows:
        display(row)

def plot_histogram(dataframe, title, xlabel, ylabel, column_name):
    """
    指定されたデータフレームの指定されたカラムに対するヒストグラムを表示する関数。
    Seabornを使用してグラフのスタイルを改善します。

    引数:
    dataframe: pd.DataFrame, データフレーム。
    title: str, グラフのタイトル。
    xlabel: str, x軸のラベルのタイトル。
    ylabel: str, y軸のラベルのタイトル。
    column_name: str, データのカラム名。
    """
    # ヒストグラムを描画
    plt.figure(figsize=(10, 6))
    sns.histplot(dataframe[column_name],
                kde=True,
                bins='auto',
                # cumulative='True',
                )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def perform_adf_test_on_dataframe(df, alpha=0.05):
    # NaNの値を確認し、存在する場合はドロップ
    df.dropna(inplace=True)
    
    # 無限の値を確認し、存在する場合はNaNで置き換えてからドロップ
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    def adf_test(series):
        # 再度、シリーズにNaNまたは無限の値がないことを確認
        if series.isnull().sum() > 0 or np.isinf(series).sum() > 0:
            raise ValueError(f"Series contains NaN or infinite values")
            
        result = adfuller(series, autolag='AIC')
        test_statistic = result[0]
        p_value = result[1]
        lags_used = result[2]
        nobs = result[3]
        critical_values = result[4]
        result_dict = {
            "ADF Test Statistic": test_statistic,
            "P-Value": p_value,
            "# Lags Used": lags_used,
            "# Observations": nobs,
            "Result": "Stationary" if p_value <= alpha else "Not Stationary"
        }
        for key, value in critical_values.items():
            result_dict[f'Critical Value ({key})'] = value
        return result_dict
    
    results = {col: adf_test(df[col]) for col in df.columns if col != 'Date'}
    return pd.DataFrame(results).T

def remove_trend_log(data, selected_columns=None) -> pd.DataFrame:
    """
    指定されたDataFrameまたはSeriesに対して対数差分を計算します。
    
    パラメーター:
    - data: pd.DataFrame or pd.Series, 一つ以上の時系列列を持つ入力データ。
    - selected_columns: list, 対数差分を計算する列。Noneの場合、'Date'を除く全ての列に対して計算します。

    戻り値:
    - pd.DataFrame or pd.Series, 元のデータの対数差分。
    """
    
    # dataがSeriesの場合
    if isinstance(data, pd.Series):
        return np.log(data).diff().dropna()
    
    # dataがDataFrameの場合
    elif isinstance(data, pd.DataFrame):
        
        # 特定の列が選択されていない場合は、'Date'を除く全ての列を選択
        if selected_columns is None:
            selected_columns = [col for col in data.columns if col != 'Date']
        
        # 選択された各列に対して対数差分を適用
        for col in selected_columns:
            data[col] = np.log(data[col]).diff()
        
        return data.dropna()
    
    else:
        raise ValueError("入力データはpandasのDataFrameまたはSeriesでなければなりません")

def display_order_selection(model, maxlags):
    """
    Automatically display the VAR model order selection using interactive widgets.
    
    Parameters:
    model : VAR model instance
        An instance of the VAR model from statsmodels.
    maxlags : int
        The maximum number of lags to consider for the VAR model order selection.
    """
    def display_interact(maxlags):
        results = model.select_order(maxlags=maxlags)
        summary_html = results.summary().as_html()
        center_aligned_html = f"<div style='display: flex; justify-content: center;'>{summary_html}</div>"
        display(HTML(center_aligned_html))
    
    # スライダーを作成し、display_order_selection関数を動的に実行する
    interact(display_interact, maxlags=IntSlider(min=1, max=maxlags, step=1, value=min(5, maxlags)))

def calculate_garch_volatility(dataframe):
    volatility_df = pd.DataFrame(index=dataframe.index)

    for column in dataframe.columns:
        # ログ収益率の計算
        returns = 100 * dataframe[column].pct_change().dropna()

        # GARCHモデルの適用
        model = arch_model(returns, vol='Garch', p=1, q=1)
        model_fit = model.fit(disp='off')  # 計算の詳細を表示しない

        # 条件付きボラティリティの計算
        volatility_df[column] = model_fit.conditional_volatility

    # NaNを含む行を削除
    return volatility_df.dropna()
