import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
# インタラクション用ライブラリ
import ipywidgets as widgets
from IPython.display import display, clear_output


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
        display_functions_with_buttons({
            'ボタンタイトル1': lambda: グラフ表示関数1(引数1, 引数2, 引数3),
            'ボタンタイトル2': lambda: グラフ表示関数2(引数1, 引数2, 引数3, 引数4),
            'ボタンタイトル3': lambda: グラフ表示関数3(引数1, 引数2),
            })


❸ 
    〈引数解説〉

    〈使用例〉

❸ 
    〈引数解説〉

    〈使用例〉

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

    ax.set_title(f'Stock Price over Time ({title})')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, axis="y")
    ax.grid(False, axis="x")
    for spine in ax.spines.values():
        spine.set_color('gray')
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=12))
    plt.style.use('_mpl-gallery')
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

    # 各データフレームに対応するボタンを生成
    for name in dataframes_dict.keys():
        button = widgets.Button(description=name)
        buttons.append(button)

    # ボタンのイベントハンドラー
    def on_button_clicked(b):
        clear_output(wait=True)
        for button in buttons:
            display(button)
        display(dataframes_dict[b.description])

    # 各ボタンにイベントハンドラーを割り当てる
    for button in buttons:
        button.on_click(on_button_clicked)

    # すべてのボタンを表示
    for button in buttons:
        display(button)

def display_images_with_buttons(functions_dict):
    """
    複数のグラフ表示関数を実行するためのボタンを作成し、
    ボタンをクリックすると対応する関数が実行される関数。

    引数:
    functions_dict: dict, キーにボタンのタイトル、値に実行するグラフ表示関数を持つ辞書。
    """

    buttons = []

    # 各関数に対応するボタンを生成
    for title in functions_dict.keys():
        button = widgets.Button(description=title)
        buttons.append(button)

    # ボタンのイベントハンドラー
    def on_button_clicked(b):
        clear_output(wait=True)
        for button in buttons:
            display(button)
        functions_dict[b.description]()

    # 各ボタンにイベントハンドラーを割り当てる
    for button in buttons:
        button.on_click(on_button_clicked)

    # すべてのボタンを表示
    for button in buttons:
        display(button)


def clean_dataframe(df):
    """
    Cleans the dataframe by dropping NaN and infinite values.

    Parameters:
    - df: pd.DataFrame, the dataframe to be cleaned.
    
    Returns:
    - pd.DataFrame, cleaned dataframe.
    """
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def adf_test(series, alpha=0.05):
    """
    Performs the Augmented Dickey-Fuller test on a time series.

    Parameters:
    - series: pd.Series, the time series to test.
    - alpha: float, significance level for the test.

    Returns:
    - dict, results of the ADF test.
    """
    if series.isnull().sum() > 0 or np.isinf(series).sum() > 0:
        raise ValueError("Series contains NaN or infinite values")
    
    result = adfuller(series, autolag='AIC')
    critical_values = {f'Critical Value ({key})': value for key, value in result[4].items()}
    return {
        "ADF Test Statistic": result[0],
        "P-Value": result[1],
        "# Lags Used": result[2],
        "# Observations": result[3],
        "Result": "Stationary" if result[1] <= alpha else "Not Stationary",
        **critical_values
    }

def perform_adf_test_on_dataframe(df, alpha=0.05):
    """
    Performs the ADF test on each column of the dataframe except 'Date'.

    Parameters:
    - df: pd.DataFrame, the dataframe to test.
    - alpha: float, significance level for the test.

    Returns:
    - pd.DataFrame, results of the ADF tests.
    """
    df = clean_dataframe(df)
    results = {col: adf_test(df[col], alpha) for col in df.columns if col != 'Date'}
    return pd.DataFrame(results).T
