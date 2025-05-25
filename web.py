import dash
from dash import html, dcc, Input, Output, State, dash_table
import pandas as pd
import base64
import io
import subprocess
import os
import plotly.express as px
import ast
import networkx as nx
import plotly.graph_objects as go

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "AOGA Optimizer"
server = app.server

OD_FILE = "uploaded_od.csv"
EDGE_FILE = "uploaded_edges.csv"

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Page 1: Input Form
input_layout = html.Div([
    html.H1("🚌 AOGA 多目標運輸網路優化工具", style={'marginBottom': '10px'}),
    html.P("請依序上傳 OD 矩陣與邊表，設定參數後點擊下方按鈕執行模擬。"),
    html.A("👉 查看結果分析 Dashboard", href="/dashboard", style={'color': 'blue'}),
    html.Hr(),

    html.Label("Upload OD Matrix (CSV):"),
    dcc.Upload(id='upload-od', children=html.Button('Upload OD Matrix'), multiple=False),
    html.Div(id='od-status'),

    html.Br(),
    html.Label("Upload Edge List (CSV with columns: from,to,weight):"),
    dcc.Upload(id='upload-edges', children=html.Button('Upload Edge List'), multiple=False),
    html.Div(id='edge-status'),

    html.Br(),
    html.Label("Detour Factor Range:"),
    dcc.RangeSlider(id='detour-slider', min=1.0, max=2.0, step=0.1, value=[1.0, 1.5],
                    marks={i: str(i) for i in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]}),
    html.Br(),
    html.Label("Bus Capacity:"),
    dcc.Input(id='bus-capacity', type='number', value=40),
    html.Br(),
    html.Label("Load Factor:"),
    dcc.Input(id='load-factor', type='number', value=1.25, step=0.05),
    html.Br(), html.Br(),
    html.Button('Run AOGA Optimization', id='run-button'),
    dcc.Loading(html.Div(id='run-status'), type='circle')
])

# Route Parser
def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode('utf-8')))

# Upload OD
@app.callback(
    Output('od-status', 'children'),
    Input('upload-od', 'contents'),
    prevent_initial_call=True
)
def handle_od_upload(contents):
    try:
        df = parse_contents(contents)
        df.to_csv(OD_FILE, index=False, encoding='utf-8-sig')
        return f"✅ OD Matrix Uploaded. Shape: {df.shape}"
    except Exception as e:
        return f"❌ Error loading OD Matrix: {str(e)}"

# Upload Edge
@app.callback(
    Output('edge-status', 'children'),
    Input('upload-edges', 'contents'),
    prevent_initial_call=True
)
def handle_edge_upload(contents):
    try:
        df = parse_contents(contents)
        df.to_csv(EDGE_FILE, index=False, encoding='utf-8-sig')
        return f"✅ Edge List Uploaded. Shape: {df.shape}"
    except Exception as e:
        return f"❌ Error loading Edge List: {str(e)}"

# ✅ Run AOGA - 顯示詳細錯誤（加上 stderr 輸出）
@app.callback(
    Output('run-status', 'children'),
    Input('run-button', 'n_clicks'),
    State('detour-slider', 'value'),
    State('bus-capacity', 'value'),
    State('load-factor', 'value'),
    prevent_initial_call=True
)
def run_solver(n_clicks, detour_range, capacity, load):
    try:
        args = ["python", "main.py",
                f"--od={OD_FILE}",
                f"--edge={EDGE_FILE}",
                f"--detour_min={detour_range[0]}",
                f"--detour_max={detour_range[1]}",
                f"--capacity={capacity}",
                f"--load_factor={load}"]

        result = subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        return html.Div([
            html.P("✅ 模擬執行完成！"),
            html.A("👉 點此前往結果 Dashboard", href="/dashboard", style={'color': 'blue'})
        ])
    except subprocess.CalledProcessError as e:
        return html.Pre(f"❌ 執行錯誤：\n{e.stderr}")

# 顯示 input 頁面
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    return input_layout if pathname != "/dashboard" else html.Div([html.P("（Dashboard 實作省略）")])

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
