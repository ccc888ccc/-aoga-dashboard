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

# ---------- Page 1: Input Form ----------
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

# ---------- Page 2: Dashboard ----------
def dashboard_page():

    try:
        df = pd.read_csv("results_all.csv")
        df['zero_transfer_ratio'] = df['0-transfer'] / (df[['0-transfer', '1-transfer', '2-transfer', 'unserved']].sum(axis=1) + 1e-6)
        df['solution_type'] = df['user_cost'].apply(lambda x: 'Users' if x < 300000 else 'Operators')

        def route_preview(text):
            try:
                routes = ast.literal_eval(text)
                return " | ".join(["→".join(map(str, r)) for r in routes[:3]])
            except:
                return ""

        df['routes_preview'] = df['routes'].apply(route_preview)

        def is_pareto(e, others):
            for _, o in others.iterrows():
                if (o['fleet_size'] <= e['fleet_size'] and o['user_cost'] <= e['user_cost'] and
                    (o['fleet_size'] < e['fleet_size'] or o['user_cost'] < e['user_cost'])):
                    return False
            return True

        df['is_pareto'] = df.apply(lambda row: is_pareto(row, df), axis=1)

        df_sorted = df.sort_values(by=['fleet_size', 'user_cost'])

        return html.Div([
            html.H2("📊 AOGA 結果分析 Dashboard"),
            html.A("🔙 回到模擬設定頁", href="/", style={'color': 'blue'}),
            html.Hr(),

            html.Label("Toggle Pareto Front Only:"),
            dcc.Checklist(
                id='pareto-toggle',
                options=[{'label': ' Show only Pareto optimal solutions', 'value': 'pareto'}],
                value=[]
            ),

            dcc.Graph(id='fleet-user-scatter'),

            html.Label("Select a Row to View Route Graph:"),
            dash_table.DataTable(
                id='summary-table',
                columns=[
                    {"name": i, "id": i} for i in df.columns if i != 'routes'
                ],
                data=df_sorted.to_dict('records'),
                page_size=12,
                row_selectable='single',
                selected_rows=[],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center', 'fontSize': 12, 'padding': '5px'}
            ),

            dcc.Graph(id='route-network-graph'),
            
            html.Pre(id='route-list-output', style={
                'whiteSpace': 'pre-wrap',
                'fontFamily': 'monospace',
                'marginTop': '20px'
            })
        ])
    except Exception as e:
        return html.Div([html.P(f"❌ 無法載入分析資料: {e}")])

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/dashboard':
        return dashboard_page()
    else:
        return input_layout

@app.callback(
    [Output('fleet-user-scatter', 'figure'),
     Output('summary-table', 'data')],
    Input('pareto-toggle', 'value')
)
def update_charts(pareto_toggle):
    df = pd.read_csv("results_all.csv")
    df['zero_transfer_ratio'] = df['0-transfer'] / (df[['0-transfer', '1-transfer', '2-transfer', 'unserved']].sum(axis=1) + 1e-6)
    df['solution_type'] = df['user_cost'].apply(lambda x: 'Users' if x < 300000 else 'Operators')

    def route_preview(text):
        try:
            routes = ast.literal_eval(text)
            return " | ".join(["→".join(map(str, r)) for r in routes[:3]])
        except:
            return ""

    df['routes_preview'] = df['routes'].apply(route_preview)

    def is_pareto(e, others):
        for _, o in others.iterrows():
            if (o['fleet_size'] <= e['fleet_size'] and o['user_cost'] <= e['user_cost'] and
                (o['fleet_size'] < e['fleet_size'] or o['user_cost'] < e['user_cost'])):
                return False
        return True

    df['is_pareto'] = df.apply(lambda row: is_pareto(row, df), axis=1)

    if 'pareto' in pareto_toggle:
        df = df[df['is_pareto']]

    df_sorted = df.sort_values(by=['fleet_size', 'user_cost'])

    fig = px.scatter(
        df_sorted,
        x='fleet_size',
        y='user_cost',
        hover_data=['generation', 'detour_factor', 'user_cost', 'fleet_size'],
        title='Fleet vs User Cost'
    )
    return fig, df_sorted.to_dict('records')

# 2️⃣ 修改 draw_route callback，增加 route_list_output 回傳
@app.callback(
    [Output('route-network-graph', 'figure'),
     Output('route-list-output', 'children')],
    [Input('summary-table', 'data'),
     Input('summary-table', 'selected_rows')]
)
def draw_route(data, selected):
    if not selected or not data:
        return go.Figure(), ""
    
    row = data[selected[0]]
    try:
        routes = ast.literal_eval(row['routes'])
    except:
        return go.Figure(), ""

    # Build network graph
    G = nx.Graph()
    for route in routes:
        for i in range(len(route) - 1):
            G.add_edge(route[i], route[i+1])

    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x, node_y, texts = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        texts.append(str(node))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=texts,
        textposition="bottom center",
        marker=dict(
            showscale=False,
            color='#00CC96',
            size=10,
            line_width=2
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title='Selected Solution Route Graph', showlegend=False,
                      margin=dict(l=20, r=20, t=40, b=20))

    # 3️⃣ Route text output
    route_text = "\n".join([f"Route {i+1}: {' → '.join(map(str, r))}" for i, r in enumerate(routes)])
    return fig, route_text



def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode('utf-8')))

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

        subprocess.run(args, check=True, stderr=subprocess.PIPE)
        return html.Div([
            html.P("✅ 模擬執行完成！"),
            html.A("👉 點此前往結果 Dashboard", href="/dashboard", style={'color': 'blue'})
        ])
    except Exception as e:
        return f"❌ 執行錯誤：{str(e)}"

@app.callback(
    Output('summary-table', 'selected_rows'),
    Input('fleet-user-scatter', 'clickData'),
    State('summary-table', 'data')
)
def sync_click_to_row(clickData, table_data):
    if not clickData:
        return []
    point = clickData['points'][0]
    fleet = point['x']
    cost = point['y']
    for i, row in enumerate(table_data):
        if abs(row['fleet_size'] - fleet) < 1e-3 and abs(row['user_cost'] - cost) < 1e-3:
            return [i]
    return []


import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
