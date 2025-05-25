import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import ast
import networkx as nx
import plotly.graph_objects as go

# Load combined results CSV
df = pd.read_csv("results_all.csv")

# Derived columns
transfer_total = df[['0-transfer', '1-transfer', '2-transfer', 'unserved']].sum(axis=1)
df['zero_transfer_ratio'] = df['0-transfer'] / (transfer_total + 1e-6)
df['solution_type'] = df['user_cost'].apply(lambda x: 'Users' if x < 300000 else 'Operators')

# Add parsed route preview (first 3 routes per solution)
def route_preview(text):
    try:
        routes = ast.literal_eval(text)
        return " | ".join(["→".join(map(str, r)) for r in routes[:3]])
    except:
        return ""

df['routes_preview'] = df['routes'].apply(route_preview)

# Pareto front flagging
def is_pareto(e, others):
    for _, o in others.iterrows():
        if (o['fleet_size'] <= e['fleet_size'] and o['user_cost'] <= e['user_cost'] and
            (o['fleet_size'] < e['fleet_size'] or o['user_cost'] < e['user_cost'])):
            return False
    return True

df['is_pareto'] = df.apply(lambda row: is_pareto(row, df), axis=1)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("AOGA Full Analytics Dashboard"),

    html.Label("Select Detour Factor:"),
    dcc.Dropdown(
        id='detour-dropdown',
        options=[{"label": f"{x:.1f}", "value": x} for x in sorted(df['detour_factor'].unique())],
        value=None,
        placeholder="All detour factors"
    ),

    html.Label("Select Generation Range:"),
    dcc.RangeSlider(
        id='gen-slider',
        min=df['generation'].min(),
        max=df['generation'].max(),
        step=1,
        value=[df['generation'].min(), df['generation'].max()],
        marks={int(i): str(int(i)) for i in range(0, df['generation'].max()+1, 10)},
        tooltip={"placement": "bottom", "always_visible": False}
    ),

    html.Label("Toggle Pareto Front Only:"),
    dcc.Checklist(
        id='pareto-toggle',
        options=[{'label': ' Show only Pareto optimal solutions', 'value': 'pareto'}],
        value=[]
    ),

    html.Div([
        dcc.Graph(id='fleet-user-scatter'),
        dcc.Graph(id='user-cost-hist')
    ], style={'display': 'flex', 'gap': '20px'}),

    html.Div([
        html.Label("Select a Row to View Route Graph:"),
        dash_table.DataTable(
            id='summary-table',
            columns=[
                {"name": i, "id": i} for i in df.columns if i != 'routes'
            ],
            data=df.to_dict('records'),
            page_size=12,
            row_selectable='single',
            selected_rows=[],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center', 'fontSize': 12, 'padding': '5px'}
        )
    ]),

    dcc.Graph(id='route-network-graph')
])

@app.callback(
    [Output('fleet-user-scatter', 'figure'),
     Output('user-cost-hist', 'figure'),
     Output('summary-table', 'data')],
    [Input('detour-dropdown', 'value'),
     Input('gen-slider', 'value'),
     Input('pareto-toggle', 'value')]
)
def update_filtered_data(detour_filter, gen_range, pareto_toggle):
    dff = df.copy()
    if detour_filter is not None:
        dff = dff[dff['detour_factor'] == detour_filter]
    dff = dff[(dff['generation'] >= gen_range[0]) & (dff['generation'] <= gen_range[1])]
    if 'pareto' in pareto_toggle:
        dff = dff[dff['is_pareto']]
    
    fig1 = px.scatter(dff, x='fleet_size', y='user_cost', title='Fleet vs User Cost')
    fig2 = px.histogram(dff, x='user_cost', nbins=25, title='User Cost Distribution')
    return fig1, fig2, dff.to_dict('records')

@app.callback(
    Output('route-network-graph', 'figure'),
    [Input('summary-table', 'data'),
     Input('summary-table', 'selected_rows')]
)
def draw_route(data, selected):
    if not selected or not data:
        return go.Figure()

    row = data[selected[0]]
    try:
        routes = ast.literal_eval(row['routes'])
    except:
        return go.Figure()

    # Create a sample graph structure
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
            line_width=2)
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title='Selected Solution Route Graph', showlegend=False,
                      margin=dict(l=20, r=20, t=40, b=20))
    return fig

if __name__ == '__main__':
    app.run(debug=True)