import dash
from dash import dcc, html
from dash.dependencies import Input, Output

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "EA Sports Cricket Menu"

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-image: url("/assets/menu_background.png");
                background-size: cover;
                background-repeat: no-repeat;
                background-position: center;
                font-family: 'Segoe UI', sans-serif;
            }
            .menu-box {
                background-color: rgba(0, 0, 0, 0.6);
                padding: 30px;
                border-radius: 12px;
                width: 400px;
                margin: 180px auto;
                text-align: center;
                box-shadow: 0 4px 25px rgba(0,0,0,0.5);
                color: white;
            }
            .menu-box h1 {
                font-size: 32px;
                margin-bottom: 25px;
                color: #ffe600;
                text-shadow: 2px 2px 5px #000;
            }
            .menu-box .dash-dropdown {
                background-color: #222 !important;
                border: 1px solid #555 !important;
                color: #fff !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([
    dcc.Location(id='url-handler'),
    html.Div([
        html.H1("EA Sports Cricket 2007"),
        dcc.Dropdown(
            id='menu-dropdown',
            options=[
                {"label": "🏏 IPL Stats & Predictor", "value": "ipl"},
                {"label": "🌍 T20I Stats & Predictor", "value": "t20"}
            ],
            placeholder="Choose a Dashboard...",
            className="dash-dropdown"
        )
    ], className="menu-box")
])

@app.callback(
    Output('url-handler', 'href'),
    Input('menu-dropdown', 'value')
)
def open_dashboard(selection):
    if selection == "ipl":
        return "http://localhost:8050"
    elif selection == "t20":
        return "http://localhost:8501"
    return dash.no_update

if __name__ == "__main__":
    app.run(port=10004)
