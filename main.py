#!/usr/bin/env python

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.express as px
import dash
import plotly.graph_objects as go
from dash import dcc, html, Input, Output

# Load data
file_path = "demand.parquet"  # Adjust as needed
df = pd.read_parquet(file_path)

# Convert date from milliseconds to datetime
df['date'] = pd.to_datetime(df['date'], unit='ms')

# Get the first product code only
first_product = df['prod_code'].dropna().unique()[0]

# Parameters
initial_stock = 2000
order_day = "Thursday"
lookback_period = 30  # Giorni usati per il calcolo del riordino
Z_score = 1.64  # 95% service level

def calculate_safety_stock(demand_series, lookback_period=30, lead_time=4, Z=1.64):
    """ Calcola lo stock di sicurezza basato sulla variabilità della domanda. """
    std_demand = demand_series['demand'].rolling(lookback_period, min_periods=1).std()
    return Z * std_demand.iloc[-1] * np.sqrt(lead_time)  # Safety Stock Formula


def predict_demand(demand_series, lookback_period=30, lead_time=4):
    """ Calculate reorder quantity based on moving average of past demand. """
    avg_demand = demand_series['demand'].rolling(lookback_period, min_periods=1).mean()
    return avg_demand.iloc[-1] * lead_time  # Forecast demand for lead time

def simulate_replenishment(demand_series, lead_time=4):
    stock = initial_stock
    orders = []
    pending_orders = []
    results = []
    
    for i, row in demand_series.iterrows():
        date, demand = row['date'], row['demand']
        stock -= demand  # Deduct demand
        if stock < 0:
            stock = 0
        
        # Process incoming orders FIRST
        for arrival, qty in pending_orders:
            if arrival <= date:
                stock += qty
        
        # Remove only the orders that have been fulfilled
        pending_orders = [(arrival, qty) for arrival, qty in pending_orders if arrival > date]

        
        # Calculate reorder quantity dynamically
        reorder_quantity = predict_demand(demand_series[demand_series['date'] <= date], lookback_period, lead_time)
        safety_stock = calculate_safety_stock(demand_series[demand_series['date'] <= date], lookback_period, lead_time, Z_score)
        
        # Evita nuovi ordini se ne è già stato inviato uno in attesa di arrivo
        if len(pending_orders) > 0:
            results.append({'date': date, 
                            'stock': stock, 
                            'demand': demand, 
                            'orders_placed': len(orders), 
                            'reorder_point': reorder_quantity, 
                            'safety_stock': safety_stock})
            continue

        # Order based on hybrid strategy (Thursday or below safety stock)
        if stock < safety_stock or (stock < reorder_quantity and date.strftime('%A') == order_day):
            orders.append((date, reorder_quantity))
            pending_orders.append((date + timedelta(days=lead_time), reorder_quantity))
        
        results.append({'date': date, 
                        'stock': stock, 
                        'demand': demand, 
                        'orders_placed': len(orders), 
                        'reorder_point': reorder_quantity, 
                        'safety_stock': safety_stock})
    
    return pd.DataFrame(results)

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Simulatore: Impatto del Lead Time sulla Gestione delle Scorte"),
    html.P(f"Questo strumento interattivo dimostra come il lead time del fornitore influisce sui livelli di scorta."),
    html.P("Strategia di Riordino:"),
    html.Ul([
        html.Li("Calendarizzazione: Ordini emessi ogni giovedì se le scorte scendono sotto il punto di riordino"),
        html.Li("Trigger di sicurezza: Ordini immediati se le scorte scendono sotto il livello di scorta di sicurezza"),
        html.Li("Politica di ordine singolo: Nessun nuovo ordine fino all'arrivo dell'ordine precedente")
    ]),
    html.P("Metriche Chiave:"),
    html.Ul([
        html.Li("Punto di Riordino = media della domanda degli ultimi 30 giorni × lead time"),
        html.Li("Scorta di Sicurezza = Z-score (1.64) × deviazione standard della domanda × √lead time")
    ]),
    html.P("Regola il cursore del lead time qui sotto per visualizzare il suo impatto sulle rotture di stock e sui livelli di inventario."),
    
    html.Div([
        html.Label("Lead Time (giorni):"),
        dcc.Slider(
            id='lead-time-slider',
            min=1,
            max=14,
            step=1,
            value=4,
            marks={i: str(i) for i in range(1, 15, 1)},
        ),
    ], style={'padding': '20px'}),
    
    # Add Loading Component
    dcc.Loading(
        id="loading-1",
        type="default",
        children=[
            dcc.Graph(id='simulation-graph'),
        ]
    ),
    
    html.Hr(),
    html.P("Codice sviluppato da Alex Mina - [GitHub](https://github.com/animalecs)", style={'text-align': 'center'})
])

server = app.server  # Necessario per Railway

@app.callback(
    Output('simulation-graph', 'figure'),
    Input('lead-time-slider', 'value')
)
def update_graph(lead_time):
    # Use the first product only
    product_data = df[df['prod_code'] == first_product].sort_values(by='date')
    demand_series = product_data[['date', 'demand']].copy()
    demand_series['demand'] = demand_series['demand'].fillna(0)
    start_date = demand_series['date'].min() + timedelta(days=lookback_period)
    demand_series = demand_series[demand_series['date'] >= start_date]
    
    # Run simulation with the selected lead time
    sim = simulate_replenishment(demand_series, lead_time=lead_time)
    
    # Filter last year for chart
    end_date = demand_series['date'].max()
    start_date = end_date - timedelta(days=365)
    sim_filtered = sim[sim['date'] >= start_date]
    
    # Create figure
    fig = px.line(sim_filtered, x='date', y=['stock', 'demand', 'reorder_point', 'safety_stock'], 
                 #title=f"Prodotto: {first_product} - Lead Time: {lead_time} giorni"
                 )

    # Customize line styles
    for trace in fig.data:
        if trace.name == 'reorder_point' or trace.name == 'safety_stock':
            trace.line.dash = 'dash'
            trace.line.width = 1
        else:
            trace.line.width = 2  # Keep stock and demand more visible

    # Generate ALL Thursdays within the dataset's time range
    start_date = sim_filtered['date'].min()
    end_date = sim_filtered['date'].max()
    
    # Create a list of all Thursdays between start_date and end_date
    all_thursdays = pd.date_range(start=start_date, end=end_date, freq='W-THU')

    # Add "X" markers on all Thursdays
    fig.add_trace(go.Scatter(
        x=all_thursdays,
        y=[sim_filtered['stock'].min()] * len(all_thursdays),  # Place markers at the bottom
        mode='text',
        text="X",
        textposition="bottom center",
        marker=dict(color="red", size=12),
        showlegend=False
    ))

    # Stockout days annotation (bold & highlighted)
    stockout_days = (sim_filtered['stock'] == 0).sum()
    fig.add_annotation(
        x=sim_filtered['date'].iloc[-1], y=sim_filtered['stock'].iloc[-1],
        text=f"<b>Stockout Days: {stockout_days}</b>",
        showarrow=False,
        font=dict(color='white', size=16, family="Arial Black"),
        align="center",
        bgcolor="red",
        bordercolor="black",
        borderwidth=2
    )

    return fig

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=5001)