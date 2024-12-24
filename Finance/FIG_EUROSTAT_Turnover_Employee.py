# Libraries
# =================================
import pandas as pd
from eurostat import get_data_df
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Extract Data
# =================================
# Get dataset from EUROSTAT
df = get_data_df('sbs_sc_ovw')

# Filter Dataframe
df = df[df['geo\\TIME_PERIOD'] == 'EU27_2020']
df = df[~df['size_emp'].isin(['TOTAL'])]
df = df[df['indic_sbs'].isin(['ENT_NR', 'EMP_NR', 'NETTUR_MEUR'])]
df = df[df['nace_r2'] == 'B-S_X_O_S94']

# Data Manipulation
# =================================
# Selection and rename columns
df = df.drop(columns=['freq', '2021', '2022'], errors='ignore')
df = df.rename(columns={
    'indic_sbs': 'indicator', 
    'nace_r2': 'economy',
    'size_emp': 'size',
    'geo\\TIME_PERIOD': 'country',
    '2023': 'value'})

# Replace indicator values
df['indicator'] = df['indicator'].replace({
    'EMP_NR': 'employees',
    'ENT_NR': 'companies',
    'NETTUR_MEUR': 'turnover'
})

# Replace economy values
df['economy'] = df['economy'].replace({
    'B-S_X_O_S94': 'Industry'
})

# Replace country values
df['country'] = df['country'].replace({
    'EU27_2020': 'Europe'
})

# Pivot columns
df = df.pivot_table(index=['size'], columns='indicator', values='value', aggfunc='sum').reset_index()

# Add columns
df['turnover_employee'] = (df['turnover'] * 1000000) // df['employees']

print(df)

# Data Visualizaiton
# =================================
# Create a figure
fig = go.Figure()

# Define Bar Colors
colors = ['#153D64', '#215C98', '#4D93D9', '#219E9B']

# Add 'companies'
fig.add_trace(go.Bar(
    x=df['size'],
    y=df['companies'],
    name='Number of Companies',
    marker_color=colors[0],
    visible=True,
    texttemplate='%{y:,}',
    hovertemplate='%{y:,.0f}'
))

# Add 'employees'
fig.add_trace(go.Bar(
    x=df['size'],
    y=df['employees'],
    name='Number of Employees',
    marker_color=colors[1],
    visible=False,
    texttemplate='%{y:,}',
    hovertemplate='%{y:,.0f}'
))

# Add 'turnover'
fig.add_trace(go.Bar(
    x=df['size'],
    y=df['turnover'],
    name='Turnover in €',
    marker_color=colors[2],
    visible=False,
    texttemplate='%{y:,}',
    hovertemplate='%{y:,.0f}'
))

# Add 'turnover_employee'
fig.add_trace(go.Bar(
    x=df['size'],
    y=df['turnover_employee'],
    name='Turnover per Employee',
    marker_color=colors[3],
    visible=False,
    texttemplate='%{y:,}',
    hovertemplate='%{y:,.0f}'
))

# Configurate layout
fig.update_layout(
    title='',
    title_x=1,
    title_xanchor='right',
    xaxis_title='Size',
    yaxis_title='Value',
    title_font=dict(size=18, family="Arial, sans-serif", color="black", weight='bold'),
    xaxis_title_font=dict(size=14, family="Arial, sans-serif", weight='bold'),
    yaxis_title_font=dict(size=14, family="Arial, sans-serif", weight='bold'),
    barmode='group',
    plot_bgcolor='white',
    paper_bgcolor='white',
    updatemenus=[{
        'buttons': [
            {
                'label': 'Companies',
                'method': 'update',
                'args': [{'visible': [True, False, False, False]}, {'title': 'Number of Companies by Size'}]
            },
            {
                'label': 'Employees',
                'method': 'update',
                'args': [{'visible': [False, True, False, False]}, {'title': 'Number of Employees by Size'}]
            },
            {
                'label': 'Turnover',
                'method': 'update',
                'args': [{'visible': [False, False, True, False]}, {'title': 'Total Turnover (€) by Size'}]
            },
            {
                'label': 'Turnover per Employee',
                'method': 'update',
                'args': [{'visible': [False, False, False, True]}, {'title': 'Turnover per Employee (€) by Size'}]
            }
        ],
        'direction': 'down',
        'showactive': True,
        'active': 0,
        'pad': {'r': 10, 't': 10},
        'x': 0,
        'xanchor': 'left',
        'y': 1.3,
        'yanchor': 'top'
    }],
    showlegend=False,
    width=800,
    height=500
)

# Save the plot
fig.write_html(r'C:\Users\guill\Downloads\FIG_EUROSTAT_Turnover_Employee.html')