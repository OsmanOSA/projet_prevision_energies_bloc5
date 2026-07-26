import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import json

st.set_page_config(
    page_title="Order Priority",
    layout="wide",
)

# Données
df = pd.DataFrame({
    "priority": ["Medium", "High", "Critical", "Low"],
    "orders": [10179, 5273, 1264, 872],
})

colors = {
    "Medium": "#0878D1",
    "High": "#73BFF2",
    "Critical": "#FF2D2D",
    "Low": "#FFA1AA",
}

pie_data = [
    {
        "name": row["priority"],
        "value": int(row["orders"]),
        "itemStyle": {"color": colors.get(row["priority"], "#999999")}
    }
    for _, row in df.iterrows()
]

options = {
    "title": {
        "text": "Order Priority",
        "left": "5%",
        "top": "3%",
        "textStyle": {"fontSize": 22, "fontWeight": "bold", "color": "#252936"},
    },
    "tooltip": {
        "trigger": "item",
        "formatter": "{b}: {c} ({d}%)",
    },
    "series": [{
        "name": "Priorité",
        "type": "pie",
        "radius": ["37%", "63%"],
        "center": ["50%", "55%"],
        "itemStyle": {
            "borderColor": "#FFFFFF",
            "borderWidth": 5,
            "borderRadius": 11,
        },
        "data": pie_data,
    }]
}

# Alternative robuste pour Colab : Injection HTML directe d'ECharts
echarts_html = f"""
<div id="echarts_container" style="width:100%;height:520px;"></div>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
<script>
    var chartDom = document.getElementById('echarts_container');
    var myChart = echarts.init(chartDom);
    var option = {json.dumps(options)};
    myChart.setOption(option);
    window.addEventListener('resize', myChart.resize);
</script>
"""

components.html(echarts_html, height=550)