import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import scipy.optimize as optimize

# ---------------------------------------------------------
# Page Configuration & Custom Styling System
# ---------------------------------------------------------
st.set_page_config(
    page_title="Sales Predictor & Budget Optimizer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark Glassmorphism CSS Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Main Background Gradient */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #1E1B4B 0%, #0F172A 45%, #020617 90%);
        color: #F8FAFC;
    }
    
    /* Glassmorphism Title Card */
    .header-container {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.6rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818CF8 0%, #C084FC 50%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    
    .sub-title {
        font-size: 1.05rem;
        color: #94A3B8;
        font-weight: 400;
    }
    
    /* Glassmorphism Metric Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.65);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.3rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        transition: transform 0.25s ease, border-color 0.25s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.5);
    }
    
    .card-accent-blue {
        border-left: 4px solid #6366F1;
    }
    
    .card-accent-purple {
        border-left: 4px solid #A855F7;
    }
    
    .metric-value-huge {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
        margin-top: 0.3rem;
    }
    
    .metric-value-purple {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #C084FC 0%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
        margin-top: 0.3rem;
    }
    
    .metric-label-clean {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* Input Section Header */
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Custom Badges */
    .badge-chip {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(99, 102, 241, 0.15);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin-right: 0.4rem;
    }
    
    /* Styled Buttons */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%) !important;
        color: #F8FAFC !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
        color: #FFFFFF !important;
        border-color: transparent !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
    }

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.85) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(30, 41, 59, 0.4);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 10px;
        color: #94A3B8;
        font-weight: 600;
        font-size: 0.9rem;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4338CA 0%, #6366F1 100%) !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data & ML Pipeline (Degree 3 Polynomial Model)
# ---------------------------------------------------------
DATA_PATH = "advertising.csv"

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except Exception as e:
        st.error(f"Error loading dataset `{DATA_PATH}`: {e}")
        np.random.seed(42)
        tv = np.random.uniform(10, 300, 200)
        radio = np.random.uniform(5, 50, 200)
        newspaper = np.random.uniform(0, 100, 200)
        sales = 4.5 + 0.054 * tv + 0.1 * radio + 0.003 * newspaper + np.random.normal(0, 1.5, 200)
        return pd.DataFrame({'TV': tv, 'Radio': radio, 'Newspaper': newspaper, 'Sales': sales})

df = load_data()

@st.cache_resource
def train_poly3_model(data):
    X = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Polynomial Regression (Degree 3) Model
    model = make_pipeline(PolynomialFeatures(degree=3), Ridge(alpha=1.0))
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return {
        "model": model,
        "name": "Polynomial Regression (Degree 3)",
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "y_test": y_test,
        "y_pred": y_pred
    }, X, y

model_info, X_full, y_full = train_poly3_model(df)
current_model = model_info["model"]

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.markdown("### ⚙️ Model Architecture")
st.sidebar.markdown("""
<div style="background: rgba(99, 102, 241, 0.12); padding: 0.8rem 1rem; border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.3); margin-bottom: 1.2rem;">
    <div style="font-size: 0.75rem; color: #818CF8; font-weight: 700; text-transform: uppercase;">Active Engine</div>
    <div style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-top: 0.2rem;">Polynomial Regression (Degree 3)</div>
    <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 0.3rem;">Captures 3rd-order non-linear channel interactions & diminishing returns.</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Scenario Presets")

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    if st.button("📊 Balanced"):
        st.session_state["tv_val"] = 150.0
        st.session_state["radio_val"] = 30.0
        st.session_state["news_val"] = 25.0
with col_sb2:
    if st.button("📺 TV Max"):
        st.session_state["tv_val"] = 260.0
        st.session_state["radio_val"] = 10.0
        st.session_state["news_val"] = 5.0

col_sb3, col_sb4 = st.sidebar.columns(2)
with col_sb3:
    if st.button("📻 Digital"):
        st.session_state["tv_val"] = 60.0
        st.session_state["radio_val"] = 45.0
        st.session_state["news_val"] = 15.0
with col_sb4:
    if st.button("🌱 Minimal"):
        st.session_state["tv_val"] = 35.0
        st.session_state["radio_val"] = 15.0
        st.session_state["news_val"] = 10.0

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Metrics")
st.sidebar.markdown(f"""
<div style="background: rgba(30, 41, 59, 0.6); padding: 0.8rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
    <div style="font-size: 0.8rem; color: #94A3B8;">ACCURACY (R²)</div>
    <div style="font-size: 1.4rem; font-weight: 700; color: #10B981;">{model_info['r2']:.4f}</div>
    <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.4rem;">MAE: <b style="color:#F8FAFC;">{model_info['mae']:.3f}</b> | RMSE: <b style="color:#F8FAFC;">{model_info['rmse']:.3f}</b></div>
</div>
""", unsafe_allow_html=True)

# Initialize Session State values
if "tv_val" not in st.session_state:
    st.session_state["tv_val"] = float(round(df['TV'].mean(), 1))
if "radio_val" not in st.session_state:
    st.session_state["radio_val"] = float(round(df['Radio'].mean(), 1))
if "news_val" not in st.session_state:
    st.session_state["news_val"] = float(round(df['Newspaper'].mean(), 1))

# ---------------------------------------------------------
# Main UI Layout Header
# ---------------------------------------------------------
st.markdown("""
<div class="header-container">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
            <div class="main-title">✨ Polynomial Sales Predictor (Degree 3)</div>
            <div class="sub-title">Simulate marketing spend across TV, Radio & Newspaper channels with 3rd-degree polynomial feature interactions.</div>
        </div>
        <div style="text-align: right;">
            <span class="badge-chip">Degree 3 Polynomial</span>
            <span class="badge-chip">Ridge Regularized</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Grid: Inputs vs KPIs
col_input, col_kpi = st.columns([1.15, 1], gap="large")

with col_input:
    st.markdown('<div class="section-header">🎛️ Advertising Budget Inputs ($k)</div>', unsafe_allow_html=True)
    
    tv_input = st.slider(
        "📺 TV Advertising Spend ($k)",
        min_value=0.0,
        max_value=350.0,
        value=float(st.session_state["tv_val"]),
        step=1.0,
        key="tv_slider"
    )
    
    radio_input = st.slider(
        "📻 Radio Advertising Spend ($k)",
        min_value=0.0,
        max_value=100.0,
        value=float(st.session_state["radio_val"]),
        step=1.0,
        key="radio_slider"
    )
    
    news_input = st.slider(
        "📰 Newspaper Advertising Spend ($k)",
        min_value=0.0,
        max_value=120.0,
        value=float(st.session_state["news_val"]),
        step=1.0,
        key="news_slider"
    )
    
    st.session_state["tv_val"] = tv_input
    st.session_state["radio_val"] = radio_input
    st.session_state["news_val"] = news_input

# Make Prediction
input_df = pd.DataFrame({
    'TV': [tv_input],
    'Radio': [radio_input],
    'Newspaper': [news_input]
})

predicted_sales = float(current_model.predict(input_df)[0])
total_budget = tv_input + radio_input + news_input
sales_per_100k = (predicted_sales / total_budget * 100) if total_budget > 0 else 0.0

with col_kpi:
    st.markdown('<div class="section-header">⚡ Live Sales Forecast</div>', unsafe_allow_html=True)
    
    kpi_col1, kpi_col2 = st.columns(2)
    with kpi_col1:
        st.markdown(f"""
        <div class="glass-card card-accent-blue">
            <div class="metric-label-clean">Predicted Sales</div>
            <div class="metric-value-huge">{predicted_sales:.2f}</div>
            <div style="font-size: 0.85rem; color: #64748B; margin-top: 0.2rem;">thousand units</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_col2:
        st.markdown(f"""
        <div class="glass-card card-accent-purple">
            <div class="metric-label-clean">Total Budget</div>
            <div class="metric-value-purple">${total_budget:.1f}k</div>
            <div style="font-size: 0.85rem; color: #64748B; margin-top: 0.2rem;">combined spend</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    kpi_col3, kpi_col4 = st.columns(2)
    with kpi_col3:
        st.metric(
            label="Yield per $100k Spend",
            value=f"{sales_per_100k:.2f} units",
            delta="Degree 3 Polynomial"
        )
    with kpi_col4:
        st.metric(
            label="Model Error Bound (MAE)",
            value=f"± {model_info['mae']:.2f} k"
        )

# ---------------------------------------------------------
# Tabs for Analytics & Visualizations
# ---------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Budget Split & Sensitivity",
    "🌐 3D Sales Response Surface",
    "🚀 Automated Budget Optimizer",
    "📁 Dataset & Model Details"
])

# Dark Plotly Template Setup
plotly_dark_layout = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(15, 23, 42, 0.4)',
    font=dict(color='#F8FAFC', family='Plus Jakarta Sans'),
    xaxis=dict(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.1)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.1)')
)

with tab1:
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("##### 🍩 Marketing Spend Allocation")
        if total_budget > 0:
            pie_df = pd.DataFrame({
                "Channel": ["TV", "Radio", "Newspaper"],
                "Budget": [tv_input, radio_input, news_input]
            })
            fig_pie = px.pie(
                pie_df, 
                values="Budget", 
                names="Channel", 
                color="Channel",
                color_discrete_map={"TV": "#6366F1", "Radio": "#10B981", "Newspaper": "#F59E0B"},
                hole=0.55
            )
            fig_pie.update_layout(**plotly_dark_layout, margin=dict(t=20, b=20, l=20, r=20), height=320)
            st.plotly_chart(fig_pie, width='stretch')
        else:
            st.info("Adjust budget sliders above to view allocation donut.")
            
    with col_chart2:
        st.markdown("##### 📈 Non-Linear Sensitivity Curves (Degree 3)")
        st.caption("Sales response when sweeping budget from $0k to $300k while keeping other variables fixed.")
        
        sweep_range = np.linspace(0, 300, 100)
        
        tv_sweep_df = pd.DataFrame({'TV': sweep_range, 'Radio': radio_input, 'Newspaper': news_input})
        tv_pred = current_model.predict(tv_sweep_df)
        
        radio_sweep_df = pd.DataFrame({'TV': tv_input, 'Radio': sweep_range, 'Newspaper': news_input})
        radio_pred = current_model.predict(radio_sweep_df)
        
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(x=sweep_range, y=tv_pred, mode='lines', name='TV Impact', line=dict(color='#6366F1', width=3)))
        fig_sens.add_trace(go.Scatter(x=sweep_range, y=radio_pred, mode='lines', name='Radio Impact', line=dict(color='#10B981', width=3)))
        
        fig_sens.update_layout(
            **plotly_dark_layout,
            xaxis_title="Channel Budget ($k)",
            yaxis_title="Predicted Sales (k units)",
            margin=dict(t=20, b=20, l=20, r=20),
            height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_sens, width='stretch')

with tab2:
    st.markdown("##### 🌐 3D Degree-3 Sales Response Surface")
    st.caption("Simulated polynomial sales response grid combining TV & Radio investments.")
    
    tv_grid = np.linspace(0, 350, 35)
    radio_grid = np.linspace(0, 100, 35)
    TV_mesh, RADIO_mesh = np.meshgrid(tv_grid, radio_grid)
    
    mesh_input = pd.DataFrame({
        'TV': TV_mesh.ravel(),
        'Radio': RADIO_mesh.ravel(),
        'Newspaper': news_input
    })
    
    SALES_mesh = current_model.predict(mesh_input).reshape(TV_mesh.shape)
    
    fig_3d = go.Figure(data=[
        go.Surface(
            z=SALES_mesh, 
            x=TV_mesh, 
            y=RADIO_mesh,
            colorscale='Plasma',
            opacity=0.9,
            name="Degree 3 Model Surface"
        ),
        go.Scatter3d(
            x=[tv_input],
            y=[radio_input],
            z=[predicted_sales],
            mode='markers',
            marker=dict(size=10, color='#38BDF8', symbol='diamond', line=dict(color='#FFFFFF', width=2)),
            name="Current Budget"
        )
    ])
    
    fig_3d.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            xaxis=dict(title='TV ($k)', gridcolor='rgba(255,255,255,0.1)', backgroundcolor='rgba(15, 23, 42, 0.5)'),
            yaxis=dict(title='Radio ($k)', gridcolor='rgba(255,255,255,0.1)', backgroundcolor='rgba(15, 23, 42, 0.5)'),
            zaxis=dict(title='Sales (k units)', gridcolor='rgba(255,255,255,0.1)', backgroundcolor='rgba(15, 23, 42, 0.5)')
        ),
        font=dict(color='#F8FAFC'),
        margin=dict(l=0, r=0, b=0, t=30),
        height=520
    )
    st.plotly_chart(fig_3d, width='stretch')

with tab3:
    st.markdown("##### 🚀 AI Sales Maximization Optimizer")
    st.caption("Automatically compute the ideal budget split across TV, Radio, and Newspaper to yield maximum sales using the Degree 3 Polynomial model.")
    
    target_max_budget = st.number_input("Target Budget Limit ($k)", min_value=10.0, max_value=600.0, value=200.0, step=10.0)
    
    if st.button("✨ Run Optimization Solver", type="primary"):
        def objective(x):
            tv_b, rad_b, news_b = x
            df_inst = pd.DataFrame({'TV': [tv_b], 'Radio': [rad_b], 'Newspaper': [news_b]})
            return -float(current_model.predict(df_inst)[0])
        
        cons = ({'type': 'ineq', 'fun': lambda x: target_max_budget - (x[0] + x[1] + x[2])})
        bnds = ((0, 350), (0, 100), (0, 120))
        init_guess = [target_max_budget/3, target_max_budget/3, target_max_budget/3]
        
        res = optimize.minimize(objective, init_guess, method='SLSQP', bounds=bnds, constraints=cons)
        
        if res.success:
            opt_tv, opt_rad, opt_news = res.x
            opt_sales = -res.fun
            
            st.success("🎉 Optimal Allocation Computed!")
            
            opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
            opt_col1.metric("Optimal TV Spend", f"${opt_tv:.1f}k")
            opt_col2.metric("Optimal Radio Spend", f"${opt_rad:.1f}k")
            opt_col3.metric("Optimal Newspaper Spend", f"${opt_news:.1f}k")
            opt_col4.metric("Max Projected Sales", f"{opt_sales:.2f}k units")
            
            if st.button("👉 Apply Optimal Budget to Controls"):
                st.session_state["tv_val"] = float(round(opt_tv, 1))
                st.session_state["radio_val"] = float(round(opt_rad, 1))
                st.session_state["news_val"] = float(round(opt_news, 1))
                st.rerun()
        else:
            st.error("Optimization failed to converge. Try adjusting the budget limit.")

with tab4:
    st.markdown("##### 📁 Data Exploration & Model Specs")
    
    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        st.markdown("**Historical Advertising Dataset**")
        st.dataframe(df, width='stretch', height=280)
        
    with col_d2:
        st.markdown("**Polynomial Regression (Degree 3) Specifications**")
        st.markdown(f"""
        - **Algorithm**: Polynomial Features (Degree 3) + Ridge Regularization
        - **$R^2$ Accuracy**: `{model_info['r2']:.4f}`
        - **Mean Absolute Error (MAE)**: `{model_info['mae']:.4f}`
        - **Root Mean Squared Error (RMSE)**: `{model_info['rmse']:.4f}`
        """)
        
        st.markdown("**Feature Correlation Heatmap**")
        corr = df.corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="Purples")
        fig_corr.update_layout(**plotly_dark_layout, height=220, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_corr, width='stretch')

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748B; font-size: 0.85rem;'>Polynomial Sales Prediction (Degree 3) • Glassmorphic Dark UI</p>", unsafe_allow_html=True)
