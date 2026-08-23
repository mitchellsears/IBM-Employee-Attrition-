"""
IBM HR Attrition — K-Means Segmentation & Scenario Simulator (Dash / Plotly Cloud version)
=============================================================================================
Built for Dr. Sarah Chen (CHRO, IBM) and the IBM HR Analytics division.

Functionally identical analytical engine to the Gradio version (app.py):
  1. Elbow Method + Silhouette Score, k = 2..10
  2. Segment Explorer: interactive k, KPI profile table, comparative charts
  3. Scenario Simulator: assign a hypothetical employee to its nearest K-Means segment

Deployment target: Plotly Cloud (Dash app). Main file for Plotly Cloud config: app_dash.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from dash import Dash, dcc, html, Input, Output, State, dash_table

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ----------------------------------------------------------------------
# 1. DATA LOADING & PREPROCESSING  (identical logic to the Gradio version)
# ----------------------------------------------------------------------
DATA_PATH = "IBM_Dataset.csv"
df_raw = pd.read_csv(DATA_PATH)

EDUCATION_LABELS = {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"}
JOBLEVEL_LABELS = {1: "Entry", 2: "Junior", 3: "Mid", 4: "Senior", 5: "Executive"}
SATISFACTION_LABELS = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
WLB_LABELS = {1: "Bad", 2: "Good", 3: "Better", 4: "Best"}
DEPARTMENTS = sorted(df_raw["Department"].unique().tolist())
GENDERS = sorted(df_raw["Gender"].unique().tolist())

NUMERIC_FEATURES = [
    "Age", "Education", "JobLevel", "JobSatisfaction", "MonthlyIncome",
    "TotalWorkingYears", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsWithCurrManager",
]

def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encoding strategy (see PDF report / vibe-coding prompt for full rationale):
      - Ordinal survey/level variables kept as-is (Education, JobLevel, JobSatisfaction, WorkLifeBalance).
      - Gender -> single 0/1 dummy (Male = 1).
      - Department (3 nominal categories) -> one-hot dummies.
      - Continuous variables used as-is prior to scaling.
      - Attrition excluded from clustering features (used only for post-hoc profiling).
      - Full feature matrix standardized (StandardScaler) before K-Means.
    """
    d = df.copy()
    d["Gender_bin"] = (d["Gender"] == "Male").astype(int)
    dept_dummies = pd.get_dummies(d["Department"], prefix="Dept")
    feat = pd.concat([d[NUMERIC_FEATURES], d[["Gender_bin"]], dept_dummies], axis=1)
    return feat

FEATURES_DF = build_feature_frame(df_raw)
FEATURE_COLUMNS = FEATURES_DF.columns.tolist()

scaler = StandardScaler()
X_SCALED = scaler.fit_transform(FEATURES_DF.values)

df_raw["AttritionBin"] = (df_raw["Attrition"] == "Yes").astype(int)
OVERALL_ATTRITION = df_raw["AttritionBin"].mean() * 100

# ----------------------------------------------------------------------
# 2. ELBOW METHOD + SILHOUETTE SCORE (k = 2..10), cached at startup
# ----------------------------------------------------------------------
K_RANGE = list(range(2, 11))
INERTIAS, SILHOUETTES, KMEANS_MODELS = [], [], {}

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_SCALED)
    INERTIAS.append(km.inertia_)
    SILHOUETTES.append(silhouette_score(X_SCALED, labels))
    KMEANS_MODELS[k] = km

BEST_SIL_K = K_RANGE[int(np.argmax(SILHOUETTES))]
DEFAULT_K = 4

def make_elbow_silhouette_fig():
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Elbow Method (Inertia / WCSS)", f"Silhouette Score (best k = {BEST_SIL_K})"))
    fig.add_trace(go.Scatter(x=K_RANGE, y=INERTIAS, mode="lines+markers", name="Inertia",
                              line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=K_RANGE, y=SILHOUETTES, mode="lines+markers", name="Silhouette",
                              line=dict(color="#d62728")), row=1, col=2)
    fig.add_vline(x=BEST_SIL_K, line_dash="dash", line_color="gray", row=1, col=2)
    fig.update_xaxes(title_text="Number of clusters (k)", dtick=1)
    fig.update_yaxes(title_text="Within-cluster sum of squares", row=1, col=1)
    fig.update_yaxes(title_text="Average silhouette score", row=1, col=2)
    fig.update_layout(showlegend=False, height=380, margin=dict(t=50, b=40, l=50, r=20))
    return fig

ELBOW_SIL_FIG = make_elbow_silhouette_fig()
ELBOW_SIL_TABLE = pd.DataFrame({
    "k": K_RANGE,
    "Inertia (WCSS)": [round(v, 1) for v in INERTIAS],
    "Silhouette Score": [round(v, 4) for v in SILHOUETTES],
})

PCA_2D = PCA(n_components=2, random_state=42)
PCA_COORDS = PCA_2D.fit_transform(X_SCALED)

CLUSTER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# ----------------------------------------------------------------------
# 3. FIT / PROFILE HELPERS
# ----------------------------------------------------------------------
def fit_solution(k: int):
    km = KMEANS_MODELS[k]
    labels = km.labels_
    d = df_raw.copy()
    d["Cluster"] = labels
    profile = d.groupby("Cluster").agg(
        Employees=("Attrition", "size"),
        AttritionRate=("AttritionBin", "mean"),
        AvgAge=("Age", "mean"),
        AvgJobLevel=("JobLevel", "mean"),
        AvgIncome=("MonthlyIncome", "mean"),
        AvgTotalWorkingYears=("TotalWorkingYears", "mean"),
        AvgYearsAtCompany=("YearsAtCompany", "mean"),
        AvgYearsInRole=("YearsInCurrentRole", "mean"),
        AvgYearsWithManager=("YearsWithCurrManager", "mean"),
        AvgJobSatisfaction=("JobSatisfaction", "mean"),
        AvgWorkLifeBalance=("WorkLifeBalance", "mean"),
        AvgEducation=("Education", "mean"),
        PctMale=("Gender", lambda s: (s == "Male").mean()),
    ).round(2)
    profile["AttritionRate"] = (profile["AttritionRate"] * 100).round(1)
    profile["PctMale"] = (profile["PctMale"] * 100).round(1)
    dept_mix = (pd.crosstab(d["Cluster"], d["Department"], normalize="index").round(3) * 100)
    return d, profile, dept_mix, km

def risk_tier(rate, all_rates):
    q1, q2 = np.percentile(all_rates, [33, 66])
    if rate >= q2:
        return "High Risk"
    elif rate >= q1:
        return "Moderate Risk"
    return "Low Risk / Stable"

# ----------------------------------------------------------------------
# 4. CHART BUILDERS (Plotly)
# ----------------------------------------------------------------------
def fig_attrition_by_cluster(profile_df):
    colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(profile_df))]
    fig = go.Figure(go.Bar(x=[str(c) for c in profile_df.index], y=profile_df["AttritionRate"],
                            marker_color=colors, text=profile_df["AttritionRate"].astype(str) + "%",
                            textposition="outside"))
    fig.add_hline(y=OVERALL_ATTRITION, line_dash="dash", line_color="black",
                  annotation_text=f"Company avg = {OVERALL_ATTRITION:.1f}%")
    fig.update_layout(title="Attrition Rate by Segment", xaxis_title="Cluster",
                       yaxis_title="Attrition Rate (%)", height=380, margin=dict(t=50, b=40))
    return fig

def fig_cluster_size(profile_df):
    colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(profile_df))]
    fig = go.Figure(go.Pie(labels=[f"Cluster {c} (n={n})" for c, n in
                                    zip(profile_df.index, profile_df["Employees"])],
                            values=profile_df["Employees"], marker_colors=colors))
    fig.update_layout(title="Workforce Share by Segment", height=380, margin=dict(t=50, b=20))
    return fig

def fig_pca_scatter(labeled_df, k):
    fig = go.Figure()
    for c in range(k):
        mask = labeled_df["Cluster"] == c
        fig.add_trace(go.Scattergl(x=PCA_COORDS[mask, 0], y=PCA_COORDS[mask, 1], mode="markers",
                                    name=f"Cluster {c}", marker=dict(size=5, color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)]),
                                    opacity=0.7))
    fig.update_layout(title="Segments in 2D (PCA projection)",
                       xaxis_title=f"PC1 ({PCA_2D.explained_variance_ratio_[0]*100:.1f}% var)",
                       yaxis_title=f"PC2 ({PCA_2D.explained_variance_ratio_[1]*100:.1f}% var)",
                       height=420, margin=dict(t=50, b=40))
    return fig

def fig_satisfaction_wlb(profile_df):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"Cluster {c}" for c in profile_df.index], y=profile_df["AvgJobSatisfaction"],
                          name="Job Satisfaction (1-4)", marker_color="#4c72b0"))
    fig.add_trace(go.Bar(x=[f"Cluster {c}" for c in profile_df.index], y=profile_df["AvgWorkLifeBalance"],
                          name="Work-Life Balance (1-4)", marker_color="#dd8452"))
    fig.update_layout(barmode="group", title="Satisfaction & Work-Life Balance by Segment",
                       yaxis=dict(range=[0, 4.5]), height=380, margin=dict(t=50, b=40))
    return fig

def fig_seniority(profile_df):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"Cluster {c}" for c in profile_df.index], y=profile_df["AvgTotalWorkingYears"],
                          name="Total Working Years", marker_color="#55a868"))
    fig.add_trace(go.Bar(x=[f"Cluster {c}" for c in profile_df.index], y=profile_df["AvgYearsAtCompany"],
                          name="Years at IBM", marker_color="#c44e52"))
    fig.add_trace(go.Bar(x=[f"Cluster {c}" for c in profile_df.index], y=profile_df["AvgYearsInRole"],
                          name="Years in Current Role", marker_color="#8172b2"))
    fig.update_layout(barmode="group", title="Seniority / Tenure Profile by Segment",
                       height=380, margin=dict(t=50, b=40))
    return fig

# ----------------------------------------------------------------------
# 5. DASH APP LAYOUT
# ----------------------------------------------------------------------
app = Dash(__name__, title="IBM HR Attrition — Segmentation & Simulator")
server = app.server  # exposed for WSGI-based hosts; harmless extra line for Plotly Cloud too

def kpi_table(df):
    return dash_table.DataTable(
        data=df.reset_index().to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.reset_index().columns],
        style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "Helvetica", "fontSize": 12, "padding": "6px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#1a2b4c", "color": "white"},
        style_data={"backgroundColor": "#f8f9fb"},
    )

d0, profile0, dept0, km0 = fit_solution(DEFAULT_K)

app.layout = html.Div(style={"fontFamily": "Helvetica, Arial, sans-serif", "maxWidth": "1100px",
                              "margin": "0 auto", "padding": "20px"}, children=[
    html.H2("IBM Workforce Attrition — K-Means Segmentation & Scenario Simulator"),
    html.P("Prepared for Dr. Sarah Chen, CHRO, and the IBM HR Analytics division. This tool uncovers natural "
           "employee segments from workforce data (unsupervised learning — K-Means clustering) and lets HR "
           "leadership simulate a hypothetical employee to see which segment — and which historical attrition "
           "rate — it maps to.", style={"color": "#444"}),

    dcc.Tabs(id="tabs", value="tab-1", children=[

        dcc.Tab(label="1. Clustering Quality (Elbow & Silhouette)", value="tab-1", children=[
            html.Div(style={"paddingTop": "16px"}, children=[
                html.P("Evaluate candidate numbers of clusters k = 2 to 10 using the Elbow Method (within-cluster "
                       "sum of squares) and the average Silhouette Score, computed on all standardized features."),
                dcc.Graph(figure=ELBOW_SIL_FIG),
                html.H4("Elbow / Silhouette values by k"),
                kpi_table(ELBOW_SIL_TABLE.set_index("k")),
                html.P([
                    html.B(f"Recommended solution: k = {DEFAULT_K}. "),
                    f"It sits at the elbow of the inertia curve and yields a strong, business-interpretable "
                    f"silhouette score (≈{SILHOUETTES[K_RANGE.index(DEFAULT_K)]:.3f}) while producing segments "
                    f"large enough, and distinct enough, to act on. k = {BEST_SIL_K} scores marginally higher on "
                    f"silhouette but collapses the workforce into a coarser split. See the PDF report for the "
                    f"full justification."
                ], style={"marginTop": "10px"}),
            ])
        ]),

        dcc.Tab(label="2. Segment Explorer", value="tab-2", children=[
            html.Div(style={"paddingTop": "16px"}, children=[
                html.Label("Number of clusters (k)"),
                dcc.Slider(id="k-slider", min=2, max=10, step=1, value=DEFAULT_K,
                           marks={i: str(i) for i in range(2, 11)}),
                html.Div(id="status-text", style={"marginTop": "10px", "fontWeight": "bold"}),
                html.Div(style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}, children=[
                    html.Div(dcc.Graph(id="attrition-plot"), style={"flex": "1", "minWidth": "420px"}),
                    html.Div(dcc.Graph(id="size-plot"), style={"flex": "1", "minWidth": "420px"}),
                ]),
                html.Div(style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}, children=[
                    html.Div(dcc.Graph(id="sat-plot"), style={"flex": "1", "minWidth": "420px"}),
                    html.Div(dcc.Graph(id="seniority-plot"), style={"flex": "1", "minWidth": "420px"}),
                ]),
                dcc.Graph(id="pca-plot"),
                html.H4("Cluster Profile Summary (KPIs)"),
                html.Div(id="profile-table"),
                html.H4("Department Composition by Segment (%)"),
                html.Div(id="dept-table"),
            ])
        ]),

        dcc.Tab(label="3. Employee Scenario Simulator", value="tab-3", children=[
            html.Div(style={"paddingTop": "16px"}, children=[
                html.P("Enter a hypothetical (or real, anonymized) employee's profile. The tool assigns it to the "
                       "nearest K-Means segment (using the currently fitted solution from Tab 2) and reports that "
                       "segment's historical attrition rate and characteristics — it does not predict an "
                       "individual outcome."),
                html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px",
                                 "maxWidth": "800px"}, children=[
                    html.Div([html.Label("Age"), dcc.Slider(18, 60, 1, value=29, id="age-in",
                                                              tooltip={"placement": "bottom"})]),
                    html.Div([html.Label("Monthly Income (USD)"),
                              dcc.Slider(1000, 20000, 100, value=4500, id="income-in",
                                         tooltip={"placement": "bottom"})]),
                    html.Div([html.Label("Department"),
                              dcc.Dropdown(DEPARTMENTS, value=DEPARTMENTS[0], id="dept-in")]),
                    html.Div([html.Label("Total Working Years"),
                              dcc.Slider(0, 40, 1, value=6, id="totalyrs-in", tooltip={"placement": "bottom"})]),
                    html.Div([html.Label("Education"),
                              dcc.Dropdown([{"label": f"{v} ({k})", "value": k} for k, v in EDUCATION_LABELS.items()],
                                           value=3, id="edu-in")]),
                    html.Div([html.Label("Work-Life Balance"),
                              dcc.Dropdown([{"label": f"{v} ({k})", "value": k} for k, v in WLB_LABELS.items()],
                                           value=2, id="wlb-in")]),
                    html.Div([html.Label("Gender"), dcc.Dropdown(GENDERS, value=GENDERS[0], id="gender-in")]),
                    html.Div([html.Label("Years at IBM"),
                              dcc.Slider(0, 40, 1, value=2, id="yearsatco-in", tooltip={"placement": "bottom"})]),
                    html.Div([html.Label("Job Level"),
                              dcc.Dropdown([{"label": f"{v} ({k})", "value": k} for k, v in JOBLEVEL_LABELS.items()],
                                           value=2, id="joblevel-in")]),
                    html.Div([html.Label("Years in Current Role"),
                              dcc.Slider(0, 18, 1, value=1, id="yearsinrole-in", tooltip={"placement": "bottom"})]),
                    html.Div([html.Label("Job Satisfaction"),
                              dcc.Dropdown([{"label": f"{v} ({k})", "value": k} for k, v in SATISFACTION_LABELS.items()],
                                           value=2, id="jobsat-in")]),
                    html.Div([html.Label("Years with Current Manager"),
                              dcc.Slider(0, 17, 1, value=1, id="yearswithmgr-in", tooltip={"placement": "bottom"})]),
                ]),
                html.Button("Assign to Segment", id="sim-btn", n_clicks=0,
                            style={"marginTop": "20px", "padding": "10px 20px", "fontSize": "15px",
                                   "backgroundColor": "#1a2b4c", "color": "white", "border": "none",
                                   "borderRadius": "6px", "cursor": "pointer"}),
                dcc.Markdown(id="sim-output", style={"marginTop": "20px"}),
            ])
        ]),
    ]),

    html.Hr(),
    html.P("Analytical engine: K-Means clustering on standardized workforce features (ordinal survey/level "
           "variables kept as-is, Gender as a 0/1 dummy, Department one-hot encoded, continuous variables "
           "standardized). Attrition is used only to characterize segments after clustering, never as a "
           "clustering input.", style={"fontSize": "12px", "color": "#777", "fontStyle": "italic"}),
])

# ----------------------------------------------------------------------
# 6. CALLBACKS
# ----------------------------------------------------------------------
@app.callback(
    Output("status-text", "children"),
    Output("attrition-plot", "figure"),
    Output("size-plot", "figure"),
    Output("sat-plot", "figure"),
    Output("seniority-plot", "figure"),
    Output("pca-plot", "figure"),
    Output("profile-table", "children"),
    Output("dept-table", "children"),
    Input("k-slider", "value"),
)
def update_segmentation(k):
    k = int(k)
    labeled_df, profile_df, dept_mix_df, _ = fit_solution(k)
    status = (f"Solution fitted with k = {k} clusters | Silhouette Score = "
              f"{SILHOUETTES[K_RANGE.index(k)]:.4f} | Inertia = {INERTIAS[K_RANGE.index(k)]:.0f}")
    return (
        status,
        fig_attrition_by_cluster(profile_df),
        fig_cluster_size(profile_df),
        fig_satisfaction_wlb(profile_df),
        fig_seniority(profile_df),
        fig_pca_scatter(labeled_df, k),
        kpi_table(profile_df),
        kpi_table(dept_mix_df),
    )

@app.callback(
    Output("sim-output", "children"),
    Input("sim-btn", "n_clicks"),
    State("k-slider", "value"),
    State("age-in", "value"), State("dept-in", "value"), State("edu-in", "value"),
    State("gender-in", "value"), State("joblevel-in", "value"), State("jobsat-in", "value"),
    State("income-in", "value"), State("totalyrs-in", "value"), State("wlb-in", "value"),
    State("yearsatco-in", "value"), State("yearsinrole-in", "value"), State("yearswithmgr-in", "value"),
    prevent_initial_call=True,
)
def simulate_employee(n_clicks, k, age, department, education, gender, joblevel, jobsat,
                       income, totalworkingyears, wlb, yearsatco, yearsinrole, yearswithmgr):
    k = int(k)
    _, profile_df, _, km = fit_solution(k)

    row = {c: 0 for c in FEATURE_COLUMNS}
    row["Age"] = age
    row["Education"] = education
    row["JobLevel"] = joblevel
    row["JobSatisfaction"] = jobsat
    row["MonthlyIncome"] = income
    row["TotalWorkingYears"] = totalworkingyears
    row["WorkLifeBalance"] = wlb
    row["YearsAtCompany"] = yearsatco
    row["YearsInCurrentRole"] = yearsinrole
    row["YearsWithCurrManager"] = yearswithmgr
    row["Gender_bin"] = 1 if gender == "Male" else 0
    dept_col = f"Dept_{department}"
    if dept_col in row:
        row[dept_col] = 1

    x = np.array([[row[c] for c in FEATURE_COLUMNS]], dtype=float)
    x_scaled = scaler.transform(x)
    assigned = int(km.predict(x_scaled)[0])

    prof_row = profile_df.loc[assigned]
    tier = risk_tier(prof_row["AttritionRate"], profile_df["AttritionRate"].values)

    return f"""
### Assigned Segment: **Cluster {assigned}**  ({tier})

| Segment KPI | Value |
|---|---|
| Employees in this segment | {int(prof_row['Employees'])} ({int(prof_row['Employees'])/len(df_raw)*100:.1f}% of workforce) |
| **Historical attrition rate in this segment** | **{prof_row['AttritionRate']:.1f}%** (company avg = {OVERALL_ATTRITION:.1f}%) |
| Avg. Job Level | {prof_row['AvgJobLevel']:.1f} |
| Avg. Monthly Income | ${prof_row['AvgIncome']:,.0f} |
| Avg. Job Satisfaction (1-4) | {prof_row['AvgJobSatisfaction']:.2f} |
| Avg. Work-Life Balance (1-4) | {prof_row['AvgWorkLifeBalance']:.2f} |
| Avg. Years at IBM | {prof_row['AvgYearsAtCompany']:.1f} |
| Avg. Total Working Years | {prof_row['AvgTotalWorkingYears']:.1f} |

**Note:** this is a *segment lookup*, not an individual prediction. Use the segment's historical attrition rate
and characteristics to gauge relative retention risk and to decide whether a targeted HR intervention is
warranted.
"""

if __name__ == "__main__":
    app.run(debug=True)
