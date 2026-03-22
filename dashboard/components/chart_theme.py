"""
Design system constants and Plotly theme for the Tech Trend Radar dashboard.
"""
import plotly.graph_objects as go
import plotly.io as pio

# ── Colour palette ──────────────────────────────────────────────
PRIMARY      = "#6C5CE7"
POSITIVE     = "#00D2A0"
NEGATIVE     = "#FF6B6B"
NEUTRAL      = "#A0AEC0"
CRISIS_HIGH  = "#E53E3E"
CRISIS_MED   = "#ED8936"
CRISIS_LOW   = "#48BB78"
BG_CARD      = "#1A1A2E"
BG_PAGE      = "#0F0F23"
TEXT_PRIMARY  = "#E2E8F0"
TEXT_MUTED    = "#718096"
ACCENT_BLUE  = "#4299E1"
ACCENT_PURPLE = "#9F7AEA"

# Convenience lists
SENTIMENT_COLORS = [POSITIVE, NEGATIVE, NEUTRAL]
SENTIMENT_MAP    = {"positive": POSITIVE, "negative": NEGATIVE, "neutral": NEUTRAL}
SEVERITY_COLORS  = {"HIGH": CRISIS_HIGH, "MEDIUM": CRISIS_MED, "LOW": CRISIS_LOW}

SOURCE_COLORS = {
    "voz":       "#6C5CE7",
    "tinhte":    "#00CEC9",
    "vnexpress": "#FDCB6E",
    "youtube":   "#E17055",
}

# ── Plotly template ─────────────────────────────────────────────
_layout = go.Layout(
    font=dict(family="Inter, sans-serif", size=13, color=TEXT_PRIMARY),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor="#2D3748", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#2D3748", showgrid=True, zeroline=False),
    margin=dict(l=48, r=16, t=44, b=40),
    hoverlabel=dict(
        bgcolor=BG_CARD,
        font_color=TEXT_PRIMARY,
        bordercolor="#4A5568",
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_PRIMARY, size=12),
    ),
    colorway=[PRIMARY, POSITIVE, NEGATIVE, ACCENT_BLUE, CRISIS_MED,
              ACCENT_PURPLE, "#00CEC9", "#FDCB6E"],
)

TEMPLATE = go.layout.Template(layout=_layout)
pio.templates["radar_dark"] = TEMPLATE
pio.templates.default = "radar_dark"


def apply_chart_style(fig: go.Figure, height: int = 400) -> go.Figure:
    """Apply consistent styling to any Plotly figure."""
    fig.update_layout(
        height=height,
        template="radar_dark",
    )
    return fig
