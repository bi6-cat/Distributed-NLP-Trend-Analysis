{% macro trend_score_calc(velocity, acceleration, engagement_norm, influencer_score) %}
    round(
          0.35 * {{ velocity }}
        + 0.25 * {{ acceleration }}
        + 0.25 * {{ engagement_norm }} * 100
        + 0.15 * {{ influencer_score }} * 1000
    , 2)
{% endmacro %}
