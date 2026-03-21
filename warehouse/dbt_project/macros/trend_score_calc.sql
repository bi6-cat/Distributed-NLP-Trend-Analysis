{% macro trend_score_calc(velocity, acceleration, engagement_norm) %}
    round(
          0.40 * {{ velocity }}
        + 0.30 * {{ acceleration }}
        + 0.30 * {{ engagement_norm }} * 100
    , 2)
{% endmacro %}
