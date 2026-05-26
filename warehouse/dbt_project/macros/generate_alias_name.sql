{% macro generate_alias_name(custom_alias_name=none, node=none) -%}
    {%- if custom_alias_name is not none -%}
        {{ custom_alias_name | trim }}
    {%- else -%}
        {{ 'dbt_' ~ node.name }}
    {%- endif -%}
{%- endmacro %}
