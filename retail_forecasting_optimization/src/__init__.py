"""Peacock subscriber forecasting and scenario package.

Modules:
    utils                 - config loading, logging, small shared helpers
    data_loader           - typed CSV ingestion (history + content calendar)
    data_validation       - schema/quality checks and cleaning
    feature_engineering   - subscriber time-series feature builder
    model_baseline        - naive + statistical baselines (incl. SARIMAX)
    model_ml              - global gradient-boosting model + advanced interface
    evaluation            - forecast accuracy metrics and slicing
    model_selection       - compare models and pick the best by WAPE, per target
    forecasting_pipeline  - forward forecasts per target and derived subscriber KPIs
    scenario_engine       - risk flags, price-change scenarios and Growth OKR rollup
    visualization         - plots
    explainability        - feature importance + plain-English explanations
"""

__version__ = "2.0.0"
