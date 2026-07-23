"""Retail demand forecasting and optimization package.

Modules:
    utils                 - config loading, logging, small shared helpers
    data_loader           - typed CSV ingestion
    data_validation       - schema/quality checks and cleaning
    feature_engineering   - retail time-series feature builder
    model_baseline        - naive + statistical baselines (incl. SARIMAX)
    model_ml              - global gradient-boosting model + advanced interface
    evaluation            - retail forecast accuracy metrics and slicing
    model_selection       - compare models and pick the best by WAPE
    forecasting_pipeline  - orchestrates training + forward forecasting
    optimization_engine   - inventory/markdown recommendations from forecasts
    visualization         - plots
    explainability        - feature importance + plain-English explanations
"""

__version__ = "1.0.0"
