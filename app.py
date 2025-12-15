import io
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def generate_sample_data(rows: int = 200) -> pd.DataFrame:
    """Create a reproducible synthetic dataset for quick experimentation."""
    rng = np.random.default_rng(seed=42)
    return pd.DataFrame(
        {
            "idade": rng.integers(18, 70, size=rows),
            "gastos": rng.normal(500, 120, size=rows).round(2),
            "satisfacao": rng.integers(1, 6, size=rows),
            "regiao": rng.choice(["Norte", "Sul", "Leste", "Oeste"], size=rows),
        }
    )


def load_data(uploaded_file: io.BytesIO) -> pd.DataFrame | None:
    """Load a CSV into a DataFrame, returning None on error."""
    try:
        return pd.read_csv(uploaded_file)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Erro ao ler o arquivo CSV: {exc}")
        return None


def handle_missing_values(df: pd.DataFrame, strategy: str) -> Tuple[pd.DataFrame, List[str]]:
    """Fill missing values according to the chosen strategy."""
    if strategy == "Nenhum":
        return df, []

    notes: List[str] = []
    cleaned = df.copy()

    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    categorical_cols = cleaned.select_dtypes(exclude=[np.number]).columns

    if strategy == "Média":
        cleaned[numeric_cols] = cleaned[numeric_cols].fillna(cleaned[numeric_cols].mean())
        notes.append("Valores numéricos preenchidos com a média.")
    elif strategy == "Mediana":
        cleaned[numeric_cols] = cleaned[numeric_cols].fillna(cleaned[numeric_cols].median())
        notes.append("Valores numéricos preenchidos com a mediana.")
    elif strategy == "Moda":
        cleaned[numeric_cols] = cleaned[numeric_cols].fillna(cleaned[numeric_cols].mode().iloc[0])
        notes.append("Valores numéricos preenchidos com a moda.")

    for col in categorical_cols:
        mode_value = cleaned[col].mode()
        if not mode_value.empty:
            cleaned[col] = cleaned[col].fillna(mode_value.iloc[0])
    if categorical_cols.any():
        notes.append("Variáveis categóricas preenchidas com a moda.")

    return cleaned, notes


def drop_duplicate_rows(df: pd.DataFrame, enabled: bool) -> Tuple[pd.DataFrame, List[str]]:
    if not enabled:
        return df, []

    before = len(df)
    deduped = df.drop_duplicates()
    removed = before - len(deduped)
    note = f"Linhas duplicadas removidas: {removed}." if removed else "Nenhuma linha duplicada encontrada."
    return deduped, [note]


def scale_features(df: pd.DataFrame, columns: List[str], method: str) -> Tuple[pd.DataFrame, List[str]]:
    if method == "Nenhum" or not columns:
        return df, []

    scaler = MinMaxScaler() if method == "Min-Max (0-1)" else StandardScaler()
    scaled_df = df.copy()
    scaled_values = scaler.fit_transform(scaled_df[columns])
    scaled_columns = [f"{col}_esc" for col in columns]
    scaled_df[scaled_columns] = scaled_values

    note = (
        "Escalonamento Min-Max aplicado. Valores entre 0 e 1."
        if isinstance(scaler, MinMaxScaler)
        else "Padronização Z-Score aplicada. Média 0 e desvio 1."
    )
    return scaled_df, [note + f" Colunas: {', '.join(columns)}."]


def render_visualizations(df: pd.DataFrame) -> None:
    st.subheader("Visualizações rápidas")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        col1, col2 = st.columns(2)
        with col1:
            hist_col = st.selectbox("Histograma", numeric_cols, key="hist_col")
            st.plotly_chart(px.histogram(df, x=hist_col, nbins=20, title=f"Distribuição de {hist_col}"), use_container_width=True)
        with col2:
            scatter_x = st.selectbox("Eixo X", numeric_cols, key="scatter_x")
            scatter_y = st.selectbox("Eixo Y", numeric_cols, key="scatter_y")
            if scatter_x and scatter_y:
                st.plotly_chart(
                    px.scatter(
                        df,
                        x=scatter_x,
                        y=scatter_y,
                        title="Relação entre variáveis",
                        trendline="ols",
                    ),
                    use_container_width=True,
                )

        corr = df[numeric_cols].corr()
        st.plotly_chart(
            px.imshow(corr, text_auto=True, color_continuous_scale="Blues", title="Matriz de correlação"),
            use_container_width=True,
        )
    else:
        st.info("Adicione ao menos uma coluna numérica para gerar gráficos.")


def main() -> None:
    st.title("Pipeline de Data Analytics")
    st.write(
        "Configure um pipeline simples para explorar, limpar e escalar dados diretamente no navegador. "
        "Envie um CSV ou utilize o conjunto sintético disponível para começar."
    )

    with st.sidebar:
        st.header("Configurações")
        uploaded_file = st.file_uploader("Envie um arquivo CSV", type=["csv"])
        use_sample = st.checkbox("Usar conjunto sintético", value=uploaded_file is None)

        missing_strategy = st.selectbox("Tratamento de ausentes", ["Nenhum", "Média", "Mediana", "Moda"])
        drop_dupes = st.checkbox("Remover duplicidades", value=True)
        scale_method = st.selectbox("Escalonamento", ["Nenhum", "Padronização (Z-Score)", "Min-Max (0-1)"])

    df: pd.DataFrame | None = None
    if uploaded_file:
        df = load_data(uploaded_file)
    elif use_sample:
        df = generate_sample_data()

    if df is None:
        st.info("Envie um CSV ou selecione o conjunto sintético para iniciar.")
        return

    st.subheader("Pré-visualização")
    st.write(df.head())
    st.caption(f"Dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scale_cols = st.multiselect("Colunas para escalonamento", numeric_cols, default=numeric_cols)

    if st.button("Executar pipeline"):
        log: List[str] = []
        processed, notes = handle_missing_values(df, missing_strategy)
        log.extend(notes)

        processed, notes = drop_duplicate_rows(processed, drop_dupes)
        log.extend(notes)

        processed, notes = scale_features(processed, scale_cols, scale_method)
        log.extend(notes)

        st.success("Pipeline executado!")

        st.subheader("Resultado processado")
        st.dataframe(processed.head())

        st.markdown("**Estatísticas descritivas**")
        st.dataframe(processed.describe(include="all", datetime_is_numeric=True))

        render_visualizations(processed)

        st.subheader("Log do pipeline")
        if log:
            for item in log:
                st.write(f"• {item}")
        else:
            st.write("Nenhuma transformação aplicada.")


if __name__ == "__main__":
    main()
