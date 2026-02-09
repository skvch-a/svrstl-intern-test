from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


@dataclass
class ProcessingMetadata:
    """Класс для хранения информации о примененных преобразованиях."""

    dropped_columns: list[str] = field(default_factory=list)
    fill_strategy: dict[str, float | str] = field(default_factory=dict)
    numeric_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    encoded_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)


class DataPreprocessor:
    """Класс для предварительной обработки табличных данных."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        self._metadata = ProcessingMetadata()

    @property
    def metadata(self) -> ProcessingMetadata:
        """Геттер для метаданных."""
        return self._metadata

    def remove_missing(self, threshold: float = 0.5) -> None:
        """Удаляет столбцы с долей пропусков > threshold.

        Остальные пропуски заполняет:
        - Медианой для чисел.
        - Модой для категорий.
        """
        missing_ratio = self.df.isna().mean()
        cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()

        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)
            self._metadata.dropped_columns = cols_to_drop

        cols_with_na = self.df.columns[self.df.isna().any()]
        for col in cols_with_na:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                fill_val = self.df[col].median()
            else:
                mode_res = self.df[col].mode()
                fill_val = mode_res[0] if not mode_res.empty else "Unknown"

            self._metadata.fill_strategy[col] = fill_val
            self.df[col] = self.df[col].fillna(fill_val)

    def encode_categorical(self) -> None:
        """One-Hot Encoding."""
        n_rows = len(self.df)
        cat_cols = []

        for col in self.df.select_dtypes(include=["object", "category"]).columns:
            n_unique = self.df[col].nunique(dropna=True)
            unique_ratio = n_unique / n_rows

            # Отсекаем некатегориальные столбцы (с большим кол-вом уникальных значений)
            max_unique = 20
            max_unique_ratio = 0.1
            if n_unique <= max_unique and unique_ratio <= max_unique_ratio:
                cat_cols.append(col)

        self._metadata.categorical_columns = cat_cols

        if not cat_cols:
            return

        cols_before = set(self.df.columns)
        self.df = pd.get_dummies(self.df, columns=cat_cols, dtype=int)
        cols_after = set(self.df.columns)
        self._metadata.encoded_columns = list(cols_after - cols_before)

    def normalize_numeric(self, method: Literal["minmax", "std"] = "minmax") -> None:
        """Нормализует числовые столбцы."""
        numeric_cols = self.df.select_dtypes(include=["number"]).columns

        for col in numeric_cols:
            if col in self._metadata.encoded_columns:
                continue

            series = self.df[col].std()
            stats = {
                "min": series.min(),
                "max": series.max(),
                "mean": series.mean(),
                "std": series.std(),
            }
            self._metadata.numeric_stats[col] = stats

            match method:
                case "minmax":
                    denom = stats["max"] - stats["min"]
                    if denom != 0:
                        self.df[col] = (series - stats["min"]) / denom
                case "std":
                    if stats["std"] != 0:
                        self.df[col] = (series - stats["mean"]) / stats["std"]

    def fit_transform(
        self, threshold: float = 0.5, norm_method: Literal["minmax", "std"] = "minmax"
    ) -> pd.DataFrame:
        """Запуск пайплайна."""
        try:
            self.remove_missing(threshold)
            self.encode_categorical()
            self.normalize_numeric(norm_method)
        except Exception as e:
            err_msg = f"Ошибка при выполнении pipeline: {e}"
            raise RuntimeError(err_msg) from e
        else:
            return self.df
