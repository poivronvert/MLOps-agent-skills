import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def table(tmp_path):
    path = tmp_path / "table.csv"
    pd.DataFrame(
        {"x": np.arange(60), "kind": ["a", "b", "c"] * 20, "y": ["ok", "fault"] * 30}
    ).to_csv(path, index=False)
    return path
