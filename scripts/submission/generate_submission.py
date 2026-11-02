"""Kaggle submission: exact original Id ordering, Sales >= 0, closed -> 0."""
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.ml.model_service import infer
from app.services.data_service import get_test, SUBMISSION_FILE

if __name__ == "__main__":
    test = get_test()
    pred = infer(test)
    output = pd.DataFrame({"Id": test.Id.astype(int), "Sales": pred.round(2)})
    assert len(output) == len(test) and output.Id.is_unique
    assert output.Sales.notna().all() and output.Sales.ge(0).all()
    assert output.loc[test.Open.eq(0), "Sales"].eq(0).all()
    SUBMISSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(SUBMISSION_FILE, index=False)
    print(f"Saved: {SUBMISSION_FILE} ({len(output):,} rows)")
