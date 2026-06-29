# Dataset

**File:** `Salaries.csv`

San Francisco city employee salaries ([Kaggle: kaggle/sf-salaries](https://www.kaggle.com/datasets/kaggle/sf-salaries)).

**Download into this folder** (requires `kagglehub`, included in repo `requirements.txt`):

```bash
python -c "
import shutil, kagglehub
from pathlib import Path
src = next(Path(kagglehub.dataset_download('kaggle/sf-salaries')).rglob('Salaries.csv'))
shutil.copy2(src, 'Salaries.csv')
print('Saved Salaries.csv')
"
```

Expected columns include: `EmployeeName`, `JobTitle`, `BasePay`, `OvertimePay`, `TotalPayBenefits`, `Year`
