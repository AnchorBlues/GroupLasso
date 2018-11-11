GroupLasso
========================

Group Lasso package for Python.


## Installation Guide

Run the following commands:

```
git clone https://github.com/AnchorBlues/GroupLasso.git
cd GroupLasso
python setup.py install
```

## Getting started
Here is the `GroupLassoRegressor` model:

```python
from grouplasso import GroupLassoRegressor
```

Create sample dataset:
```python
import numpy as np
np.random.seed(0)
X = np.random.randn(10, 3)
# target variable is strongly correlated with 0th feature.
y = X[:, 0] + np.random.randn(10) * 0.1
```

Set group_ids, which specify group membership:
```python
# 0th feature and 1st feature are the same group.
group_ids = np.array([0, 0, 1])
```

You can now train Group Lasso:
```python
model = GroupLassoRegressor(group_ids=group_ids, random_state=42, verbose=False, alpha=1e-1)
model.fit(X, y)
```

Note that all the members of a particular group are either selected(`coef_ != 0`) or not selected(`coef_ == 0`).
```python
model.coef_
# array([ 0.84795715, -0.01193528, -0.        ])
```
