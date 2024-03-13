# Weighted statistics

## What?
We've re-implemented some weighted statistics inspired by the weights R package (https://github.com/cran/weights/tree/master).
At the moment, it's just the weighted t-test, but this may expand.

## How?
```python
import numpy as np
from wstats import wtd_t_test

# Example data for testing
x = np.random.normal(0, 1, 100)
y = np.random.normal(0.5, 1, 100)
wx = np.random.rand(100)
wy = np.random.rand(100)

# Execute tests
# Without bootstrapping
result_no_boot = wtd_t_test(x, y, wx, wy, alternative="two-sided", bootse=False)
# With bootstrapping
result_boot = wtd_t_test(x, y, wx, wy, alternative="two-sided", bootse=True, bootn=int(1e4))

(result_no_boot, result_boot)

```
