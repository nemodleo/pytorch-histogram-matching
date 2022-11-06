# pytorch-histogram-matching

## Installation
```bash
pip install pytorch_histogram_matching
```

## Usage
```python
from pytorch_histogram_matching import Histogram_Matching

import torch
dst = torch.randint(0, 256, (4, 3, 512,512)).cuda() / 255.
ref = torch.randint(0, 256, (4, 3, 512,512)).cuda() / 255.

HM = Histogram_Matching()
rst = HM(dst, ref)
```

## Test
```bash
python test.py
```
![img](src/total.jpg)