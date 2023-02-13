# PopTorch Experimental Addons

A collection of addons to [PopTorch](https://github.com/graphcore/poptorch), with general utility.

## Contents

```python
import poptorch_experimental_addons as pea
```

| API | Description | Note |
| --- | --- | --- |
| `pea.collectives.all_reduce` | All-Reduce | Includes a flag for replacing with identity in the forward or backwards pass |
| `pea.collectives.reduce_scatter` | Reduce-Scatter | |
| `pea.collectives.all_gather` | All-Gather | |
| `pea.collectives.all_to_all` | All-To-All | |
| `pea.sparse.block_coo_spmm` | Static sparse-dense matmul | Also includes weight-sparse modules |
| `pea.misc.scaled` | Separate fwd/bwd scaling | ... |
