![](doc/img/wordart.png)
Tools for the simulation and study of metamaterials using [MEEP](https://github.com/NanoComp/meep) (FTDT).
 
## Features
- MEEP simulation wrapper for easy simulation of metamaterials and effective parameter retrieval.
- Lithography printing translation of metamaterials defined from MEEP geometry.

## Installation and usage
```bash
$ git clone https://github.com/marcdmc/metamaterials.git
$ cd metamaterials
$ pip install .
```
### Serial run
```bash
$ conda create -n mp -c conda-forge pymeep pymeep-extras
$ conda activate mp
```

In Python:
```python
import meep_retrieval as mr
```

### MPI
```bash
$ conda create -n pmp -c conda-forge pymeep=*=mpi_mpich_*
$ conda activate pmp
```
Running an example with MPI
```bash
$ mpirun -n 4 python example.py
```

## Documentation
Basic example: how to get the S-parameters from a square grating.
```python
import meep as mp
import numpy as np
from meep_metamaterials import metamaterials as mm

geometry = [mp.Block(mp.Vector3(.6, .2), center=mp.Vector3(), material=mp.Medium(epsilon=12))]
freq_range = np.linspace(0.5, 1.8, 200)
sim = mm.MetamaterialSimulation(period=1.2, geometry=geometry, freq_range=freq_range, dimensions=2)

sim.run()

[S11, S21] = sim.get_s_params()
```