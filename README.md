![](doc/img/wordart.png)
Tools for the simulation and study of metamaterials using [MEEP](https://github.com/NanoComp/meep) (FTDT).
 
## Features
- MEEP simulation wrapper for easy simulation of metamaterials and effective parameter retrieval.
- Lithography printing translation of metamaterials defined from MEEP geometry.

## Installation and usage
### Serial run
```bash
$ conda create -n mp -c conda-forge pymeep pymeep-extras
$ conda activate mp
```
```python
import meep_retrieval as mr

geom = ...
sim = mr.Simulation()
sim.add_geometry(geom)
sim.run()
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
WIP