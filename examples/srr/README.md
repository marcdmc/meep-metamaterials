## Split Ring Resonators

Examples on how to run a simulation of split ring resonators (SRR) metamaterial and retrieve the effective parameters.

Retrieval of:
- S parameters: $S_{11}, S_{21}$.
- Permittivity $\epsilon$ and permeability $\mu$.
- Index of refraction $n$.

To run the example in parallel use:
``` bash
$ ./run_srr.sh
```
outside of a conda environment.

In the script one can change the number of workers (to 8 for example) by setting:
```bash
#SBATCH -n 8

mpirun -n 8 python examples/srr/srr.py > output.txt
```