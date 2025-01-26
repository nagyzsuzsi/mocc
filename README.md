# Overview
This repository contains the prototpye implementation of the proposed Multi-perspective Online Conformance Checking (MOCC) approach presented in the paper __*An Alignment-based Multi-Perspective Online Conformance Checking Technique*__ by Zsuzsanna Nagy and Agnes Werner-Stark. The paper is available [here](https://acta.uni-obuda.hu/Nagy_Werner-Stark_122.pdf).

This prototype was implemented using version 2.7.12 of the [PM4Py library](https://github.com/process-intelligence-solutions/pm4py) and [the implementation of the proposed incremental A* approach](https://github.com/fit-daniel-schuster/online_process_monitoring_using_incremental_state-space_expansion_an_exact_algorithm). The source code of the incremental A* algorithm was modified to be able to calculate multi-perspective (prefix-)alignemnts and integrated into this version of PM4Py. Additionally, program codes were developed to allow importing DPN process models from PNML files and to generate and solve a MILP problem for determining the OVAs based on the given variable writings and guard functions.

# Repository structure
- [pm4py](pm4py): Modified version of the PM4Py library used in the experiments.
- [input](input): Contains DPN process models and event logs utilized during the experiments.
- [experiments.py](experiments.py): Contains the code for implementing and running the experiments.

# Changes made to the PM4Py library
- Added files:
  - [incremental_a_star_mp.py](pm4py/algo/conformance/alignments/petri_net/variants/incremental_a_star_mp.py): Modified version of the incremental A* algorithm used by the MOCC approach to calculate multi-perspective (prefix-)alignments.
  - [mp_utils.py](pm4py/algo/conformance/alignments/petri_net/utils/mp_utils.py): Contains functions to generate and solve a MILP problem for finding the Optimal Variable Assignment (OVA) for an alignment, given a variable writing sequence.
  - [incremental_a_star.py](pm4py/algo/conformance/alignments/petri_net/variants/incremental_a_star.py): Modified version of the incremental A* algorithm to support both offline and online alignments.
- Modified files:
  - [obj.py](pm4py/objects/petri_net/obj.py): Added the DataPetriNet class to handle DPNs (Data Petri Nets).
  - [pnml.py](pm4py/objects/petri_net/importer/variants/pnml.py): Extended to import DPN process models with initial variable values.
  - [petri_utils.py](pm4py/objects/petri_net/utils/petri_utils.py): Extended to support DPNs.
  - [align_utils.py](pm4py/objects/petri_net/utils/align_utils.py): Updated the standard cost function to assign a cost of 1 for every deviation from the model.
