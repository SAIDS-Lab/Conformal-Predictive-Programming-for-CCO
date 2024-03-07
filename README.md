# Conformal-Predictive-Programming-for-CCO
## Overview
This repository contains the implementation of [Conformal Predictive Programming for Chance Constrained Optimization](https://arxiv.org/pdf/2402.07407.pdf). We will walk you through how to use the codes in this repository for your own project and how the implementations for the case studies in the paper are structured. Below is the abstract of the paper:

Motivated by the advances in conformal prediction (CP), we propose conformal predictive programming (CPP), an approach to solve chance constrained optimization (CCO) problems, i.e., optimization problems with nonlinear constraint functions affected by arbitrary random
parameters. CPP utilizes samples from these random parameters along with the quantile lemma – which is central to CP – to transform the CCO problem into a deterministic optimization problem. We then present two tractable reformulations of CPP by: (1) writing the quantile as a linear program along with its KKT conditions (CPP-KKT), and (2) using mixed integer programming (CPP-MIP). CPP comes with marginal probabilistic feasibility guarantees for the CCO problem that are conceptually different from existing approaches, e.g., the sample approximation and the scenario approach. While we explore algorithmic similarities with the sample approximation approach, we emphasize that the strength of CPP is that it can easily be extended to incorporate different variants of CP. To illustrate this, we present robust conformal predictive programming to deal with distribution shifts in the uncertain parameters of the CCO problem. 

<table cellpadding="0" cellspacing="0" border="0" width="100%">
<tr><td align="center">
<img src="featured.png" width = 60%>
</td></tr>
</table>

## Conformal Predictive Programming
The codes for the solvers are included in the `resources` folder. One can access the solvers through the `solve` function provided in `resources/solver`. The `solve` function takes in the following arguments and follows the representation in Equation (3) of the paper:

- x_dim: The dimension of the decision variable x. Acceptable argument is either an integer in the case when x is a scalar or a vector or a size 2 tuple in the case when x is a matrix (see case study 2).

## Contact Information