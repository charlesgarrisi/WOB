# WOB (Wasserstein Optimal Bidding)

WOB is a Julia-based optimization framework for day-ahead energy market bidding. It implements a data-driven Wasserstein DRO model to find the optimal energy nomination $n^*$ by hedging against the uncertainty of renewable generation and asymmetric dual-pricing imbalance markets.

To start with, we use generated data and do not allow negative price. 

## Quick Start

Clone the repository:
   ```bash
   git clone [https://github.com/charlesgarrisi/WOB.git](https://github.com/charlesgarrisi/WOB.git)
   cd WOB
