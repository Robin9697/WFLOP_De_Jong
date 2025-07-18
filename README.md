# The Joint Optimization of a Wind Farm Layout and Wake Steering
## *A case study of IJmuiden Ver on the co-design of the wind farm layout and yaw control*

Maximizing the extraction of energy from wind farms with ever higher densities is becoming increasingly more important in order to achieve climate targets and simultaneously preserve nature. Improving the yield of a wind farm can be achieved by optimizing the layout, applying control, especially wake steering through yaw control has shown great results, or even combining the optimization of the layout and control into one joint optimization. In this thesis, a case study is performed on the Dutch wind farm ’IJmuiden Ver’ to investigate the real-world applicability of joint optimization. The employed method uses the genetic algorithm, capable of handling the discontinuous domain, and an improved version of the geometric yaw relationship, making coupled or nested optimization redundant. In the IJmuiden Ver case, the levelized cost of electricity (LCOE) of a joint optimized layout compared to a sequential optimized layout is around 0.3% better, even remaining around 0.2% to 0.3% better when shrinking the domain to give nature more space. This shows that joint optimization is applicable in practice and has the potential to increase the yield of a wind farm substantially without significantly increasing the computational intensity of the wind farm layout optimization problem (WFLOP).

## Code Organization

- `src/`: Contains the source code.
- `docs/`: Contains the thesis.
- `src/results/`: Contains the results and figures.
- `src/wflop/`: Contains the code of the developed method.
- `src/wind_rose/`: Contains the code and data to generate the wind rose.
- `setup.py`: Setup file for the project.

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/Robin9697/WFLOP_De_Jong.git
   ```

2. Navigate to the project directory:

   ```bash
   cd WFLOP_De_Jong
   ```

## Installation

Install the package using the provided setup file:

```bash
python setup.py install
```

## Usage

Once installed, you can use the `WFLOP_GA_class` as groundwork for your projects on the WFLOP, which can be found in: `src/wflop/genetic_algorithm`.


## License

This project is licensed under the [GNU GENERAL PUBLIC LICENSE](LICENSE). Feel free to use, modify, and distribute the code in accordance with the terms of the license.