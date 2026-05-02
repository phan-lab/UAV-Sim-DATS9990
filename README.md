## Setup

Create a virtual environment and activate it

``
python3 -m venv venv
``

``
source venv/bin/activate
``

To install the required packages, run setup.py

``
python3 setup.py
``

## Running the Simulator

``
python3 sandbox.py
``

The number of runs can be configured in the sandbox.py script through the parameter ***N_RUNS***

## Observations

The SE3 controller has been modified to be fault-tolerant by integrating the model's predictions and adding mode transition logic.

The simulation creates a csv file of the IMU readings, along with the injected fault information.
