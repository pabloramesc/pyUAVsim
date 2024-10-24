# Python UAV Simulator
A Python unmanned aircraft simulator based on the book *Small Unmanned Aircraft: Theory and Practice* by Randy Beard and Tim McLain.

For more information visit https://github.com/randybeard/mavsim_public

## Overview
The Python UAV Simulator provides a comprehensive environment for simulating the behavior of unmanned aerial vehicles (UAVs). It allows users to explore aircraft dynamics, sensor readings, autopilot behavior, and actuator responses in a controlled simulation framework.

![](pics/screenshot_01.png)

## Getting Started

To get started with the Python UAV Simulator, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/pabloramesc/pyUAVsim.git
   ```

2. Open the project folder:
   ```bash
   cd pyUAVsim
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the example:
   ```bash
   python example.py
   ```
   
## Description
The simulator is composed of four main modules, each with distinct responsibilities and submodules. These modules are organized into three levels: Physical, Electrical, and Software.

![](pics/diagram.png)

### Physical Level (Aircraft)
**Forces and Moments**:
Takes the current aircraft state and control deltas from the actuators.
Computes the forces (fx, fy, fz) and moments (l, m, n) acting on the aircraft.

**Kinematics and Dynamics**:
Uses the current state and forces and moments to compute the new state of the aircraft.

### Electrical Level
**Sensors**:
Simulates sensor readings and noise based on the true aircraft state.
Includes IMU (Inertial Measurement Unit), GPS, barometer, etc.

**Actuators**:
Takes autopilot control commands and computes the state of the actuators (servos and DC motors).
Outputs the control deltas for the aircraft simulator, closing the loop in the simulation.

### Software Level (Autopilot)
**Estimation Filters**:
Applies filters (e.g., Low Pass Filter, Kalman Filter) to sensor readings to estimate the aircraft state.

**Controll and Navigation**:
Uses the estimated state along with mission parameters or joystick/controller input to manage attitude control, path following (line or orbit), path managing, and path planning.

## Development

- [x] Main aircraft data handlers: `AircraftParameters`, `AircrafState` ...
- [x] Aircraft forces and moments (gravity, aerodynamic and propulsion) simulation
- [x] Aircraft kinematics and dynamics integration to calculate new state
- [x] Multi functional visualization tool based on Matplotlib
- [x] Trim calculation
- [x] Quaternions support
- [ ] Wind and atmospheric models integration
- [ ] Realistic sensor data simulation including IMU, GPS, and barometer.
- [ ] Actuators simulation (servos and DC motor) with realistic transfer functions
- [ ] Estimation filters for airtcraft's state estimation
- [x] Implement all autopilot layers: flight control, path following, path navigator and route manager.
- [x] Mission parametes (waypoints) support
- [ ] Add TECS and LQR control
- [ ] Add PX4 attitude controller
- [ ] Add control gains auto-tuning method
- [ ] Implement Genetic Algorithm optimization
- [ ] Real-time external joystick/controller and autopilot modes (manual, FBW, auto...) support
- [ ] Unity 3D visualiztion framework 

## Tests
To run all the tests and generate a coverage report in HTML:
```bash
python -m pytest --cov=simulator --cov-report html
```

## License

This project is licensed under the MIT - see the [LICENSE](LICENSE) file for details.
