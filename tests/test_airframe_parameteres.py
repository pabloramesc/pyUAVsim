"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

from simulator.aircraft.airframe_parameters import (
    AirframeParameters,
    load_airframe_parameters_from_json,
    load_airframe_parameters_from_yaml,
    load_airframe_parameters_from_toml,
)

# Test loading from JSON
json_file = r"config/aerosonde_parameters.json"
airframe_params_json = load_airframe_parameters_from_json(json_file)
print("Airframe Parameters (JSON):")
print(airframe_params_json)
print()

# Test loading from YAML
yaml_file = r"config/aerosonde_parameters.yaml"
airframe_params_yaml = load_airframe_parameters_from_yaml(yaml_file)
print("Airframe Parameters (YAML):")
print(airframe_params_yaml)
print()

# Test loading from TOML
toml_file = r"config/aerosonde_parameters.toml"
airframe_params_toml = load_airframe_parameters_from_toml(toml_file)
print("Airframe Parameters (TOML):")
print(airframe_params_toml)
print()
