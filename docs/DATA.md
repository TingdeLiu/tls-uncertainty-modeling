# Data specification

The research measurements are not distributed with this repository.

## PLY schema

The original notebook reads the first PLY element and expects these properties:

| Property | Meaning |
| --- | --- |
| `x`, `y`, `z` | Cartesian point coordinates |
| `scalar_Scan_ID` | Scan identifier |
| `scalar_intensity` | Return intensity |
| `scalar_incidence` | Incidence angle |
| `scalar_spotsize` | Estimated laser spot size |
| `scalar_sigma_dist` | Distance standard deviation |
| `scalar_range_measured` | Measured range |
| `scalar_range_residual` | Target range residual |

The default physical-feature vector contains intensity, incidence angle,
distance standard deviation, and measured range. Spot size is loaded for
reference but was not used by the original RePN experiment.

## Filtering

The cleaned preprocessing function retains records satisfying:

```text
abs(range_residual) < 0.01 m
incidence_angle > 10 degrees
```

This expresses the intent documented in the original notebook and corrects its
ambiguous boolean expression.

## Group arrays

The model expects a local XYZ neighbourhood for every retained target record.
The historical experiment loads one NumPy file per scan:

```text
data/processed/groups/Group_<scan_id>.npy
```

Each file must have shape `(number_of_records, neighbourhood_size, 3)`. The
concatenated file order must match the filtered target and physical-feature
rows. The default model uses up to 128 neighbours.

The original repository does not include the script used to create these group
arrays. This is the main missing step for end-to-end reproduction.

## Data governance

Before sharing the measurements, confirm the dataset owner, consent,
institutional policy, and an appropriate data license. Large generated files
and checkpoints are excluded by `.gitignore`.
