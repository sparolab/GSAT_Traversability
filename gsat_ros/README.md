# GSAT ROS — Dataset Collection

Collect synchronized **perspective LiDAR** scans and **supervision** (pose + traversability pseudo-label) for self-supervised training.

## Quick start

```bash
# Terminal 1: play your bag (or run simulation)
rosbag play <your_bag>.bag

# Terminal 2: collect dataset
roslaunch gsat_ros data_collection.launch
```

Edit `config/data_collection.yaml` before launch (`output_dir`, topics, label params).

## Output layout

Example run (`dataset_name: gazebo_hill`):

```
collect_data/gazebo/hill/original/
├── supervision.csv          # supervision: pose + Travel_label per sample
└── lidar/
    ├── 12734375000000.bin   # perspective sensor frame (odom timestamp, ns)
    ├── 12734987000000.bin
    └── ...
```

| Path | Role |
|------|------|
| `supervision.csv` | Supervision metadata; one row per saved sample |
| `lidar/<timestamp>.bin` | Perspective sensor point cloud at the same timestamp |

Rows in `supervision.csv` and files in `lidar/` are paired by `TIMESTAMP` (odometry header time, nanoseconds).

### `supervision.csv` columns

| Column | Description |
|--------|-------------|
| `TIMESTAMP` | Sample id (ns), matches `lidar/<TIMESTAMP>.bin` |
| `robot_posi_x/y/z` | Robot position (world) |
| `robot_ori_x/y/z/w` | Robot orientation (quaternion) |
| `Travel_label` | Traversability pseudo-label τ ∈ [0, 1] from `cmd_vel` vs. odometry |

### `lidar/*.bin` format

Binary point cloud: each point is **4 × float32** — `x, y, z, intensity` (intensity `0` if missing in the source cloud).

## Optional: `sw_sync`

`data_collection.launch` also starts `sw_sync`, which time-syncs raw LiDAR + odometry and republishes `filter/points` and `filter/odom`. Use it when your bag does not already provide those topics.

Change input topics in `src/sw_sync.cpp` (`lidar_sub_`, `odom_sub_`).

## Configuration (`config/data_collection.yaml`)

```yaml
dataset_name: "gazebo_hill"

gazebo_hill:
  output_dir: "/path/to/collect_data/gazebo/hill/original"
  lidar_topic: "filter/points"
  odom_topic: "filter/odom"
  cmd_vel_topic: "cmd_vel"
  sampling_time: 0.3          # min interval between samples (s)
  queue_size: 10
  max_interval_sec: 0.02      # sync tolerance (s)
  use_pseudo_label: true      # false → Travel_label is always 1 (ignore eta, v_th)
  eta: 20.0                   # used only if use_pseudo_label: true
  v_th: 0.2                   # used only if use_pseudo_label: true
```

**Sampling rules:** samples are saved only when `|cmd_vel.linear.x| > 0` and `dt ≥ sampling_time`.

## Label (`Travel_label`)

### Mode A — `use_pseudo_label: true` (default)

Computed in `src/supervision/convert_pseudo_label.cpp`:

- Body-frame velocity from odometry vs. commanded `cmd_vel`
- Velocity error → sigmoid: τ = σ(−η · (v_error − v_th))
- Higher τ ≈ better command tracking (higher traversability)

Tune `eta` and `v_th` in the yaml.

### Mode B — `use_pseudo_label: false`

No pseudo-label parameters needed. Every sample is written with **`Travel_label = 1`** (fully traversable). Use this when you only need pose + LiDAR and will label later, or when all collected motion is assumed traversable.
