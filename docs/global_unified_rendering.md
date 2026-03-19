# SMPL Viewer - Global Unified Rendering

This document describes the new `global_unified` parameter added to the `render_smpl_sequences` function.

## Overview

The `render_smpl_sequences` function now supports three different rendering modes:

1. **Individual rendering** (`global_scene=False`, `global_unified=False`)
   - Renders each SMPL sequence individually with its own camera
   - Creates one video per person

2. **Global scene rendering** (`global_scene=True`, `global_unified=False`) 
   - Renders all SMPL sequences together but creates one video per camera-person pair
   - Results in multiple videos per camera angle (one for each person)

3. **Global unified rendering** (`global_unified=True`) **[NEW]**
   - Renders all SMPL sequences together in a single scene
   - Creates exactly one video per camera angle with all persons visible
   - All sequences use consistent coloring
   - Much more efficient than global scene rendering

## CLI Usage

```bash
# Individual rendering (default)
python viz/smpl_viewer.py render --smpl-path /path/to/smpl --bvh-path /path/to/bvh

# Global scene rendering (original behavior)
python viz/smpl_viewer.py render --smpl-path /path/to/smpl --bvh-path /path/to/bvh --global-scene

# Global unified rendering (NEW - recommended for multi-person scenes)
python viz/smpl_viewer.py render --smpl-path /path/to/smpl --bvh-path /path/to/bvh --global-unified

# Additional parameters
python viz/smpl_viewer.py render \
    --smpl-path /path/to/smpl \
    --bvh-path /path/to/bvh \
    --global-unified \
    --skeleton \
    --sigma 10.0 \
    --frame-range 0 1000
```

## Output Structure

### Individual rendering:
```
export/headless/individual/smplx/
├── person1/
│   └── person1.mp4
├── person2/
│   └── person2.mp4
└── ...
```

### Global scene rendering:
```
export/headless/global/smplx/
├── person1/
│   ├── cam_0/
│   │   └── cam_0_person1.mp4
│   ├── cam_1/
│   │   └── cam_1_person1.mp4
│   └── ...
├── person2/
│   ├── cam_0/
│   │   └── cam_0_person2.mp4
│   └── ...
└── ...
```

### Global unified rendering (NEW):
```
export/headless/global_unified/smplx/
├── cam_0/
│   └── cam_0_all.mp4
├── cam_1/
│   └── cam_1_all.mp4
├── cam_2/
│   └── cam_2_all.mp4
└── cam_3/
    └── cam_3_all.mp4
```

## Benefits of Global Unified Rendering

1. **Efficiency**: Generates only 4 videos (one per camera angle) instead of 4×N videos (where N is the number of persons)
2. **Consistency**: All persons appear with the same color scheme
3. **Overview**: Easier to get a full scene overview with all participants visible
4. **Storage**: Significantly reduces output file count and storage requirements

## Parameters

- `global_unified: bool = False` - Enable unified global scene rendering
- `global_scene: bool = False` - Enable original global scene rendering (per camera-person pair)
- `skeleton: bool = False` - Render skeletons instead of SMPL meshes
- `sigma: float = 10.0` - Camera smoothing factor
- `frame_range: List[int] = [0, 5000]` - Frame range to render

Note: `global_unified` takes precedence over `global_scene` if both are True.