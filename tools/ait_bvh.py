from aitviewer.renderables.skeletons import Skeletons
from aitviewer.viewer import Viewer
from pathlib import Path
from aitviewer.configuration import CONFIG as C
import tyro

C.window_type = "pyglet"

def main(bvh_path: Path):
    bvh = Skeletons.from_bvh(bvh_path.expanduser(), z_up=True)
    viewer = Viewer()
    viewer.scene.add(bvh)

    viewer.center_view_on_node(bvh)
    viewer.auto_set_camera_target = False

    viewer.run_animations = True
    viewer.run()

if __name__ == "__main__":
    tyro.cli(main)