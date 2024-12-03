from aitviewer.renderables.skeletons import Skeletons
from aitviewer.viewer import Viewer

from aitviewer.configuration import CONFIG as C
C.window_type = "pyqt6"

if __name__ == "__main__":
    bvh = Skeletons.from_bvh("/Users/ojas/projects/data/Session_1/c/c-cut.bvh")
    viewer = Viewer()
    viewer.scene.add(bvh)

    viewer.center_view_on_node(bvh)
    viewer.auto_set_camera_target = False

    viewer.run_animations = True
    viewer.run()