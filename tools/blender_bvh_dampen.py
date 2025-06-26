import argparse
import sys

import bpy
import mathutils


def parse_args():
    """Parse command line arguments after `--`."""
    argv = sys.argv
    if "--" not in argv:
        return None, []
    argv = argv[argv.index("--") + 1 :]

    parser = argparse.ArgumentParser(description="Dampen motion of specific joints in a BVH file.")
    parser.add_argument("--bvh", type=str, required=True, help="Path to BVH file.")
    parser.add_argument("--joints", type=str, required=True, help="Comma-separated list of joint names.")
    parser.add_argument("--factor", type=float, default=0.5, help="Damping factor (0.0 to 1.0).")
    args = parser.parse_args(argv)
    return args, argv


def load_bvh(filepath):
    bpy.ops.import_anim.bvh(filepath=filepath)


def dampen_joint_motion(joint_name, factor):
    """Reduce joint motion by interpolating rotation toward identity."""
    bone = bpy.context.object.pose.bones.get(joint_name)
    if not bone:
        print(f"⚠️ Joint '{joint_name}' not found.")
        return

    for fcurve in bpy.context.object.animation_data.action.fcurves:
        if fcurve.data_path.startswith(f'pose.bones["{joint_name}"].rotation_euler'):
            for keyframe in fcurve.keyframe_points:
                original_value = keyframe.co[1]
                keyframe.co[1] = original_value * (1.0 - factor)


def main():
    args, _ = parse_args()
    if not args:
        print("❌ Failed to parse arguments.")
        return

    print(f"📂 Loading BVH file: {args.bvh}")
    load_bvh(args.bvh)

    joints = [j.strip() for j in args.joints.split(",")]
    print(f"🎯 Joints to dampen: {joints}")
    print(f"🔧 Damping factor: {args.factor}")

    bpy.ops.object.mode_set(mode="POSE")
    for joint in joints:
        dampen_joint_motion(joint, args.factor)

    bpy.ops.object.mode_set(mode="OBJECT")
    print("✅ Damping complete.")


if __name__ == "__main__":
    main()
