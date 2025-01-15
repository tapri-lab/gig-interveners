import bpy

bpy.ops.import_anim.bvh("")

armature = bpy.context.object
for bone in armature.pose.bones:
    print(bone.name)

def dampen_bone_motion(bone_name, dampening_factor):
    bone = armature.pose.bones[bone_name]
    for fcurve in bone.animation_data.action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.co[1] *= dampening_factor


bones_to_dampen = ["Hips", "Spine", "Spine1", "Spine2", "Neck", "Head"] # read this from a config file
dampening_factor = 0.5 # read this from a config file

for bone in bones_to_dampen:
    dampen_bone_motion(bone, dampening_factor)