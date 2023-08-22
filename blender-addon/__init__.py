bl_info = {
    "name": "Gaussian Splatting Importer",
    "author": "Alex Carlier",
    "version": (0, 0, 1),
    "blender": (2, 80, 0),
    "location": "3D Viewport > Sidebar > Gaussian Splatting",
    "description": "Import Gaussian Splatting scenes",
}

import bpy
import numpy as np
import time

from .plyfile import PlyData, PlyElement


class OBJECT_OT_AddTenCircles(bpy.types.Operator):
    bl_idname = "object.add_ten_circles"
    bl_label = "Add 10 Circles"
    bl_description = "Adds 10 circle meshes to the scene"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        for i in range(10):
            bpy.ops.mesh.primitive_circle_add()
        return {'FINISHED'}


class OBJECT_OT_ImportGaussianSplatting(bpy.types.Operator):
    bl_idname = "object.import_gaussian_splatting"
    bl_label = "Import Gaussian Splatting"
    bl_description = "Import a Gaussian Splatting file into the scene"
    bl_options = {"REGISTER", "UNDO"}
    
    filepath: bpy.props.StringProperty(
        name="File Path",
        description="Path to the Gaussian Splatting file",
        default="",
        maxlen=1024,
        subtype='FILE_PATH',
    )

    def execute(self, context):
        if not self.filepath:
            self.report({'WARNING'}, "Filepath is not set!")
            return {'CANCELLED'}
        
        # bpy.ops.import_mesh.ply(filepath=self.filepath)  # Use Blender's built-in ply importer

        plydata = PlyData.read(self.filepath)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        start_time = time.time()

        # Create a new uv sphere called "Ellipsoid"
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.01, location=(0, 0, 0))
        bpy.context.active_object.name = "Ellipsoid"


        N = len(xyz)

        for i in range(N):
            x, y, z = xyz[i]

            # bpy.ops.mesh.primitive_uv_sphere_add(radius=0.01, location=(x, y, z))
            # bpy.ops.surface.primitive_nurbs_surface_sphere_add(radius=0.01, location=(x, y, z))

            new_ellipsoid = bpy.data.objects["Ellipsoid"].copy()
            new_ellipsoid.location = (x, y, z)
            # new_ellipsoid.rotation_euler = (rot_x, rot_y, rot_z)
            # new_ellipsoid.scale = (scale_x, scale_y, scale_z)

            # Link the new ellipsoid to the scene
            bpy.context.collection.objects.link(new_ellipsoid)

            # # Set the material color
            # mat = bpy.data.materials.new(name="Material" + str(row))
            # mat.diffuse_color = (color_r, color_g, color_b, 1)  # The last value is alpha
            # new_ellipsoid.data.materials.append(mat)


        print("Processing time: ", time.time() - start_time)

        return {'FINISHED'}

class GaussianSplattingPanel(bpy.types.Panel):
    
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    bl_category = "Gaussian Splatting 123"
    bl_label = "Gaussian Splatting 123"

    def draw(self, context):
        layout = self.layout

        # Filepath input
        layout.prop(context.scene, "ply_file_path", text="PLY Filepath")
        
        # Import PLY button
        row = layout.row()
        row.operator(OBJECT_OT_ImportGaussianSplatting.bl_idname, text="Import Gaussian Splatting").filepath = context.scene.ply_file_path


def register():
    bpy.utils.register_class(OBJECT_OT_ImportGaussianSplatting)
    bpy.utils.register_class(GaussianSplattingPanel)
    bpy.utils.register_class(OBJECT_OT_AddTenCircles)

    bpy.types.Scene.ply_file_path = bpy.props.StringProperty(name="PLY Filepath", subtype='FILE_PATH')

def unregister():
    bpy.utils.unregister_class(OBJECT_OT_ImportGaussianSplatting)
    bpy.utils.unregister_class(GaussianSplattingPanel)
    bpy.utils.unregister_class(OBJECT_OT_AddTenCircles)

    del bpy.types.Scene.ply_file_path

if __name__ == "__main__":
    register()
