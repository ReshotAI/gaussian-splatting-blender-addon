bl_info = {
    "name": "Gaussian Splatting Importer",
    "author": "Alex Carlier",
    "version": (0, 0, 1),
    "blender": (2, 80, 0),
    "location": "3D Viewport > Sidebar > Gaussian Splatting",
    "description": "Import Gaussian Splatting scenes",
}

import bpy


class OBJECT_OT_AddTenCircles(bpy.types.Operator):
    bl_idname = "object.add_ten_circles"
    bl_label = "Add 10 Circles"
    bl_description = "Adds 10 circle meshes to the scene"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        for i in range(10):
            bpy.ops.mesh.primitive_circle_add()
        return {'FINISHED'}


class GaussianSplattingPanel(bpy.types.Panel):
    
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    bl_category = "Gaussian Splatting"
    bl_label = "Gaussian Splatting"

    def draw(self, context):
        row = self.layout.row()

        row.operator(OBJECT_OT_AddTenCircles.bl_idname, text="Add 10 Circles")


def register():
    bpy.utils.register_class(GaussianSplattingPanel)
    bpy.utils.register_class(OBJECT_OT_AddTenCircles)

def unregister():
    bpy.utils.unregister_class(GaussianSplattingPanel)
    bpy.utils.unregister_class(OBJECT_OT_AddTenCircles)

if __name__ == "__main__":
    register()
