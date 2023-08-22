bl_info = {
    "name": "Gaussian Splatting Importer",
    "author": "Alex Carlier",
    "version": (0, 0, 1),
    "blender": (2, 80, 0),
    "location": "3D Viewport > Sidebar > Gaussian Splatting",
    "description": "Import Gaussian Splatting scenes",
}

import bpy


class GaussianSplattingPanel(bpy.types.Panel):
    
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    bl_category = "Gaussian Splatting"
    bl_label = "Gaussian Splatting"

    def draw(self, context):
        row = self.layout.row()

        row.operator("mesh.primitive_cube_add", text="Add Cube", icon="CUBE")


def register():
    bpy.utils.register_class(GaussianSplattingPanel)

def unregister():
    bpy.utils.unregister_class(GaussianSplattingPanel)

if __name__ == "__main__":
    register()
