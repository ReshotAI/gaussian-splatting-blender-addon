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
import random

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

        N = len(xyz)

        mesh = bpy.data.meshes.new(name="Mesh")
        mesh.from_pydata(xyz.tolist(), [], [])
        mesh.update()

        attr = mesh.attributes.new(name="opacity", type='FLOAT', domain='POINT')
        for i, v in enumerate(mesh.vertices):
            attr.data[i].value = opacities[i]

        obj = bpy.data.objects.new("GaussianSplatting", mesh)
        bpy.context.collection.objects.link(obj)

        geo_node_mod = obj.modifiers.new(name="GeometryNodes", type='NODES')

        geo_tree = bpy.data.node_groups.new(name="GaussianSplatting", type='GeometryNodeTree')
        geo_node_mod.node_group = geo_tree

        # Clear default nodes
        for node in geo_tree.nodes:
            geo_tree.nodes.remove(node)

        # Add a Group Input node
        group_input_node = geo_tree.nodes.new('NodeGroupInput')
        group_input_node.location = (0, 0)

        # Add MeshToPoints node
        mesh_to_points_node = geo_tree.nodes.new('GeometryNodeMeshToPoints')
        mesh_to_points_node.location = (100, 0)

        # Add Icosphere node
        ico_node = geo_tree.nodes.new('GeometryNodeMeshIcoSphere')
        ico_node.location = (100, 100)
        ico_node.inputs["Subdivisions"].default_value = 1
        ico_node.inputs["Radius"].default_value = 0.01

        # Add InstanceOnPoints node
        instance_node = geo_tree.nodes.new('GeometryNodeInstanceOnPoints')
        instance_node.location = (200, 0)

        # Connect MeshToPoints to InstanceOnPoints
        # geo_tree.links.new(
        #     mesh_to_points_node.outputs["Geometry"],
        #     instance_node.inputs["Geometry"]
        # )

        # Add a Group Output node
        group_output_node = geo_tree.nodes.new('NodeGroupOutput')
        group_output_node.location = (300, 0)

        # Connect InstanceOnPoints to Group Output
        # geo_tree.links.new(
        #     instance_node.outputs["Geometry"],
        #     group_output_node.inputs["Geometry"]
        # )

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
