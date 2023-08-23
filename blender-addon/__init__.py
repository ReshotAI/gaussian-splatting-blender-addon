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

        start_time = time.time()
        
        bpy.context.scene.render.engine = 'CYCLES'

        ##############################
        # Load PLY
        ##############################

        plydata = PlyData.read(self.filepath)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        ##############################
        # Load PLY
        ##############################

        mesh = bpy.data.meshes.new(name="Mesh")
        mesh.from_pydata(xyz.tolist(), [], [])
        mesh.update()

        opacity_attr = mesh.attributes.new(name="opacity", type='FLOAT', domain='POINT')
        for i, v in enumerate(mesh.vertices):
            opacity_attr.data[i].value = opacities[i]
        
        scale_attr = mesh.attributes.new(name="scale", type='FLOAT_VECTOR', domain='POINT')
        for i, v in enumerate(mesh.vertices):
            scale_attr.data[i].vector = scales[i]
        
        color_attr = mesh.attributes.new(name="color", type='FLOAT_VECTOR', domain='POINT')
        for i, v in enumerate(mesh.vertices):
            color_attr.data[i].vector = features_dc[i]

        # rot_attr = mesh.attributes.new(name="rotation", type='FLOAT_VECTOR', domain='POINT')
        # for i, v in enumerate(mesh.vertices):
        #     rot_attr.data[i].vector = rots[i]

        obj = bpy.data.objects.new("GaussianSplatting", mesh)
        bpy.context.collection.objects.link(obj)

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)


        ##############################
        # Materials
        ##############################

        mat = bpy.data.materials.new(name="GaussianSplatting")
        mat.use_nodes = True

        mat_tree = mat.node_tree

        for node in mat_tree.nodes:
            mat_tree.nodes.remove(node)

        color_attr_node = mat_tree.nodes.new('ShaderNodeAttribute')
        color_attr_node.location = (0, 0)
        color_attr_node.attribute_name = "color"
        color_attr_node.attribute_type = 'INSTANCER'

        opacity_attr_node = mat_tree.nodes.new('ShaderNodeAttribute')
        opacity_attr_node.location = (0, -200)
        opacity_attr_node.attribute_name = "opacity"
        opacity_attr_node.attribute_type = 'INSTANCER'

        principled_node = mat_tree.nodes.new('ShaderNodeBsdfPrincipled')
        principled_node.location = (200, 0)
        principled_node.inputs["Base Color"].default_value = (0, 0, 0, 1)
        principled_node.inputs["Specular"].default_value = 0
        principled_node.inputs["Roughness"].default_value = 0

        output_node = mat_tree.nodes.new('ShaderNodeOutputMaterial')
        output_node.location = (400, 0)

        mat_tree.links.new(
            color_attr_node.outputs["Color"],
            principled_node.inputs["Emission"]
        )

        mat_tree.links.new(
            opacity_attr_node.outputs["Fac"],
            principled_node.inputs["Alpha"]
        )

        mat_tree.links.new(
            principled_node.outputs["BSDF"],
            output_node.inputs["Surface"]
        )




        ##############################
        # Geometry Nodes
        ##############################

        geo_node_mod = obj.modifiers.new(name="GeometryNodes", type='NODES')

        geo_tree = bpy.data.node_groups.new(name="GaussianSplatting", type='GeometryNodeTree')
        geo_node_mod.node_group = geo_tree

        for node in geo_tree.nodes:
            geo_tree.nodes.remove(node)
        
        geo_tree.inputs.new('NodeSocketGeometry', "Geometry")
        geo_tree.outputs.new('NodeSocketGeometry', "Geometry")

        group_input_node = geo_tree.nodes.new('NodeGroupInput')
        group_input_node.location = (0, 0)

        mesh_to_points_node = geo_tree.nodes.new('GeometryNodeMeshToPoints')
        mesh_to_points_node.location = (200, 0)

        ico_node = geo_tree.nodes.new('GeometryNodeMeshIcoSphere')
        ico_node.location = (200, 200)
        ico_node.inputs["Subdivisions"].default_value = 3
        ico_node.inputs["Radius"].default_value = 0.005

        set_shade_smooth_node = geo_tree.nodes.new('GeometryNodeSetShadeSmooth')
        set_shade_smooth_node.location = (200, 200)

        instance_node = geo_tree.nodes.new('GeometryNodeInstanceOnPoints')
        instance_node.location = (400, 0)

        set_material_node = geo_tree.nodes.new('GeometryNodeSetMaterial')
        set_material_node.location = (600, 0)
        set_material_node.inputs["Material"].default_value = mat

        group_output_node = geo_tree.nodes.new('NodeGroupOutput')
        group_output_node.location = (800, 0)

        geo_tree.links.new(
            group_input_node.outputs["Geometry"],
            mesh_to_points_node.inputs["Mesh"]
        )

        geo_tree.links.new(
            ico_node.outputs["Mesh"],
            set_shade_smooth_node.inputs["Geometry"]
        )

        geo_tree.links.new(
            set_shade_smooth_node.outputs["Geometry"],
            instance_node.inputs["Instance"]
        )

        geo_tree.links.new(
            mesh_to_points_node.outputs["Points"],
            instance_node.inputs["Points"]
        )

        geo_tree.links.new(
            instance_node.outputs["Instances"],
            set_material_node.inputs["Geometry"]
        )

        geo_tree.links.new(
            set_material_node.outputs["Geometry"],
            group_output_node.inputs["Geometry"]
        )

        print("Processing time: ", time.time() - start_time)

        return {'FINISHED'}

class GaussianSplattingPanel(bpy.types.Panel):
    
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    bl_category = "Gaussian Splatting"
    bl_label = "Gaussian Splatting"

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
