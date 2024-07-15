bl_info = {
    "name": "3D Gaussian Splatting",
    "author": "Alex Carlier",
    "version": (0, 0, 1),
    "blender": (4, 0, 0),
    "location": "3D Viewport > Sidebar > 3D Gaussian Splatting",
    "description": "3D Gaussian Splatting tool",
}

import bpy
import bmesh
import mathutils
import numpy as np
import time
import random
from pathlib import Path

from .plyfile import PlyData, PlyElement

SH2RGB_OSL = Path(__file__).parent / "sh2rgb.osl"


class ImportGaussianSplatting(bpy.types.Operator):
    bl_idname = "object.import_gaussian_splatting"
    bl_label = "Import Gaussian Splatting"
    bl_description = "Import a 3D Gaussian Splatting file into the scene"
    bl_options = {"REGISTER", "UNDO"}

    filepath: bpy.props.StringProperty(
        name="File Path",
        description="Path to the Gaussian Splatting file",
        subtype="FILE_PATH",
    )

    def execute(self, context):
        if not self.filepath:
            self.report({'WARNING'}, "Filepath is not set!")
            return {'CANCELLED'}

        start_time_0 = time.time()

        bpy.context.scene.render.engine = 'CYCLES'

        if context.preferences.addons["cycles"].preferences.has_active_device():
            bpy.context.scene.cycles.device = 'GPU'

        bpy.context.scene.cycles.transparent_max_bounces = 20

        RECOMMENDED_MAX_GAUSSIANS = 200_000

        if SH2RGB_OSL.name not in bpy.data.texts:
            bpy.ops.text.open(filepath=str(SH2RGB_OSL), internal=True)
        SH2RGB_OSL_TEXT = bpy.data.texts[SH2RGB_OSL.name]

        bpy.context.scene.cycles.shading_system = True

        ##############################
        # Load PLY
        ##############################

        start_time = time.time()

        plydata = PlyData.read(self.filepath)

        print(f"PLY loaded in {time.time() - start_time} seconds")

        start_time = time.time()

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)

        N = len(xyz)
        print(f"ply data: {plydata.elements[0]}")
        if 'opacity' in plydata.elements[0]:
            log_opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
            opacities = 1 / (1 + np.exp(-log_opacities))
        else:
            log_opacities = np.asarray(1)[..., np.newaxis]
            opacities = 1 / (1 + np.exp(-log_opacities))

        features_dc = np.zeros((N, 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))

        features_extra = np.zeros((N, len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((N, 3, 15))

        log_scales = np.stack((np.asarray(plydata.elements[0]["scale_0"]),
                               np.asarray(plydata.elements[0]["scale_1"]),
                               np.asarray(plydata.elements[0]["scale_2"])), axis=1)

        scales = np.exp(log_scales)

        quats = np.stack((np.asarray(plydata.elements[0]["rot_0"]),
                          np.asarray(plydata.elements[0]["rot_1"]),
                          np.asarray(plydata.elements[0]["rot_2"]),
                          np.asarray(plydata.elements[0]["rot_3"])), axis=1)

        rots_euler = np.zeros((N, 3))

        for i in range(N):
            quat = mathutils.Quaternion(quats[i].tolist())
            euler = quat.to_euler()
            rots_euler[i] = (euler.x, euler.y, euler.z)

        print("Data loaded in", time.time() - start_time, "seconds")

        ##############################
        # Mesh
        ##############################

        start_time = time.time()

        mesh = bpy.data.meshes.new(name="Mesh")
        mesh.from_pydata(xyz.tolist(), [], [])
        mesh.update()

        print("Mesh loaded in", time.time() - start_time, "seconds")

        start_time = time.time()

        log_opacity_attr = mesh.attributes.new(name="log_opacity", type='FLOAT', domain='POINT')
        log_opacity_attr.data.foreach_set("value", log_opacities.flatten())

        opacity_attr = mesh.attributes.new(name="opacity", type='FLOAT', domain='POINT')
        opacity_attr.data.foreach_set("value", opacities.flatten())

        scale_attr = mesh.attributes.new(name="scale", type='FLOAT_VECTOR', domain='POINT')
        scale_attr.data.foreach_set("vector", scales.flatten())

        logscale_attr = mesh.attributes.new(name="logscale", type='FLOAT_VECTOR', domain='POINT')
        logscale_attr.data.foreach_set("vector", log_scales.flatten())

        sh0_attr = mesh.attributes.new(name="sh0", type='FLOAT_VECTOR', domain='POINT')
        sh0_attr.data.foreach_set("vector", features_dc.flatten())

        for j in range(0, 15):
            sh_attr = mesh.attributes.new(name=f"sh{j + 1}", type='FLOAT_VECTOR', domain='POINT')
            sh_attr.data.foreach_set("vector", features_extra[:, :, j].flatten())

        rot_quatxyz_attr = mesh.attributes.new(name="quatxyz", type='FLOAT_VECTOR', domain='POINT')
        rot_quatxyz_attr.data.foreach_set("vector", quats[:, :3].flatten())

        rot_quatw_attr = mesh.attributes.new(name="quatw", type='FLOAT', domain='POINT')
        rot_quatw_attr.data.foreach_set("value", quats[:, 3].flatten())

        rot_euler_attr = mesh.attributes.new(name="rot_euler", type='FLOAT_VECTOR', domain='POINT')
        rot_euler_attr.data.foreach_set("vector", rots_euler.flatten())

        obj = bpy.data.objects.new("GaussianSplatting", mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        obj.rotation_mode = 'XYZ'
        obj.rotation_euler = (-np.pi / 2, 0, 0)

        obj["gaussian_splatting"] = True

        print("Mesh attributes added in", time.time() - start_time, "seconds")

        ##############################
        # Materials
        ##############################

        start_time = time.time()

        mat = bpy.data.materials.new(name="GaussianSplatting")
        mat.use_nodes = True
        mat.blend_method = "HASHED"

        mat_tree = mat.node_tree

        for node in mat_tree.nodes:
            mat_tree.nodes.remove(node)

        sh_attr_nodes = []
        sh_inst_attr_nodes = []  # ellipsoids
        sh_geom_attr_nodes = []  # point cloud

        for j in range(0, 16):
            sh_inst_attr_node = mat_tree.nodes.new('ShaderNodeAttribute')
            sh_inst_attr_node.location = (1800, 200 * j)
            sh_inst_attr_node.attribute_name = f"sh{j}"
            sh_inst_attr_node.attribute_type = 'INSTANCER'
            sh_inst_attr_nodes.append(sh_inst_attr_node)

            sh_geom_attr_node = mat_tree.nodes.new('ShaderNodeAttribute')
            sh_geom_attr_node.location = (1800, 200 * j)
            sh_geom_attr_node.attribute_name = f"sh{j}"
            sh_geom_attr_node.attribute_type = 'GEOMETRY'
            sh_geom_attr_nodes.append(sh_geom_attr_node)

            vector_math_node = mat_tree.nodes.new('ShaderNodeVectorMath')
            vector_math_node.operation = 'ADD'
            vector_math_node.location = (2000, 200 * j)

            mat_tree.links.new(
                sh_inst_attr_node.outputs["Vector"],
                vector_math_node.inputs[0]
            )

            mat_tree.links.new(
                sh_geom_attr_node.outputs["Vector"],
                vector_math_node.inputs[1]
            )

            sh_attr_nodes.append(vector_math_node)

        sh = [sh_attr_node.outputs["Vector"] for sh_attr_node in sh_attr_nodes]

        position_attr_node = mat_tree.nodes.new('ShaderNodeAttribute')
        position_attr_node.attribute_name = "position"
        position_attr_node.attribute_type = 'GEOMETRY'
        position_attr_node.location = (0, 0)

        opacity_geom_attr_node = mat_tree.nodes.new('ShaderNodeAttribute')
        opacity_geom_attr_node.location = (2800, -200)
        opacity_geom_attr_node.attribute_name = "opacity"
        opacity_geom_attr_node.attribute_type = 'GEOMETRY'

        opacity_inst_attr_node = mat_tree.nodes.new('ShaderNodeAttribute')
        opacity_inst_attr_node.location = (2800, -200)
        opacity_inst_attr_node.attribute_name = "opacity"
        opacity_inst_attr_node.attribute_type = 'INSTANCER'

        opacity_attr_node = mat_tree.nodes.new('ShaderNodeMath')
        opacity_attr_node.operation = 'ADD'
        opacity_attr_node.location = (2800, -200)

        mat_tree.links.new(
            opacity_geom_attr_node.outputs["Fac"],
            opacity_attr_node.inputs[0]
        )

        mat_tree.links.new(
            opacity_inst_attr_node.outputs["Fac"],
            opacity_attr_node.inputs[1]
        )

        principled_node = mat_tree.nodes.new('ShaderNodeBsdfPrincipled')
        principled_node.location = (3200, 600)
        principled_node.inputs["Base Color"].default_value = (0, 0, 0, 1)
        principled_node.inputs["Specular IOR Level"].default_value = 0
        principled_node.inputs["Roughness"].default_value = 0
        principled_node.inputs["Emission Strength"].default_value = 1

        output_node = mat_tree.nodes.new('ShaderNodeOutputMaterial')
        output_node.location = (3600, 0)

        # Camera location
        combine_xyz_node = mat_tree.nodes.new('ShaderNodeCombineXYZ')
        combine_xyz_node.location = (-200, 200)

        vector_transform_node = mat_tree.nodes.new('ShaderNodeVectorTransform')
        vector_transform_node.vector_type = 'POINT'
        vector_transform_node.convert_from = 'CAMERA'
        vector_transform_node.convert_to = 'WORLD'
        vector_transform_node.location = (0, 200)

        mat_tree.links.new(
            combine_xyz_node.outputs["Vector"],
            vector_transform_node.inputs["Vector"]
        )

        # View direction

        dir_node = mat_tree.nodes.new('ShaderNodeVectorMath')
        dir_node.operation = 'SUBTRACT'
        dir_node.location = (200, 200)

        normalize_node = mat_tree.nodes.new('ShaderNodeVectorMath')
        normalize_node.operation = 'NORMALIZE'
        normalize_node.location = (400, 200)

        mat_tree.links.new(position_attr_node.outputs["Vector"], dir_node.inputs[0])

        mat_tree.links.new(vector_transform_node.outputs["Vector"], dir_node.inputs[1])

        mat_tree.links.new(dir_node.outputs["Vector"], normalize_node.inputs[0])

        # Load external osl shader
        script_node = mat_tree.nodes.new("ShaderNodeScript")
        script_node.script = SH2RGB_OSL_TEXT
        script_node.location = (2600, 200)

        mat_tree.links.new(normalize_node.outputs["Vector"], script_node.inputs["xyz"])
        for j in range(0, 16):
            mat_tree.links.new(
                sh_attr_nodes[j].outputs["Vector"], script_node.inputs[f"sh{j}"]
            )

        math_node = mat_tree.nodes.new("ShaderNodeVectorMath")
        math_node.operation = "ADD"
        math_node.inputs[1].default_value = (0.5, 0.5, 0.5)
        math_node.location = (2800, 200)

        gamma_node = mat_tree.nodes.new('ShaderNodeGamma')
        gamma_node.inputs["Gamma"].default_value = 2.2
        gamma_node.location = (3000, 200)

        # mat_tree.links.new(add_node.outputs["Vector"], math_node.inputs[0])
        mat_tree.links.new(script_node.outputs["result"], math_node.inputs[0])

        mat_tree.links.new(
            math_node.outputs["Vector"],
            gamma_node.inputs["Color"]
        )

        mat_tree.links.new(
            gamma_node.outputs["Color"],
            principled_node.inputs["Emission Color"],
        )

        geometry_node = mat_tree.nodes.new('ShaderNodeNewGeometry')
        geometry_node.location = (2600, 0)

        vector_math_node = mat_tree.nodes.new('ShaderNodeVectorMath')
        vector_math_node.operation = 'DOT_PRODUCT'
        vector_math_node.location = (2800, 0)

        mat_tree.links.new(
            geometry_node.outputs["Normal"],
            vector_math_node.inputs[0]
        )

        mat_tree.links.new(
            geometry_node.outputs["Incoming"],
            vector_math_node.inputs[1]
        )

        math_node = mat_tree.nodes.new('ShaderNodeMath')
        math_node.operation = 'MULTIPLY'
        math_node.location = (3000, 0)

        mat_tree.links.new(
            opacity_attr_node.outputs["Value"],
            math_node.inputs[0]
        )

        mat_tree.links.new(
            vector_math_node.outputs["Value"],
            math_node.inputs[1]
        )

        mat_tree.links.new(
            math_node.outputs["Value"],
            principled_node.inputs["Alpha"]
        )

        # Output

        mat_tree.links.new(
            principled_node.outputs["BSDF"],
            output_node.inputs["Surface"]
        )

        print("Material created in", time.time() - start_time, "seconds")

        ##############################
        # Geometry Nodes
        ##############################

        start_time = time.time()

        geo_node_mod = obj.modifiers.new(name="Geometry Nodes", type='NODES')

        geo_tree = bpy.data.node_groups.new(name="GaussianSplatting", type='GeometryNodeTree')
        geo_node_mod.node_group = geo_tree

        for node in geo_tree.nodes:
            geo_tree.nodes.remove(node)

        # geo_tree.inputs.new('NodeSocketGeometry', "Geometry")
        # geo_tree.outputs.new('NodeSocketGeometry', "Geometry")
        geo_tree.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
        geo_tree.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')

        group_input_node = geo_tree.nodes.new('NodeGroupInput')
        group_input_node.location = (0, 0)

        mesh_to_points_node = geo_tree.nodes.new('GeometryNodeMeshToPoints')
        mesh_to_points_node.location = (200, 0)
        mesh_to_points_node.inputs["Radius"].default_value = 0.01

        # Thresholding

        opacity_attr_gn_node = geo_tree.nodes.new('GeometryNodeInputNamedAttribute')
        opacity_attr_gn_node.location = (100, 100)
        opacity_attr_gn_node.data_type = 'FLOAT'
        opacity_attr_gn_node.inputs["Name"].default_value = "opacity"

        threshold_value_node = geo_tree.nodes.new('ShaderNodeValue')
        threshold_value_node.location = (250, 100)

        threshold_node = geo_tree.nodes.new('ShaderNodeMath')
        threshold_node.location = (200, 100)
        threshold_node.operation = 'GREATER_THAN'

        join_selection_node = geo_tree.nodes.new('FunctionNodeBooleanMath')
        join_selection_node.location = (300, 100)
        join_selection_node.operation = 'AND'

        random_value_node = geo_tree.nodes.new('FunctionNodeRandomValue')
        random_value_node.location = (0, 400)
        random_value_node.data_type = 'BOOLEAN'
        if "Probability" in random_value_node.inputs:
            random_value_node.inputs["Probability"].default_value = min(RECOMMENDED_MAX_GAUSSIANS / N, 1)
        else:
            print("Error: 'Probability' input not found on 'FunctionNodeRandomValue'")

        maximum_node = geo_tree.nodes.new('ShaderNodeMath')
        maximum_node.location = (0, 400)
        maximum_node.operation = 'MAXIMUM'

        is_point_cloud_node = geo_tree.nodes.new('FunctionNodeInputBool')
        is_point_cloud_node.location = (0, 600)
        is_point_cloud_node.boolean = True

        ico_node = geo_tree.nodes.new('GeometryNodeMeshIcoSphere')
        ico_node.location = (200, 200)
        ico_node.inputs["Subdivisions"].default_value = 1
        ico_node.inputs["Radius"].default_value = 1

        set_shade_smooth_node = geo_tree.nodes.new('GeometryNodeSetShadeSmooth')
        set_shade_smooth_node.location = (200, 200)

        instance_node = geo_tree.nodes.new('GeometryNodeInstanceOnPoints')
        instance_node.location = (400, 0)

        switch_node = geo_tree.nodes.new('GeometryNodeSwitch')
        switch_node.location = (600, 0)
        switch_node.input_type = 'GEOMETRY'

        set_material_node = geo_tree.nodes.new('GeometryNodeSetMaterial')
        set_material_node.location = (800, 0)
        set_material_node.inputs["Material"].default_value = mat

        group_output_node = geo_tree.nodes.new('NodeGroupOutput')
        group_output_node.location = (1200, 0)

        set_point_radius_node = geo_tree.nodes.new('GeometryNodeSetPointRadius')
        set_point_radius_node.location = (200, 400)

        realize_instances_node = geo_tree.nodes.new('GeometryNodeRealizeInstances')
        realize_instances_node.location = (1000, 0)

        geo_tree.links.new(
            group_input_node.outputs["Geometry"],
            mesh_to_points_node.inputs["Mesh"]
        )

        geo_tree.links.new(
            is_point_cloud_node.outputs["Boolean"],
            maximum_node.inputs[0]
        )

        geo_tree.links.new(
            random_value_node.outputs[3],
            maximum_node.inputs[1]
        )

        geo_tree.links.new(
            opacity_attr_gn_node.outputs["Attribute"],
            threshold_node.inputs[0]
        )

        geo_tree.links.new(
            threshold_value_node.outputs[0],
            threshold_node.inputs[1]
        )

        geo_tree.links.new(
            threshold_node.outputs[0],
            join_selection_node.inputs[0]
        )

        geo_tree.links.new(
            maximum_node.outputs["Value"],
            join_selection_node.inputs[1]
        )

        geo_tree.links.new(
            join_selection_node.outputs[0],
            mesh_to_points_node.inputs["Selection"]
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
            is_point_cloud_node.outputs["Boolean"],
            switch_node.inputs[1]
        )

        geo_tree.links.new(
            instance_node.outputs["Instances"],
            switch_node.inputs[14]
        )

        geo_tree.links.new(
            mesh_to_points_node.outputs["Points"],
            set_point_radius_node.inputs["Points"]
        )

        geo_tree.links.new(
            set_point_radius_node.outputs["Points"],
            switch_node.inputs[15]
        )

        geo_tree.links.new(
            switch_node.outputs[6],
            set_material_node.inputs["Geometry"]
        )

        geo_tree.links.new(
            mesh_to_points_node.outputs["Points"],
            instance_node.inputs["Points"]
        )

        geo_tree.links.new(
            set_material_node.outputs["Geometry"],
            realize_instances_node.inputs["Geometry"]
        )

        geo_tree.links.new(
            realize_instances_node.outputs["Geometry"],
            group_output_node.inputs["Geometry"]
        )

        # Scale
        scale_attr = geo_tree.nodes.new('GeometryNodeInputNamedAttribute')
        scale_attr.location = (0, 200)
        scale_attr.data_type = 'FLOAT_VECTOR'
        scale_attr.inputs["Name"].default_value = "scale"

        scale_node = geo_tree.nodes.new('ShaderNodeVectorMath')
        scale_node.operation = 'SCALE'
        scale_node.location = (0, 200)
        scale_node.inputs["Scale"].default_value = 2

        avg_node = geo_tree.nodes.new('ShaderNodeVectorMath')
        avg_node.operation = 'DOT_PRODUCT'
        avg_node.inputs[1].default_value = (1 / 3, 1 / 3, 1 / 3)

        geo_tree.links.new(
            scale_attr.outputs["Attribute"],
            scale_node.inputs[0]
        )

        geo_tree.links.new(
            scale_node.outputs["Vector"],
            instance_node.inputs["Scale"]
        )

        geo_tree.links.new(
            scale_node.outputs["Vector"],
            avg_node.inputs[0]
        )

        geo_tree.links.new(
            avg_node.outputs["Value"],
            set_point_radius_node.inputs["Radius"]
        )

        # Rotation
        rot_euler_attr = geo_tree.nodes.new('GeometryNodeInputNamedAttribute')
        rot_euler_attr.location = (0, 400)
        rot_euler_attr.data_type = 'FLOAT_VECTOR'
        rot_euler_attr.inputs["Name"].default_value = "rot_euler"

        geo_tree.links.new(
            rot_euler_attr.outputs["Attribute"],
            instance_node.inputs["Rotation"]
        )

        print("Geometry nodes created in", time.time() - start_time, "seconds")

        print("Total Processing time: ", time.time() - start_time_0)

        return {'FINISHED'}

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = bpy.path.abspath("//point_cloud.ply")

        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


def construct_list_of_attributes():
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(3):
        l.append('f_dc_{}'.format(i))
    for i in range(45):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(3):
        l.append('scale_{}'.format(i))
    for i in range(4):
        l.append('rot_{}'.format(i))
    return l


class ExportGaussianSplatting(bpy.types.Operator):
    bl_idname = "object.export_gaussian_splatting"
    bl_label = "Export 3D Gaussian Splatting"
    bl_description = "Export a 3D Gaussian Splatting to file"

    filepath: bpy.props.StringProperty(
        name="File Path",
        description="Path to the Gaussian Splatting file",
        subtype="FILE_PATH"
    )

    def execute(self, context):
        if not self.filepath.lower().endswith('.ply'):
            self.filepath += ".ply"

        obj = context.active_object

        if obj is None or "gaussian_splatting" not in obj:
            self.report({'WARNING'}, "No Gaussian Splatting selected")
            return {'CANCELLED'}

        mesh: bpy.types.Mesh = obj.data

        N = len(mesh.vertices)

        xyz = np.zeros((N, 3))
        normals = np.zeros((N, 3))
        f_dc = np.zeros((N, 3))
        f_rest = np.zeros((N, 45))
        opacities = np.zeros((N, 1))
        scale = np.zeros((N, 3))
        rotation = np.zeros((N, 4))

        position_attr = mesh.attributes.get("position")
        log_opacity_attr = mesh.attributes.get("log_opacity")
        logscale_attr = mesh.attributes.get("logscale")
        sh0_attr = mesh.attributes.get("sh0")
        sh_attrs = [mesh.attributes.get(f"sh{j + 1}") for j in range(15)]
        rot_quatxyz_attr = mesh.attributes.get("quatxyz")
        rot_quatw_attr = mesh.attributes.get("quatw")

        for i, _ in enumerate(mesh.vertices):
            xyz[i] = position_attr.data[i].vector.to_tuple()
            opacities[i] = log_opacity_attr.data[i].value
            scale[i] = logscale_attr.data[i].vector.to_tuple()

            f_dc[i] = sh0_attr.data[i].vector.to_tuple()
            for j in range(15):
                f_rest[i, j:j + 45:15] = sh_attrs[j].data[i].vector.to_tuple()

            rotxyz_quat = rot_quatxyz_attr.data[i].vector.to_tuple()
            rotw_quat = rot_quatw_attr.data[i].value
            rotation[i] = (*rotxyz_quat, rotw_quat)

            # euler = mathutils.Euler(rot_euler_attr.data[i].vector)
            # quat = euler.to_quaternion()
            # rotation[i] = (quat.x, quat.y, quat.z, quat.w)

        # opacities = np.log(opacities / (1 - opacities))
        # scale = np.log(scale)

        dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]

        elements = np.empty(N, dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(self.filepath)

        return {'FINISHED'}

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = bpy.path.abspath("//point_cloud.ply")

        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class GaussianSplattingPanel(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    bl_idname = "OBJECT_PT_gaussian_splatting"
    bl_category = "3D Gaussian Splatting"
    bl_label = "3D Gaussian Splatting"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object

        # Import Gaussian Splatting button
        row = layout.row()
        row.operator(ImportGaussianSplatting.bl_idname, text="Import Gaussian Splatting")

        if obj is not None and "gaussian_splatting" in obj:

            # Display Options
            row = layout.row()
            row.prop(obj.modifiers["Geometry Nodes"].node_group.nodes.get("Boolean"), "boolean",
                     text="As point cloud (faster)")

            row = layout.row()
            col1 = row.column()
            col2 = row.column()
            col1.prop(bpy.data.node_groups["GaussianSplatting"].nodes["Named Attribute"].inputs[0],
                      "default_value", text="Attribute"
                      )
            col2.prop(obj.modifiers["Geometry Nodes"].node_group.nodes.get("Value").outputs[0], "default_value",
                      text="Threshold")

            if not obj.modifiers["Geometry Nodes"].node_group.nodes.get("Boolean").boolean:
                row = layout.row()
                row.prop(obj.modifiers["Geometry Nodes"].node_group.nodes.get("Random Value").inputs["Probability"],
                         "default_value", text="Display Percentage")

            # Select active
            row = layout.row()
            row.operator(SelectActiveSplats.bl_idname, text="Select Active Splats")
            row.enabled = context.mode == "EDIT_MESH"

            # Export Gaussian Splatting button
            row = layout.row()
            row.operator(ExportGaussianSplatting.bl_idname, text="Export Gaussian Splatting")


class SelectActiveSplats(bpy.types.Operator):
    bl_idname = "object.select_active_splats"
    bl_label = "Select Active Splats"
    bl_description = "Select filtered splats in edit mode"

    def execute(self, context):
        obj = context.active_object

        if context.mode != "EDIT_MESH":
            self.report({"WARNING"}, "Edit mode operator.")
            return {"CANCELLED"}

        bm = bmesh.from_edit_mesh(obj.data)
        attr_name = bpy.data.node_groups["GaussianSplatting"].nodes["Named Attribute"].inputs[0].default_value
        attr_thresh = bpy.data.node_groups["GaussianSplatting"].nodes["Value"].outputs[0].default_value

        attr_id = bm.verts.layers.float.get(attr_name)
        if attr_id is None:
            self.report({"WARNING"},
                        "Attribute not found or not supported. Only float attributes are currently supported.")
            return {'CANCELLED'}

        for v in bm.verts:
            v.select_set(v[attr_id] >= attr_thresh)
        bm.select_flush_mode()
        bmesh.update_edit_mesh(obj.data)
        return {'FINISHED'}


def register():
    bpy.utils.register_class(ImportGaussianSplatting)
    bpy.utils.register_class(GaussianSplattingPanel)
    bpy.utils.register_class(SelectActiveSplats)
    bpy.utils.register_class(ExportGaussianSplatting)

    bpy.types.Scene.ply_file_path = bpy.props.StringProperty(name="PLY Filepath", subtype='FILE_PATH')


def unregister():
    bpy.utils.unregister_class(ImportGaussianSplatting)
    bpy.utils.unregister_class(GaussianSplattingPanel)
    bpy.utils.unregister_class(SelectActiveSplats)
    bpy.utils.unregister_class(ExportGaussianSplatting)

    del bpy.types.Scene.ply_file_path


if __name__ == "__main__":
    register()
