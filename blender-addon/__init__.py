bl_info = {
    "name": "Gaussian Splatting Importer",
    "author": "Alex Carlier",
    "version": (0, 0, 1),
    "blender": (3, 4, 0),
    "location": "3D Viewport > Sidebar > Gaussian Splatting",
    "description": "Import Gaussian Splatting scenes",
}

import bpy
import mathutils
import numpy as np
import time
import random

from .plyfile import PlyData, PlyElement



class ImportGaussianSplatting(bpy.types.Operator):
    bl_idname = "object.import_gaussian_splatting"
    bl_label = "Import 3D Gaussian Splatting"
    bl_description = "Import a 3D Gaussian Splatting file into the scene"
    bl_options = {"REGISTER", "UNDO"}
    
    filepath: bpy.props.StringProperty(
        name="File Path",
        description="Path to the Gaussian Splatting file",
        default="",
        subtype='FILE_PATH',
    )

    def execute(self, context):
        if not self.filepath:
            self.report({'WARNING'}, "Filepath is not set!")
            return {'CANCELLED'}

        start_time_0 = time.time()
        
        bpy.context.scene.render.engine = 'CYCLES'

        # bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'OPTIX'

        if context.preferences.addons["cycles"].preferences.has_active_device():
            bpy.context.scene.cycles.device = 'GPU'

        bpy.context.scene.cycles.transparent_max_bounces = 20

        RECOMMENDED_MAX_GAUSSIANS = 200_000

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
        
        log_opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        opacities = 1 / (1 + np.exp(-log_opacities))

        features_dc = np.zeros((N, 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        
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
        # Load PLY
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
            sh_attr = mesh.attributes.new(name=f"sh{j+1}", type='FLOAT_VECTOR', domain='POINT')
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

        mat_tree = mat.node_tree

        for node in mat_tree.nodes:
            mat_tree.nodes.remove(node)
        
        sh_attr_nodes = []
        sh_inst_attr_nodes = []  # ellipsoids
        sh_geom_attr_nodes = []  # point cloud

        for j in range(0, 16):
            sh_inst_attr_node = mat_tree.nodes.new('ShaderNodeAttribute')
            sh_inst_attr_node.location = (0, 0)
            sh_inst_attr_node.attribute_name = f"sh{j}"
            sh_inst_attr_node.attribute_type = 'GEOMETRY' # 'INSTANCER'
            sh_inst_attr_nodes.append(sh_inst_attr_node)
            
            sh_geom_attr_node = mat_tree.nodes.new('ShaderNodeAttribute')
            sh_geom_attr_node.location = (0, 0)
            sh_geom_attr_node.attribute_name = f"sh{j}"
            sh_geom_attr_node.attribute_type = 'GEOMETRY' # 'INSTANCER'
            sh_geom_attr_nodes.append(sh_geom_attr_node)

            vector_math_node = mat_tree.nodes.new('ShaderNodeVectorMath')
            vector_math_node.operation = 'ADD'

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

        opacity_attr_node = mat_tree.nodes.new('ShaderNodeAttribute')
        opacity_attr_node.location = (0, -200)
        opacity_attr_node.attribute_name = "opacity"
        opacity_attr_node.attribute_type = 'GEOMETRY'

        principled_node = mat_tree.nodes.new('ShaderNodeBsdfPrincipled')
        principled_node.location = (200, 0)
        principled_node.inputs["Base Color"].default_value = (0, 0, 0, 1)
        principled_node.inputs["Specular"].default_value = 0
        principled_node.inputs["Roughness"].default_value = 0

        output_node = mat_tree.nodes.new('ShaderNodeOutputMaterial')
        output_node.location = (400, 0)

        # Camera location
        combine_xyz_node = mat_tree.nodes.new('ShaderNodeCombineXYZ')
        vector_transform_node = mat_tree.nodes.new('ShaderNodeVectorTransform')
        vector_transform_node.vector_type = 'POINT'
        vector_transform_node.convert_from = 'CAMERA'
        vector_transform_node.convert_to = 'WORLD'

        mat_tree.links.new(
            combine_xyz_node.outputs["Vector"],
            vector_transform_node.inputs["Vector"]
        )

        # View direction

        dir_node = mat_tree.nodes.new('ShaderNodeVectorMath')
        dir_node.operation = 'SUBTRACT'

        normalize_node = mat_tree.nodes.new('ShaderNodeVectorMath')
        normalize_node.operation = 'NORMALIZE'

        mat_tree.links.new(
            position_attr_node.outputs["Vector"],
            dir_node.inputs[0]
        )

        mat_tree.links.new(
            vector_transform_node.outputs["Vector"],
            dir_node.inputs[1]
        )

        mat_tree.links.new(
            dir_node.outputs["Vector"],
            normalize_node.inputs[0]
        )

        # Coordinate system transform (x -> -y, y -> -z, z -> x)  TODO: REMOVE

        separate_xyz_node = mat_tree.nodes.new('ShaderNodeSeparateXYZ')
        combine_xyz_node = mat_tree.nodes.new('ShaderNodeCombineXYZ')

        multiply_node = mat_tree.nodes.new('ShaderNodeVectorMath')
        multiply_node.operation = 'MULTIPLY'
        multiply_node.inputs[1].default_value = (-1, -1, 1)

        mat_tree.links.new(
            normalize_node.outputs["Vector"],
            separate_xyz_node.inputs["Vector"]
        )

        mat_tree.links.new(
            separate_xyz_node.outputs["X"],
            combine_xyz_node.inputs["Y"]
        )

        mat_tree.links.new(
            separate_xyz_node.outputs["Y"],
            combine_xyz_node.inputs["Z"]
        )

        mat_tree.links.new(
            separate_xyz_node.outputs["Z"],
            combine_xyz_node.inputs["X"]
        )

        mat_tree.links.new(
            combine_xyz_node.outputs["Vector"],
            multiply_node.inputs[0]
        )


        # SH Coefficients

        C0 = 0.28209479177387814
        C1 = 0.4886025119029199
        C2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396
        ]
        C3 = [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435
        ]

        separate_xyz_node = mat_tree.nodes.new('ShaderNodeSeparateXYZ')

        mat_tree.links.new(
            multiply_node.outputs["Vector"],
            separate_xyz_node.inputs["Vector"]
        )

        x = separate_xyz_node.outputs["X"]
        y = separate_xyz_node.outputs["Y"]
        z = separate_xyz_node.outputs["Z"]

        xx_node = mat_tree.nodes.new('ShaderNodeMath')
        xx_node.operation = 'MULTIPLY'
        mat_tree.links.new(x, xx_node.inputs[0])
        mat_tree.links.new(x, xx_node.inputs[1])
        xx = xx_node.outputs["Value"]

        yy_node = mat_tree.nodes.new('ShaderNodeMath')
        yy_node.operation = 'MULTIPLY'
        mat_tree.links.new(y, yy_node.inputs[0])
        mat_tree.links.new(y, yy_node.inputs[1])
        yy = yy_node.outputs["Value"]

        zz_node = mat_tree.nodes.new('ShaderNodeMath')
        zz_node.operation = 'MULTIPLY'
        mat_tree.links.new(z, zz_node.inputs[0])
        mat_tree.links.new(z, zz_node.inputs[1])
        zz = zz_node.outputs["Value"]

        xy_node = mat_tree.nodes.new('ShaderNodeMath')
        xy_node.operation = 'MULTIPLY'
        mat_tree.links.new(x, xy_node.inputs[0])
        mat_tree.links.new(y, xy_node.inputs[1])
        xy = xy_node.outputs["Value"]

        yz_node = mat_tree.nodes.new('ShaderNodeMath')
        yz_node.operation = 'MULTIPLY'
        mat_tree.links.new(x, yz_node.inputs[0])
        mat_tree.links.new(y, yz_node.inputs[1])
        yz = yz_node.outputs["Value"]

        xz_node = mat_tree.nodes.new('ShaderNodeMath')
        xz_node.operation = 'MULTIPLY'
        mat_tree.links.new(x, xz_node.inputs[0])
        mat_tree.links.new(y, xz_node.inputs[1])
        xz = xz_node.outputs["Value"]


        # SH 0

        scale_node_0 = mat_tree.nodes.new('ShaderNodeVectorMath')
        scale_node_0.operation = 'SCALE'
        scale_node_0.inputs["Scale"].default_value = C0

        mat_tree.links.new(
            sh[0],
            scale_node_0.inputs[0]
        )

        # SH 1

        math_node = mat_tree.nodes.new('ShaderNodeMath')
        math_node.operation = 'MULTIPLY'
        math_node.inputs[1].default_value = -C1

        mat_tree.links.new(
            y,
            math_node.inputs[0]
        )

        scale_node_1 = mat_tree.nodes.new('ShaderNodeVectorMath')
        scale_node_1.operation = 'SCALE'

        mat_tree.links.new(
            sh[1],
            scale_node_1.inputs[0]
        )

        mat_tree.links.new(
            math_node.outputs["Value"],
            scale_node_1.inputs["Scale"]
        )

        # SH 2

        math_node = mat_tree.nodes.new('ShaderNodeMath')
        math_node.operation = 'MULTIPLY'
        math_node.inputs[1].default_value = C1

        mat_tree.links.new(
            z,
            math_node.inputs[0]
        )

        scale_node_2 = mat_tree.nodes.new('ShaderNodeVectorMath')
        scale_node_2.operation = 'SCALE'

        mat_tree.links.new(
            sh[2],
            scale_node_2.inputs[0]
        )

        mat_tree.links.new(
            math_node.outputs["Value"],
            scale_node_2.inputs["Scale"]
        )


        # SH 3

        math_node = mat_tree.nodes.new('ShaderNodeMath')
        math_node.operation = 'MULTIPLY'
        math_node.inputs[1].default_value = -C1

        mat_tree.links.new(
            x,
            math_node.inputs[0]
        )

        scale_node_3 = mat_tree.nodes.new('ShaderNodeVectorMath')
        scale_node_3.operation = 'SCALE'

        mat_tree.links.new(
            sh[3],
            scale_node_3.inputs[0]
        )

        mat_tree.links.new(
            math_node.outputs["Value"],
            scale_node_3.inputs["Scale"]
        )

        # SH 4

        math_node = mat_tree.nodes.new('ShaderNodeMath')
        math_node.operation = 'MULTIPLY'
        math_node.inputs[1].default_value = C2[0]

        mat_tree.links.new(
            xy,
            math_node.inputs[0]
        )

        scale_node_4 = mat_tree.nodes.new('ShaderNodeVectorMath')
        scale_node_4.operation = 'SCALE'

        mat_tree.links.new(
            sh[4],
            scale_node_4.inputs[0]
        )

        mat_tree.links.new(
            math_node.outputs["Value"],
            scale_node_4.inputs["Scale"]
        )

        # SH 5

        math_node = mat_tree.nodes.new('ShaderNodeMath')
        math_node.operation = 'MULTIPLY'
        math_node.inputs[1].default_value = C2[1]

        mat_tree.links.new(
            yz,
            math_node.inputs[0]
        )

        scale_node_5 = mat_tree.nodes.new('ShaderNodeVectorMath')
        scale_node_5.operation = 'SCALE'

        mat_tree.links.new(
            sh[5],
            scale_node_5.inputs[0]
        )

        mat_tree.links.new(
            math_node.outputs["Value"],
            scale_node_5.inputs["Scale"]
        )

        # SH 6

        math_node1 = mat_tree.nodes.new('ShaderNodeMath')
        math_node1.operation = 'MULTIPLY'
        math_node1.inputs[1].default_value = C2[2]

        mat_tree.links.new(
            zz,
            math_node1.inputs[0]
        )

        math_node2 = mat_tree.nodes.new('ShaderNodeMath')
        math_node2.operation = 'SUBTRACT'

        mat_tree.links.new(
            math_node1.outputs["Value"],
            math_node2.inputs[0]
        )

        mat_tree.links.new(
            xx,
            math_node2.inputs[1]
        )

        math_node3 = mat_tree.nodes.new('ShaderNodeMath')
        math_node3.operation = 'SUBTRACT'

        mat_tree.links.new(
            math_node2.outputs["Value"],
            math_node3.inputs[0]
        )

        mat_tree.links.new(
            yy,
            math_node3.inputs[1]
        )

        math_node4 = mat_tree.nodes.new('ShaderNodeMath')
        math_node4.operation = 'MULTIPLY'
        math_node4.inputs[1].default_value = C2[1]

        mat_tree.links.new(
            math_node3.outputs["Value"],
            math_node4.inputs[0]
        )

        scale_node_6 = mat_tree.nodes.new('ShaderNodeVectorMath')
        scale_node_6.operation = 'SCALE'

        mat_tree.links.new(
            sh[6],
            scale_node_6.inputs[0]
        )

        mat_tree.links.new(
            math_node4.outputs["Value"],
            scale_node_6.inputs["Scale"]
        )

        # SH 7

        math_node = mat_tree.nodes.new('ShaderNodeMath')
        math_node.operation = 'MULTIPLY'
        math_node.inputs[1].default_value = C2[3]

        mat_tree.links.new(
            xz,
            math_node.inputs[0]
        )

        scale_node_7 = mat_tree.nodes.new('ShaderNodeVectorMath')
        scale_node_7.operation = 'SCALE'

        mat_tree.links.new(
            sh[7],
            scale_node_7.inputs[0]
        )

        mat_tree.links.new(
            math_node.outputs["Value"],
            scale_node_7.inputs["Scale"]
        )

        # SH 8

        math_node1 = mat_tree.nodes.new('ShaderNodeMath')
        math_node1.operation = 'SUBTRACT'

        mat_tree.links.new(
            xx,
            math_node1.inputs[0]
        )

        mat_tree.links.new(
            yy,
            math_node1.inputs[1]
        )

        math_node2 = mat_tree.nodes.new('ShaderNodeMath')
        math_node2.operation = 'MULTIPLY'
        math_node2.inputs[1].default_value = C2[4]

        mat_tree.links.new(
            math_node1.outputs["Value"],
            math_node2.inputs[0]
        )

        scale_node_8 = mat_tree.nodes.new('ShaderNodeVectorMath')
        scale_node_8.operation = 'SCALE'

        mat_tree.links.new(
            sh[8],
            scale_node_8.inputs[0]
        )

        mat_tree.links.new(
            math_node2.outputs["Value"],
            scale_node_8.inputs["Scale"]
        )

        # SH 9

        scale_node_9 = mat_tree.nodes.new('ShaderNodeVectorMath')
        scale_node_9.operation = 'SCALE'

        # SH 10

        scale_node_10 = mat_tree.nodes.new('ShaderNodeVectorMath')
        scale_node_10.operation = 'SCALE'

        # SH 11

        scale_node_11 = mat_tree.nodes.new('ShaderNodeVectorMath')
        scale_node_11.operation = 'SCALE'

        # SH 12

        scale_node_12 = mat_tree.nodes.new('ShaderNodeVectorMath')
        scale_node_12.operation = 'SCALE'

        # SH 13

        scale_node_13 = mat_tree.nodes.new('ShaderNodeVectorMath')
        scale_node_13.operation = 'SCALE'

        # SH 14

        scale_node_14 = mat_tree.nodes.new('ShaderNodeVectorMath')
        scale_node_14.operation = 'SCALE'

        # SH 15

        scale_node_15 = mat_tree.nodes.new('ShaderNodeVectorMath')
        scale_node_15.operation = 'SCALE'


        # Result

        res_nodes = [
            scale_node_0, scale_node_1, scale_node_2, scale_node_3, scale_node_4, scale_node_5, scale_node_6, scale_node_7,
            scale_node_8, scale_node_9, scale_node_10, scale_node_11, scale_node_12, scale_node_13, scale_node_14, scale_node_15
        ]

        add_node = mat_tree.nodes.new('ShaderNodeVectorMath')
        add_node.operation = 'ADD'

        mat_tree.links.new(
            res_nodes[0].outputs["Vector"],
            add_node.inputs[0]
        )

        mat_tree.links.new(
            res_nodes[1].outputs["Vector"],
            add_node.inputs[1]
        )

        for i in range(2, 16):
            new_add_node = mat_tree.nodes.new('ShaderNodeVectorMath')
            new_add_node.operation = 'ADD'

            mat_tree.links.new(
                res_nodes[i].outputs["Vector"],
                new_add_node.inputs[0]
            )

            mat_tree.links.new(
                add_node.outputs["Vector"],
                new_add_node.inputs[1]
            )

            add_node = new_add_node

        math_node = mat_tree.nodes.new('ShaderNodeVectorMath')
        math_node.operation = 'ADD'
        math_node.inputs[1].default_value = (0.5, 0.5, 0.5)

        gamma_node = mat_tree.nodes.new('ShaderNodeGamma')
        gamma_node.inputs["Gamma"].default_value = 2.2


        mat_tree.links.new(
            add_node.outputs["Vector"],
            math_node.inputs[0]
        )

        mat_tree.links.new(
            math_node.outputs["Vector"],
            gamma_node.inputs["Color"]
        )

        mat_tree.links.new(
            gamma_node.outputs["Color"],
            principled_node.inputs["Emission"]
        )


        geometry_node = mat_tree.nodes.new('ShaderNodeNewGeometry')

        vector_math_node = mat_tree.nodes.new('ShaderNodeVectorMath')
        vector_math_node.operation = 'DOT_PRODUCT'

        mat_tree.links.new(
            geometry_node.outputs["Incoming"],
            vector_math_node.inputs[0]
        )

        mat_tree.links.new(
            geometry_node.outputs["Normal"],
            vector_math_node.inputs[1]
        )

        math_node = mat_tree.nodes.new('ShaderNodeMath')
        math_node.operation = 'MULTIPLY'

        mat_tree.links.new(
            opacity_attr_node.outputs["Fac"],
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
        mesh_to_points_node.inputs["Radius"].default_value = 0.01

        random_value_node = geo_tree.nodes.new('FunctionNodeRandomValue')
        random_value_node.location = (0, 400)
        random_value_node.inputs["Probability"].default_value = min(RECOMMENDED_MAX_GAUSSIANS / N, 1)
        random_value_node.data_type = 'BOOLEAN'

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
        group_output_node.location = (1000, 0)

        set_point_radius_node = geo_tree.nodes.new('GeometryNodeSetPointRadius')
        set_point_radius_node.location = (200, 400)


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
            maximum_node.outputs["Value"],
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
        avg_node.inputs[1].default_value = (1/3, 1/3, 1/3)

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
        default="point_cloud.ply",
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

        print("N", N)
        
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
        sh_attrs = [mesh.attributes.get(f"sh{j+1}") for j in range(15)]
        rot_quatxyz_attr = mesh.attributes.get("quatxyz")
        rot_quatw_attr = mesh.attributes.get("quatw")

        for i, _ in enumerate(mesh.vertices):
            xyz[i] = position_attr.data[i].vector.to_tuple()
            opacities[i] = log_opacity_attr.data[i].value
            scale[i] = logscale_attr.data[i].vector.to_tuple()

            f_dc[i] = sh0_attr.data[i].vector.to_tuple()
            for j in range(15):
                f_rest[i, j:j+45:15] = sh_attrs[j].data[i].vector.to_tuple()
            
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
        # if not self.filepath:
        #     self.filepath = bpy.path.abspath("//point_cloud.ply")

        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class GaussianSplattingPanel(bpy.types.Panel):
    
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    bl_idname = "OBJECT_PT_gaussian_splatting"
    bl_category = "Gaussian Splatting"
    bl_label = "Gaussian Splatting"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object

        # Import Gaussian Splatting button
        row = layout.row()
        row.operator(ImportGaussianSplatting.bl_idname, text="Import Gaussian Splatting")

        # Display Options
        if obj is not None and "gaussian_splatting" in obj:
            
            row = layout.row()
            row.prop(obj.modifiers["GeometryNodes"].node_group.nodes.get("Boolean"), "boolean", text="As point cloud (faster)")

            if not obj.modifiers["GeometryNodes"].node_group.nodes.get("Boolean").boolean:
                row = layout.row()
                row.prop(obj.modifiers["GeometryNodes"].node_group.nodes.get("Random Value").inputs["Probability"], "default_value", text="Display Percentage")
        
        # Export Gaussian Splatting button
        row = layout.row()
        row.operator(ExportGaussianSplatting.bl_idname, text="Export Gaussian Splatting")

def register():
    bpy.utils.register_class(ImportGaussianSplatting)
    bpy.utils.register_class(GaussianSplattingPanel)
    bpy.utils.register_class(ExportGaussianSplatting)

    bpy.types.Scene.ply_file_path = bpy.props.StringProperty(name="PLY Filepath", subtype='FILE_PATH')

def unregister():
    bpy.utils.unregister_class(ImportGaussianSplatting)
    bpy.utils.unregister_class(GaussianSplattingPanel)
    bpy.utils.unregister_class(ExportGaussianSplatting)

    del bpy.types.Scene.ply_file_path

if __name__ == "__main__":
    register()
