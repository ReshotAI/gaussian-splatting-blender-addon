# 3D Gaussian Splatting Blender Addon

## Disclaimer
This add-on was developed mostly as an experimentation, it is slow on large scenes, and not fully accurate. It can be used as a tool to clean floaters in Gaussian Splatting captures.

In order to be faster in non-pointcloud mode, it would be needed to implement gaussians as camera-facing quads, as some implementations for other engines have done.

## Blender 4.0 and up
- Downlaod the repo
- unzip
- zip blender addon
- Edit<Preferences<Install
- choose zip file
- toggle addon
- have the sidebar enabled (shortcut n)
- now you can choose 3d gaussian splatting in the sidebar
- import gaussian splatting with a ply file

## Lower than blender 4.0
- go to this commit message and do the steps like above https://github.com/ReshotAI/gaussian-splatting-blender-addon/tree/41ca08a4ff593bd68b8d06bc56c9c00739bbaac1

This is an older version!!

The reason you can not use the newer version is that the blender4.0 changes are not backwards compatible and have many breaking api changes
