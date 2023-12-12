# render each camera of a 3d reconstruction from blender camera objects

# export EXE=blender
# export EXE=/media/hayko/DATA/code/third/unknown/blender-2.80-linux-glibc217-x86_64/blender
# $EXE -b -P blender_batch_render.py -- <config>
#
###############################################################################
# ISSUES
# DONE: flip camera towards the object
# TODO: camera principle point offset
# TODO: crop smaller parts out of the city meshes

###############################################################################
# IMPORT

if 1:
    import numpy
    import json
    import os
if 1:
    import bpy
    from mathutils import Vector
    from mathutils import Matrix
    import sys
    from math import pi
    import getopt
    from pathlib import Path

    # TODO: remove and instally locally within blender
    if 1:
        # help blender find the installed packages
        sys.path.append(os.path.abspath('../'))
        sys.path.append('/home/hayko/miniconda3/envs/tf1.11/lib/python3.6/site-packages/')

import loggy
import configy

###############################################################################
# CONFIG
flag_resolution_percentage = 100

agisoft2blender_rotation = Matrix(
        ((-1, 0, 0,0),
        (0, 1, 0,0),
        (0, 0, -1,0),
        (0, 0, 0,1))
        )

logger = loggy.logger
config = None

###############################################################################
# FUNCTIONS

def setup(configfile):
    global config
    global logger

    config = configy.Config()

    if os.path.exists(configfile):
        config.import_jsonfile(configfile)
        logger = loggy.setup(config.get_logfile()+'render')
    else:
        logger.error ("config not found: " + str(configfile))
        exit(-1)

def blender_print_objects():
    # print objects in this scene
    logger.info('blender_debug_list_objects()')
    for ob in bpy.data.objects:
        print(ob.type, ob.name)
        try:
            for mat in ob.data.materials:
                print ('\t' + mat.name)
        except:
            pass


def blender_setup_environment():

    logger.info('blender_setup_environment()')
    bpy.ops.object.select_all(action='DESELECT')
    scn = bpy.context.scene
    scn.render.alpha_mode = 'TRANSPARENT'
    bpy.data.worlds["World"].light_settings.use_ambient_occlusion = True
    scn.render.image_settings.file_format = 'PNG'
    scn.render.image_settings.quality = 100

    bpy.context.scene.render.tile_x = 2048
    bpy.context.scene.render.tile_y = 2048 #   Time: 01:57.73 (Saving: 00:01.20)

    bpy.context.scene.render.tile_x = 1024
    bpy.context.scene.render.tile_y = 1024 #   Time: 00:57.38 (Saving: 00:01.27)

    # CYCLES NEEDS FIX OF TEXTURE
    #bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    #bpy.context.scene.cycles.device = 'CPU'

    for ob in bpy.data.objects:
        if ob.type == 'CAMERA':
            #print(ob.type, ob.name, 'clipping')
            ob.data.clip_end = 1000000
            ob.data.clip_start = 0.01
            ob.select = False

    # recover original resolution
    bpy.context.scene.render.resolution_x = config.resolution_x
    bpy.context.scene.render.resolution_y = config.resolution_y
    bpy.context.scene.render.resolution_percentage = flag_resolution_percentage


def blender_clean_cameras_lights():
    # delete default camera and lamps for make clean lighting

    logger.info('blender_clean_cameras_lights()')

    # TODO: use bpy versioning to setup
    # 2.79 blender
    bpy.ops.object.select_by_type(extend=False, type='LAMP')
    # 2.8 blender
    #bpy.ops.object.select_by_type(extend=False, type='LIGHT')
    bpy.ops.object.delete(use_global=False)

    #bpy.ops.object.select_by_type(extend=False, type='CAMERA')
    #bpy.ops.object.delete(use_global=False)

    for ob in bpy.data.objects:
        if ob.type == 'CAMERA':
         if ob.name == 'Camera':
             ob.select = True
             bpy.ops.object.delete(use_global=False)

    bpy.ops.object.select_all(action='DESELECT')


def blender_render_loop_cameras(name,outpath):
    # loop over all cameras and render into file
    file1 = open("sullens_cameras.txt","a")
    for ob in bpy.data.objects:
        if ob.type == 'CAMERA':

            camera_name = ob.name
            render_filename = name+ '_cam_' +  camera_name + '.png'
            bpy.context.scene.render.filepath = outpath+render_filename
            file1.write(camera_name + ' ')
            file1.write(''.join(ob.matrix_world))
            file1.write('\n')

            # transform
            ob.matrix_world *= agisoft2blender_rotation

            # set active
            bpy.context.scene.camera = ob

            if not os.path.exists(bpy.context.scene.render.filepath):
                logger.info(render_filename)
                #bpy.ops.render.render(write_still=True)
            else:
                logger.info('file already rendered... next...')

    if 1:
        render_filename = name+ '_cam_' +  camera_name + '.blend'
        bpy.ops.wm.save_as_mainfile(filepath=outpath+render_filename)
        #break

        logger.info('...')


def blender_render_single_camera(name, outpath):

    if 0:
        for ob in bpy.data.objects:
            if ob.type == 'CAMERA':
               print(ob.type, ob.name, ob.data.lens, ob.data.shift_x)

    # pick random camera for debug
    bpy.context.scene.camera = bpy.data.objects[2]
    camera_name = bpy.context.scene.camera.name

    if 0: # debug save as blender file format (warning huge)
        render_filename = name+ '_cam_' +  camera_name + '.blend'
        bpy.ops.wm.save_as_mainfile(filepath=outpath+render_filename)


    #print(bpy.context.scene.camera.matrix_world)
    bpy.context.scene.camera.matrix_world *= agisoft2blender_rotation
    #print(bpy.context.scene.camera.matrix_world)

    # setup filenames
    render_filename = name+ '_cam_' + camera_name + '.png'
    bpy.context.scene.render.filepath = outpath+render_filename
    bpy.ops.render.render(write_still=True)




def main():
    # LOAD MESH
    logger.info('loading 3d model data...')
    file_to_open = config.get_input()
    #bpy.ops.import_scene.fbx(filepath=file_to_open, axis_forward='-Z', axis_up='Y', use_anim=False)
    bpy.ops.wm.open_mainfile(filepath=file_to_open)

    outpath  = config.get_renderdir()

    if 1: blender_print_objects()

    if 1: blender_setup_environment()
    if 1: blender_clean_cameras_lights()

    if 0: blender_render_single_camera(config.name, outpath)

    if 1: blender_render_loop_cameras(config.name, outpath)

    logger.info('kapow - all done!')


###############################################################################
# MAIN ROUTINE (only when run, not when imported)
if __name__== "__main__":


    print ('Argument List: ' + str(sys.argv))
    sys.argv = sys.argv[sys.argv.index("--") + 1:]  # get all args after "--"
    print ('Argument List: ' + str(sys.argv))

    configfile = 'resources/trinity.json'
    configfile = 'resources/goblin.json'
    configfile = 'resources/sullens.json'

    if len(sys.argv)>0:
        configfile = str(sys.argv[0])

    setup(configfile)

    main()
