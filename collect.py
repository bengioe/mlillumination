
import os, sys, random
import multiprocessing
import numpy

# warning: even though I add mitsuba to python's paths, you must still
# run this: `source /.../mitsuba/setpath.sh` for the imports to work
# (or maybe I'm doing something wrong)

mitsuba_path = "/home/bengioe/local/mitsuba/dist"

sys.path.append(mitsuba_path + "/python/2.7/")
os.environ["PATH"] += os.pathsep + mitsuba_path

import mitsuba
from mitsuba.core import *
from mitsuba.render import *

#unused?
fileResolver = Thread.getThread().getFileResolver()
fileResolver.appendPath("cbox");
paramMap = StringMap()
####

pmgr = PluginManager.getInstance()

#scene = SceneHandler.loadScene(fileResolver.resolve("cbox/cbox_nolight_nocam.xml"), paramMap)

def makeScene(lightpos):

    paramMap = StringMap()
    paramMap['myParameter'] = 'value'

    scene = SceneHandler.loadScene(fileResolver.resolve("cbox/cbox_nolight_nocam.xml"), paramMap)
    r = numpy.random.random
    camera = pmgr.create({
            'type': 'perspective',
            'toWorld': Transform.lookAt(
                Point(600-r()*5, 500-r()*1, -800+r()*4),
                Point(250,250,250),
                Vector(0,1,0)),
            'film':{
                'type': 'ldrfilm',
                'width': 64,
                'height': 64},
            'sampler':{
                'type': 'ldsampler',
                'sampleCount': 1
                },
            'fov': 45.0
            })
    scene.setSensor(camera)

    scene.addChild(pmgr.create({
                'type':'point',
                'position': Point(lightpos[0],lightpos[1],lightpos[2]),
                'intensity': Spectrum(40000)
                }))
    
    scene.configure()

    return scene

def main():
    for i in range(300):
        l = [random.randint(100,400),
             random.randint(100,400),
             random.randint(-100,0)]
        print i,l
        #l = [300,300,100]
        scene = makeScene(l)
        render(scene,"images/image"+str(i)+".png")

def render(scene,path="image.png"):


    scene.setDestinationFile(path)

    # Create a render job and insert it into the queue
    queue = RenderQueue()
    job = RenderJob('myRenderJob', scene, queue)
    job.start()
    job.join()
    #print(dir(job))
    # Wait for all jobs to finish and release resources
    queue.waitLeft(0)
    queue.join()
    # Print some statistics about the rendering process
    print(Statistics.getInstance().getStats())
    time.sleep(0.01)

import time

    
scheduler = Scheduler.getInstance()
for i in range(0, multiprocessing.cpu_count()):
    print i
    scheduler.registerWorker(LocalWorker(i,'wrk'+str(i)))

scheduler.start()

main()

def foo():

    scene = SceneHandler.loadScene(fileResolver.resolve("cbox/cbox3.xml"),paramMap)

    print(scene)
    scene.getSensor().setWorldTransform(Transform.lookAt(Point(528.246,398.667,-759.992),
                                                         Point(527.996, 398.541,-759.032),
                                                         Vector(0,1,0)))

    print(dir(scene.getFilm().getSize()))
    print(scene.getFilm().getSize())
