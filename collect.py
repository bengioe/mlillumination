
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


values = [[float(i) for i in j.split(":")] for j in "400:0.343, 404:0.445, 408:0.551, 412:0.624, 416:0.665, 420:0.687, 424:0.708, 428:0.723, 432:0.715, 436:0.71, 440:0.745, 444:0.758, 448:0.739, 452:0.767, 456:0.777, 460:0.765, 464:0.751, 468:0.745, 472:0.748, 476:0.729, 480:0.745, 484:0.757, 488:0.753, 492:0.75, 496:0.746, 500:0.747, 504:0.735, 508:0.732, 512:0.739, 516:0.734, 520:0.725, 524:0.721, 528:0.733, 532:0.725, 536:0.732, 540:0.743, 544:0.744, 548:0.748, 552:0.728, 556:0.716, 560:0.733, 564:0.726, 568:0.713, 572:0.74, 576:0.754, 580:0.764, 584:0.752, 588:0.736, 592:0.734, 596:0.741, 600:0.74, 604:0.732, 608:0.745, 612:0.755, 616:0.751, 620:0.744, 624:0.731, 628:0.733, 632:0.744, 636:0.731, 640:0.712, 644:0.708, 648:0.729, 652:0.73, 656:0.727, 660:0.707, 664:0.703, 668:0.729, 672:0.75, 676:0.76, 680:0.751, 684:0.739, 688:0.724, 692:0.73, 696:0.74, 700:0.737".split(", ")]

interp = InterpolatedSpectrum(len(values))
for l,v in values:
    interp.append(l,v)
interp.zeroExtend()
bsdf_box = Spectrum()
bsdf_box.fromContinuousSpectrum(interp);
bsdf_box.clampNegative()
print bsdf_box

class DefaultPlugins:

    integrator = 0
    
    @classmethod
    def getObjects(cls):
        for i in os.listdir("cbox/meshes"):
            yield pmgr.create({
                    'type':'obj',
                    'filename': 'cbox/meshes/'+i,
                    'bsdf': {
                        'type':'diffuse',
                        'reflectance': bsdf_box
                        }
                    })

def makeScene(lightpos):
    scene = Scene()
    r = numpy.random.random
    camera = pmgr.create({
            'type': 'perspective',
            'toWorld': Transform.lookAt(
                Point(528.246-r()*100, 398.667-r()*10, -759.992+r()*100),
                Point(250,250,250),
                Vector(0,1,0)),
            'film':{
                'type': 'ldrfilm',
                'width': 32,
                'height': 32},
            'sampler':{
                'type': 'ldsampler',
                'sampleCount': 1
                },
            'fov': 45.0
            })
    scene.addChild(camera)
    scene.addChild(pmgr.create({
            'type':'path_lr',
            'light':Point(lightpos[0],lightpos[1],lightpos[2])
            }))

    scene.addChild(pmgr.create({
                'type':'point',
                'position': Point(lightpos[0],lightpos[1],lightpos[2]),
                'intensity': Spectrum(40000)
                }))
    
    # for some reason I must recreate the objects each time... kind of
    # weird. I guess that maybe some work is done on them when added
    # to the scene that attaches them to it
    for o in DefaultPlugins.getObjects():
        scene.addChild(o)

    scene.configure()

    return scene

def main():
    for i in range(400):
        scene = makeScene([random.randint(0,400),
                           random.randint(0,400),
                           random.randint(-100,100)])
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
