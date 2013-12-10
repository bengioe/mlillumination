/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2012 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "glwidget.h"
#include "preview.h"
#include <string.h> 
#include <mitsuba/core/timer.h>
#include <mitsuba/hw/gpuprogram.h>
#include <mitsuba/hw/gpugeometry.h>
#include <mitsuba/hw/gputexture.h>
#include "../shapes/instance.h"
#include <fstream>
#include <streambuf>
#include <chrono>

PreviewThread::PreviewThread(Device *parentDevice, Renderer *parentRenderer)
	: AbstractPreviewThread("prvw"), 
	  m_parentDevice(parentDevice), m_parentRenderer(parentRenderer),
		m_context(NULL), m_quit(false) {
	MTS_AUTORELEASE_BEGIN()
	m_session = Session::create();
	m_device = Device::create(m_session);
	m_renderer = Renderer::create(m_session);
	m_mutex = new Mutex();
	m_queueCV = new ConditionVariable(m_mutex);
	m_bufferCount = 3;
	m_backgroundScaleFactor = 1.0f;
	m_queueEntryIndex = 0;
	m_session->init();
	m_timer = new Timer();
	m_accumBuffer = NULL;
	m_sleep = false;
	m_started = new WaitFlag();
	m_shaderManager = new VPLShaderManager(m_renderer);

	m_accumProgram = m_renderer->createGPUProgram("Accumulation program");
	m_accumProgram->setSource(GPUProgram::EVertexProgram,
		"void main() {\n"
		"	gl_Position = ftransform();\n"
		"	gl_TexCoord[0]  = gl_MultiTexCoord0;\n"
		"}\n"
	);

	m_accumProgram->setSource(GPUProgram::EFragmentProgram,
		"uniform sampler2D source1, source2;\n"
		"void main() {\n"
		"	gl_FragColor = texture2D(source1, gl_TexCoord[0].xy) + \n"
		"	               texture2D(source2, gl_TexCoord[0].xy);\n"
		"}\n"
	);

	m_framebuffer = m_renderer->createGPUTexture("Framebuffer");
	for (int i=0; i<m_bufferCount; ++i)
		m_recycleQueue.push_back(PreviewQueueEntry(m_queueEntryIndex++));

	m_random = new Random();

	MTS_AUTORELEASE_END()
}

PreviewThread::~PreviewThread() {
	MTS_AUTORELEASE_BEGIN()
	m_session->shutdown();
	MTS_AUTORELEASE_END()
}

void PreviewThread::quit() {
	if (!isRunning())
		return;

	std::vector<PreviewQueueEntry> temp;
	temp.reserve(m_bufferCount);

	/* Steal all buffers */
	UniqueLock lock(m_mutex);
	while (true) {
		while (!m_readyQueue.empty()) {
			temp.push_back(m_readyQueue.back());
			m_readyQueue.pop_back();
		}

		while (!m_recycleQueue.empty()) {
			temp.push_back(m_recycleQueue.back());
			m_recycleQueue.pop_back();
		}

		if ((int) temp.size() == m_bufferCount)
			break;

		m_queueCV->wait();
	}

	/* Put the buffers back */
	for (size_t i=0; i<temp.size(); ++i)
		m_recycleQueue.push_back(temp[i]);
	m_quit = true;

	m_queueCV->signal();
	lock.unlock();

	/* Wait for the thread to terminate */
	if (isRunning())
		join();

	m_shaderManager->setScene(NULL);
	m_shaderManager->cleanup();

	while (!m_readyQueue.empty()) {
		m_readyQueue.back().cleanup();
		m_readyQueue.pop_back();
	}

	while (!m_recycleQueue.empty()) {
		m_recycleQueue.back().cleanup();
		m_recycleQueue.pop_back();
	}
}

void PreviewThread::setSceneContext(SceneContext *context, bool swapContext, bool motion) {
	if (!isRunning())
		return;

	std::vector<PreviewQueueEntry> temp;
	temp.reserve(m_bufferCount);

	m_sleep = true;
	LockGuard lock(m_mutex);

	/* Steal all buffers from the rendering
	   thread to make sure we get its attention :) */
	while (true) {
		while (!m_readyQueue.empty()) {
			temp.push_back(m_readyQueue.front());
				m_readyQueue.pop_front();
		}

		while (!m_recycleQueue.empty()) {
			temp.push_back(m_recycleQueue.back());
			m_recycleQueue.pop_back();
		}

		if ((int) temp.size() == m_bufferCount)
			break;

		m_queueCV->wait();
	}

	if (swapContext && m_context) {
		m_context->vpls = m_vpls;
		m_context->previewBuffer = temp[0];
		m_recycleQueue.push_back(PreviewQueueEntry(m_queueEntryIndex++));

		/* Put back all buffers */
		for (size_t i=1; i<temp.size(); ++i)
			m_recycleQueue.push_back(temp[i]);
	} else {
		for (size_t i=0; i<temp.size(); ++i)
			m_recycleQueue.push_back(temp[i]);
	}

	if (swapContext && context && context->previewBuffer.vplSampleOffset > 0) {
		/* Resume from a stored state */
		m_vplSampleOffset = context->previewBuffer.vplSampleOffset;
		m_vpls = context->vpls;
		m_accumBuffer = context->previewBuffer.buffer;

		/* Take ownership of the buffer */
		m_recycleQueue.push_back(context->previewBuffer);
		context->previewBuffer.buffer = NULL;
		context->previewBuffer.sync = NULL;
		context->previewBuffer.vplSampleOffset = 0;

		if (m_recycleQueue.size() > (size_t) m_bufferCount) {
			PreviewQueueEntry entry = m_recycleQueue.front();
			m_recycleQueue.pop_front();
			if (entry.buffer)
				entry.buffer->decRef();
			if (entry.sync)
				entry.sync->decRef();
		}
	} else {
		/* Reset the VPL rendering progress */
		m_vplSampleOffset = 0;
		m_vpls.clear();
		m_accumBuffer = NULL;
	}

	if (m_context != context)
		m_minVPLs = 0;

	m_vplsPerSecond = 0;
	m_raysPerSecond = 0;
	m_vplCount = 0;
	m_timer->reset();

	if (!context)
		m_shaderManager->setScene(NULL);

	m_context = context;

	if (m_context) {
		ProjectiveCamera *camera = static_cast<ProjectiveCamera *>
			(m_context->scene->getSensor());
		m_camTransform = camera->getWorldTransform();
	}

	if (motion && !m_motion) {
		emit statusMessage("");
		m_minVPLs = 0;
	}

	m_motion = motion;
	m_queueCV->signal();
	m_sleep = false;
}

void PreviewThread::resume() {
	m_queueCV->signal();
}

PreviewQueueEntry PreviewThread::acquireBuffer(int ms) {
	PreviewQueueEntry entry;

	UniqueLock lock(m_mutex);
	while (m_readyQueue.size() == 0) {
		if (m_quit)
			return entry;
		if (!m_queueCV->wait(ms)) {
			return entry;
		}
	}
	entry = m_readyQueue.front();
	m_readyQueue.pop_front();
	lock.unlock();

#if 0
	if (m_context->previewMethod == ERayTrace ||
		m_context->previewMethod == ERayTraceCoherent)
		entry.buffer->refresh();
	else
#endif
	if (m_useSync)
		entry.sync->enqueueWait();

	return entry;
}

void PreviewThread::releaseBuffer(PreviewQueueEntry &entry) {
	LockGuard lock(m_mutex);

	if (m_motion)
		m_readyQueue.push_front(entry);
	else
		m_recycleQueue.push_back(entry);

	if (m_useSync)
		entry.sync->cleanup();

	m_queueCV->signal();
}

void PreviewThread::run() {
	MTS_AUTORELEASE_BEGIN()

	bool initializedGraphics = false;

	try {
		m_device->init(m_parentDevice);
		m_device->setVisible(false);

		/* We have alrady seen this once */
		m_renderer->setLogLevel(ETrace);
		m_renderer->setWarnLogLevel(ETrace);
		m_renderer->init(m_device, m_parentRenderer);
		m_renderer->setLogLevel(EDebug);
		m_renderer->setWarnLogLevel(EWarn);

		m_accumProgram->init();
		m_accumProgramParam_source1 = m_accumProgram->getParameterID("source1");
		m_accumProgramParam_source2 = m_accumProgram->getParameterID("source2");
		m_useSync = m_renderer->getCapabilities()->isSupported(RendererCapabilities::ESyncObjects);
		m_shaderManager->init();

		initializedGraphics = true;
		m_started->set(true);

		while (true) {
			PreviewQueueEntry target;

			UniqueLock lock(m_mutex);
			while (!(m_quit || (m_context != NULL && m_context->mode == EPreview
					&& m_context->previewMethod != EDisabled
					&& ((m_readyQueue.size() != 0 && !m_motion) || m_recycleQueue.size() != 0))))
				m_queueCV->wait();

			MTS_AUTORELEASE_END()
			MTS_AUTORELEASE_BEGIN()

			if (m_quit) {
				break;
			} else if (m_recycleQueue.size() != 0) {
				target = m_recycleQueue.front();
				m_recycleQueue.pop_front();
			} else if (m_readyQueue.size() != 0 && !m_motion) {
				target = m_readyQueue.front();
				m_readyQueue.pop_front();
			} else {
				Log(EError, "Internal error!");
			}

			if (m_motion && m_vplCount >= m_minVPLs && m_minVPLs != 0) {
				/* The user is currently moving around, and a good enough
				   preview has already been rendered. Don't improve it to
				   avoid flicker */
				m_recycleQueue.push_back(target);
				m_queueCV->wait();
				continue;
			}

			lock.unlock();

			if (m_vplSampleOffset == 0) {
				m_accumBuffer = NULL;
				m_shaderManager->resetCounter();
			}

			const Film *film = m_context->scene->getFilm();
			Point3i size(film->getCropSize().x, film->getCropSize().y, 1);

			if (target.buffer == NULL || target.buffer->getSize() != size) {
				if (target.buffer) {
					target.buffer->cleanup();
					target.buffer->decRef();
					target.sync->decRef();
				}
				target.buffer = m_renderer->createGPUTexture(formatString("Communication buffer %i", target.id));
				target.buffer->setComponentFormat(GPUTexture::EFloat32);
				target.buffer->setPixelFormat(GPUTexture::ERGB);
				target.buffer->setSize(size);
				target.buffer->setFilterType(GPUTexture::ENearest);
				target.buffer->setFrameBufferType(GPUTexture::EColorAndDepthBuffer);
				target.buffer->setMipMapped(false);
				target.buffer->init();
				target.buffer->incRef();
				target.sync = m_renderer->createGPUSync();
				target.sync->incRef();
				m_renderer->finish();
			}

			int method = m_context->previewMethod;

			if (method != EOpenGL) {
				/* Do nothing, fall asleep in the next iteration */
			} else {
				bool initializeFramebuffer = (m_framebuffer == NULL)
					|| (m_framebuffer->getSize() != size);
				if (m_shaderManager->getScene() != m_context->scene) {
					m_shaderManager->setScene(m_context->scene);
					initializeFramebuffer = true;
				}

				if (initializeFramebuffer) {
					if (m_framebuffer)
						m_framebuffer->cleanup();
					m_framebuffer->setComponentFormat(GPUTexture::EFloat32);
					m_framebuffer->setPixelFormat(GPUTexture::ERGB);
					m_framebuffer->setSize(size);
					m_framebuffer->setFilterType(GPUTexture::ENearest);
					m_framebuffer->setFrameBufferType(GPUTexture::EColorBuffer);
					m_framebuffer->setMipMapped(false);
					m_framebuffer->init();
				}

				m_shaderManager->setShadowMapResolution(m_context->shadowMapResolution);
				m_shaderManager->setClamping(m_context->clamping);
				m_shaderManager->setDiffuseSources(m_context->diffuseSources);
				m_shaderManager->setDiffuseReceivers(m_context->diffuseReceivers);

				if (m_timer->getMilliseconds() > 1000) {
					Float count = m_vplsPerSecond / (Float) m_timer->getMilliseconds() * 1000;
					if (!m_motion)
						emit statusMessage(QString(formatString("%.1f VPLs/sec", count).c_str()));
					m_vplsPerSecond = 0;
					m_timer->reset();
				}

				if (m_vpls.empty()) {
					size_t oldOffset = m_vplSampleOffset;
					m_vplSampleOffset = generateVPLs(m_context->scene, m_random,
						m_vplSampleOffset, 1, m_context->pathLength, !m_motion, m_vpls);
					m_backgroundScaleFactor = m_vplSampleOffset - oldOffset;
				}

				VPL vpl = m_vpls.front();
				m_vpls.pop_front();

				oglRenderVPL(target, vpl);

				if (m_useSync)
					target.sync->init();
			}

			lock.lock();
			m_vplsPerSecond++;
			m_vplCount++;

			if (m_minVPLs == 0) {
				if (m_timer->getMilliseconds() > 50)
					m_minVPLs = m_vplCount;
			}

			if (m_vplCount >= m_minVPLs && m_minVPLs > 0)
				m_readyQueue.push_back(target);
			else
				m_recycleQueue.push_back(target);
			m_queueCV->signal();
			lock.unlock();

			if (m_sleep)
				sleep(10);
		}
	} catch (std::exception &e) {
		m_started->set(true);
		Log(EWarn, "Caught an exception: %s", e.what());
		emit caughtException(e.what());
	}

	if (initializedGraphics) {
		if (m_shaderManager)
			m_shaderManager->cleanup();

		m_accumProgram->cleanup();

		LockGuard lock(m_mutex);
		while (!m_readyQueue.empty()) {
			PreviewQueueEntry &entry = m_readyQueue.back();
			if (entry.buffer)
				entry.buffer->decRef();
			if (entry.sync)
				entry.sync->decRef();
			m_readyQueue.pop_back();
		}

		while (!m_recycleQueue.empty()) {
			PreviewQueueEntry &entry = m_recycleQueue.back();
			if (entry.buffer)
				entry.buffer->decRef();
			if (entry.sync)
				entry.sync->decRef();
			m_recycleQueue.pop_back();
		}

		m_renderer->shutdown();
		m_device->shutdown();
	}

	MTS_AUTORELEASE_END()
}

void PreviewThread::oglRenderVPL(PreviewQueueEntry &target, const VPL &vpl) {
	if (!m_context->scene->getSensor()->getClass()->derivesFrom(MTS_CLASS(ProjectiveCamera))) {
		/* This camera type is not supported! */
		target.buffer->activateTarget();
		target.buffer->clear();
		target.buffer->releaseTarget();
		m_accumBuffer = target.buffer;
		return;
	}

	UniqueLock lock(m_mutex);
	const ProjectiveCamera *sensor = static_cast<const ProjectiveCamera *>
		(m_context->scene->getSensor());
	Point2 aaSample(.5f), apertureSample(0.5f);
	if (!m_motion && !m_context->showKDTree && m_accumBuffer != NULL) {
		aaSample = Point2(m_random->nextFloat(), m_random->nextFloat());
		if (sensor->needsApertureSample())
			apertureSample = Point2(m_random->nextFloat(), m_random->nextFloat());
	}

	Transform projTransform = sensor->getProjectionTransform(apertureSample, aaSample);
	Transform worldTransform = m_camTransform->eval(
		sensor->getShutterOpen() +
			(m_motion ? 0.5f : m_random->nextFloat()) * sensor->getShutterOpenTime()
	);

	target.vplSampleOffset = m_vplSampleOffset;
	lock.unlock();

	m_shaderManager->setVPL(vpl);
	m_framebuffer->activateTarget();
	m_framebuffer->clear();
	m_renderer->setCamera(projTransform.getMatrix(), worldTransform.getInverseMatrix());
	m_shaderManager->drawAllGeometryForVPL(vpl, sensor);
	m_shaderManager->drawBackground(sensor, projTransform, vpl.emitterScale);
	m_framebuffer->releaseTarget();

	target.buffer->activateTarget();
	m_renderer->setDepthMask(false);
	m_renderer->setDepthTest(false);
	m_framebuffer->bind(0);
		if (m_accumBuffer == NULL) {
		/* First pass, there is no accumulation buffer yet */
		target.buffer->clear();
		m_renderer->blitTexture(m_framebuffer, true);
		m_framebuffer->blit(target.buffer, GPUTexture::EDepthBuffer);
	} else {
		/* Accumulate .. */
		m_accumBuffer->bind(1);
		m_accumProgram->bind();
		m_accumProgram->setParameter(m_accumProgramParam_source1, m_accumBuffer);
		m_accumProgram->setParameter(m_accumProgramParam_source2, m_framebuffer);
		m_renderer->blitQuad(true);
		m_accumProgram->unbind();
		m_accumBuffer->unbind();
		m_accumBuffer->blit(target.buffer, GPUTexture::EDepthBuffer);
	}
	m_renderer->setDepthMask(true);
	m_renderer->setDepthTest(true);
	m_framebuffer->unbind();
	target.buffer->releaseTarget();
	m_accumBuffer = target.buffer;

	static int i = 0;
	if ((++i % 4) == 0 || m_motion) {
		/* Don't let the queue get too large -- this makes
		   the whole system unresponsive */
		m_renderer->finish();
	} else {
		if (m_useSync) {
			m_renderer->flush();
		} else {
			/* No sync objects available - we have to wait
			   for everything to finish */
			m_renderer->finish();
		}
	}
}

void PreviewQueueEntry::cleanup() {
	if (buffer) {
		buffer->cleanup();
		buffer->decRef();
		buffer = NULL;
	}
	if (sync) {
		sync->cleanup();
		sync->decRef();
		sync = NULL;
	}
}


/////////////////////////////////////////////////////

CustomPreviewThread::CustomPreviewThread(Device *parentDevice, Renderer *parentRenderer)
  : AbstractPreviewThread("cpvwt"),
    m_parentDevice(parentDevice), m_parentRenderer(parentRenderer)
{
  MTS_AUTORELEASE_BEGIN();
  m_session = Session::create();
  m_device = Device::create(m_session);
  m_renderer = Renderer::create(m_session);
  m_mutex = new Mutex();
  m_session->init();
  MTS_AUTORELEASE_END();
}

CustomPreviewThread::~CustomPreviewThread() {
  MTS_AUTORELEASE_BEGIN()
    m_session->shutdown();
  MTS_AUTORELEASE_END()
}

static void fprintmatrix(FILE* f, Matrix4x4 m, int w, int h){
  for (int i=0;i<w;i++){
    for (int j=0;j<h;j++){
      fprintf(f,"%2.3f ",m(i,j));
    }
    fprintf(f,"\n");
  }
}


float* W1;
int W1_dim[2];
float* b1;
int b1_dim;

float* W2;
int W2_dim[2];
float* b2;
int b2_dim;

float* W3;
int W3_dim[2];
float* b3;
int b3_dim;

float* all_weights;
int all_weights_n = 0;


static void loadWeights(){
  std::ifstream file;
  file.open("/home/bengioe/data/udem/3d/weights_40.dat");
  int nWeights = 0;
  file.read((char*)&nWeights, sizeof(int));
  float** wps[] = {&W1, &b1, &W2, &b2, &W3, &b3};
  int* dims[] = {(int*)&W1_dim, &b1_dim, 
		 (int*)&W2_dim, &b2_dim,
		 (int*)&W3_dim, &b3_dim};
  int ndims[] = {2, 1, 2, 1, 2, 1};
  int flat_sizes[6];
  
  for (int i=0;i<nWeights;i++){
    int flat_size = 1;
    for (int j=0;j<ndims[i];j++){
      int d;
      file.read((char*)&d, sizeof(int));
      printf("%d ",d);
      dims[i][j] = d;
      flat_size *= d;
    }
    printf("\n");
    float* w = new float[flat_size];
    file.read((char*)w, sizeof(float)*flat_size);
    printf("%f %f %f...\n",w[0],w[1],w[2]);
    *wps[i] = w;
    all_weights_n += flat_size;
    flat_sizes[i] = flat_size;
  }

  all_weights = new float[all_weights_n];
  int windex = 0;
  for (int i=0;i<nWeights;i++){
    printf("%d/%d\n",windex,all_weights_n);
    memcpy(all_weights + windex, *wps[i], flat_sizes[i]*sizeof(float));
    windex += flat_sizes[i];
  }

  file.close();
}

void CustomPreviewThread::run(){
  MTS_AUTORELEASE_BEGIN();
  FILE* f = fopen("/dev/stdout","w");
  fprintf(f,"Running CustomPreviewThread\n");
  try{
    m_device->init(m_parentDevice);
    m_device->setVisible(false);

    /* We have alrady seen this once */
    m_renderer->setLogLevel(ETrace);
    m_renderer->setWarnLogLevel(ETrace);
    m_renderer->init(m_device, m_parentRenderer);
    m_renderer->setLogLevel(EDebug);
    m_renderer->setWarnLogLevel(EWarn);

  }catch (std::exception &e) {
    //m_started->set(true);
    Log(EWarn, "Caught an exception: %s", e.what());
    emit caughtException(e.what());
  }
  PreviewQueueEntry& target = m_lastTarget;
  
  
  ref<GPUProgram> prog = m_renderer->createGPUProgram("basic_shader");
  prog->setSource(GPUProgram::EVertexProgram,
		  "uniform mat4 instanceTransform;\n"
		  "varying vec3 posInWorldSpace;\n"
		  "varying vec3 normal;\n"
		  "void main(){\n"
		  "  vec4 pos = instanceTransform * gl_Vertex;\n"
		  "  gl_Position = gl_ModelViewProjectionMatrix * pos;\n"
		  "  posInWorldSpace = pos.xyz;\n"
		  "  normal = (instanceTransform * vec4(gl_Normal, 0.0)).xyz;\n"
		  "}\n");

  std::ifstream frag_file("/home/bengioe/local/mitsuba/mlp_shader.frag");
  std::string frag_shader((std::istreambuf_iterator<char>(frag_file)),
			  std::istreambuf_iterator<char>());
  prog->setSource(GPUProgram::EFragmentProgram, frag_shader);
		  /*
		  "uniform float weightVector[N_WEIGHTS];\n"
		  "varying vec3 posInWorldSpace;\n"
		  "void main(){\n"
		  "  gl_FragColor = vec4(posInWorldSpace.y < 400.0 ? posInWorldSpace.y/400.0:0, weightVector[0], 0, 1);\n"
		  "}\n");*/

  loadWeights();
  printf("NLAYER 1,2,3 %d %d %d\n",b1_dim, b2_dim, b3_dim);
  char buffer[255];
  sprintf(buffer,"%d",b1_dim);
  prog->define("N_LAYER1", buffer);
  sprintf(buffer,"%d",b2_dim);
  prog->define("N_LAYER2", buffer);
  sprintf(buffer,"%d",b3_dim);
  prog->define("N_LAYER3", buffer);
  sprintf(buffer,"%d",all_weights_n);
  prog->define("N_WEIGHTS", buffer);
  prog->init();
  prog->incRef();
  int u_instanceTransform = prog->getParameterID("instanceTransform");
  int u_weightVector = prog->getParameterID("weightVector");
  int u_camPos = prog->getParameterID("camPos");
  typedef std::chrono::high_resolution_clock Clock;
  typedef std::chrono::milliseconds milliseconds;
  Clock::time_point t0 = Clock::now();
  
  while (1){
    MTS_AUTORELEASE_END();
    Clock::time_point t1 = Clock::now();
    milliseconds ms = std::chrono::duration_cast<milliseconds>(t1 - t0);
    printf("Frame took %d ms\n",ms.count());
    usleep(50000);  
    MTS_AUTORELEASE_BEGIN();
    t0 = Clock::now();
    if (!m_context) continue;
    
    if (!target.buffer){
      const Film *film = m_context->scene->getFilm();
      Point3i size(film->getCropSize().x, film->getCropSize().y, 1);
      if (target.buffer == NULL || target.buffer->getSize() != size) {
	target.buffer = m_renderer->createGPUTexture(formatString("Communication buffer %i", target.id));
	target.buffer->setComponentFormat(GPUTexture::EFloat32);
	target.buffer->setPixelFormat(GPUTexture::ERGB);
	target.buffer->setSize(size);
	target.buffer->setFilterType(GPUTexture::ENearest);
	target.buffer->setFrameBufferType(GPUTexture::EColorBuffer);
	target.buffer->setMipMapped(false);
	target.buffer->init();
	target.buffer->incRef();
	target.sync = m_renderer->createGPUSync();
	target.sync->incRef();
      }
    }
    UniqueLock lock(m_mutex);
    target.buffer->activateTarget();
    target.buffer->clear();
    int glError = glGetError();
    //fprintf(f, "draw... %d %d\n",glError, target.buffer->getRefCount());
    m_renderer->setDepthTest(true);
    m_renderer->beginDrawingMeshes();
    
    const ProjectiveCamera *sensor = static_cast<const ProjectiveCamera *>
      (m_context->scene->getSensor());
    Point2 aaSample(.5f), apertureSample(0.5f);
    Transform projTransform = sensor->getProjectionTransform(apertureSample, aaSample);
    Transform worldTransform = sensor->getWorldTransform()->eval(sensor->getShutterOpen());
    //fprintmatrix(f,projTransform.getMatrix(),4,4);
    //fprintmatrix(f,worldTransform.getMatrix(),4,4);
    m_renderer->setCamera(projTransform.getMatrix(), worldTransform.getInverseMatrix());


    prog->bind();
    Point p = sensor->getWorldTransform()->eval(0).transformAffine(Point(0.0f));
    prog->setParameter(u_weightVector, all_weights_n, all_weights);
    prog->setParameter(u_camPos, p);
    
    Matrix4x4 currentObjTrafo;
    currentObjTrafo.setIdentity();
    for (std::vector<Renderer::TransformedGPUGeometry>::const_iterator it = m_geometry.begin();
	 it != m_geometry.end(); ++it) {
      const GPUGeometry *geo = (*it).first;
      //fprintf(f,"geo: %p\n",geo);
      const Matrix4x4 &trafo = (*it).second;
      const BSDF *bsdf = geo->getTriMesh()->getBSDF();
      const Emitter *emitter = geo->getTriMesh()->getEmitter();
      bool hasNormals = !geo->getTriMesh()->hasVertexNormals();
      Shader *bsdfShader = m_renderer->getShaderForResource(bsdf);
      if (trafo != currentObjTrafo || 1){
	//fprintmatrix(f,trafo,4,4);
	currentObjTrafo = trafo;
	prog->setParameter(u_instanceTransform, trafo);
	
      }
      m_renderer->drawMesh(geo);
    }

    prog->unbind();
    m_renderer->flush();
    m_renderer->finish();
    m_renderer->endDrawingMeshes();
    m_renderer->checkError();
    target.buffer->releaseTarget();
    Color3 c = target.buffer->getPixel(200,200);
    lock.unlock();
  }

  


  fclose(f);
  
  MTS_AUTORELEASE_END()
}

void CustomPreviewThread::setSceneContext(SceneContext *context, bool swapContext, bool motion) {
  
  MTS_AUTORELEASE_BEGIN();
  UniqueLock lock(m_mutex);
  m_context = context;
  FILE* f = fopen("/dev/stdout","w");
  //fprintf(f,"setSceneContext %p %d\n",context,motion);
  
  if (m_context){
    if (context->scene == m_scene){
      lock.unlock();
      MTS_AUTORELEASE_END();
      return;
    }
    fprintf(f,"new Scene: %p\n",context);
    m_scene = context->scene;
    const ref_vector<Shape> &shapes = context->scene->getShapes();
    Matrix4x4 identityTrafo;
    identityTrafo.setIdentity();

    /* Upload all geometry to the GPU, create shaders for scattering models */
    m_geometry.clear();
    for (size_t i=0; i<shapes.size(); ++i) {
      const Shape *shape = shapes[i].get();
      fprintf(f,"shape %p\n", shape);

      if (shape->getClass()->getName() == "Instance") {
	const Instance *instance = static_cast<const Instance *>(shape);
	const std::vector<const Shape *> &instantiatedShapes =
	  instance->getShapeGroup()->getKDTree()->getShapes();
	const AnimatedTransform *atrafo = instance->getWorldTransform();
	const Matrix4x4 &trafo = atrafo->eval(0).getMatrix();

	for (size_t j=0; j<instantiatedShapes.size(); ++j) {
	  shape = instantiatedShapes[j];
	  GPUGeometry *gpuGeo = m_renderer->registerGeometry(shape);
	  if (!gpuGeo)
	    continue;

	  Shader *shader = m_renderer->registerShaderForResource(shape->getBSDF());
	  if (shader && !shader->isComplete()) {
	    m_renderer->unregisterShaderForResource(shape->getBSDF());
	    shader = NULL;
	  }

	  gpuGeo->setShader(shader);
	  ssize_t geometryIndex = (ssize_t) m_geometry.size(), opaqueGeometryIndex = -1;
	  m_geometry.push_back(std::make_pair(gpuGeo, trafo));

	  /*if (shader && !(shader->getFlags() & Shader::ETransparent)) {
	    opaqueGeometryIndex = (ssize_t) m_opaqueGeometry.size();
	    m_opaqueGeometry.push_back(std::make_pair(gpuGeo, trafo));
	  }

	  if (!atrafo->isStatic()) {
	    m_animatedGeometry.push_back(AnimatedGeometryRecord(atrafo,
	    geometryIndex, opaqueGeometryIndex));
	    }*/
	}
      } else {
	GPUGeometry *gpuGeo = m_renderer->registerGeometry(shape);
	if (!gpuGeo)
	  continue;

	Shader *shader = m_renderer->registerShaderForResource(shape->getBSDF());
	if (shader && !shader->isComplete()) {
	  m_renderer->unregisterShaderForResource(shape->getBSDF());
	  shader = NULL;
	}

	gpuGeo->setShader(shader);
	m_geometry.push_back(std::make_pair(gpuGeo, identityTrafo));

	//if (shader && !(shader->getFlags() & Shader::ETransparent))
	//  m_opaqueGeometry.push_back(std::make_pair(gpuGeo, identityTrafo));
      }
    }
  }
   

  if (m_context) {
    ProjectiveCamera *camera = static_cast<ProjectiveCamera *>
      (m_context->scene->getSensor());
    m_camTransform = camera->getWorldTransform();
  }
  fprintf(f,"done. setSceneContext %p\n",context);
  fclose(f);
  lock.unlock();
  MTS_AUTORELEASE_END();
}

void CustomPreviewThread::resume(){
  
}

void CustomPreviewThread::waitUntilStarted(){
  
}

PreviewQueueEntry CustomPreviewThread::acquireBuffer(int ms){
  UniqueLock lock(m_mutex);
  //FILE* f = fopen("/dev/stdout","w");
  //fprintf(f,"acquireBuffer %d\n",ms);
  lock.unlock();
  PreviewQueueEntry target = m_lastTarget;
  target.vplSampleOffset = 1;
  //fclose(f);
  return target;
}

void CustomPreviewThread::releaseBuffer(PreviewQueueEntry &entry){
  entry.sync->cleanup();
  
}

void CustomPreviewThread::quit(){
  
}
