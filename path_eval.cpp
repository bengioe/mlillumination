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

#include <mitsuba/render/scene.h>
#include <mitsuba/core/statistics.h>
#include <iostream>
#include <fstream>
using namespace std;

MTS_NAMESPACE_BEGIN

static StatsCounter avgPathLength("Path tracer", "Average path length", EAverage);

/*! \plugin{path}{Path tracer}
 * \order{2}
 * \parameters{
 *     \parameter{maxDepth}{\Integer}{Specifies the longest path depth
 *         in the generated output image (where \code{-1} corresponds to $\infty$).
 *	       A value of \code{1} will only render directly visible light sources.
 *	       \code{2} will lead to single-bounce (direct-only) illumination,
 *	       and so on. \default{\code{-1}}
 *	   }
 *	   \parameter{rrDepth}{\Integer}{Specifies the minimum path depth, after
 *	      which the implementation will start to use the ``russian roulette''
 *	      path termination criterion. \default{\code{5}}
 *	   }
 *     \parameter{strictNormals}{\Boolean}{Be strict about potential
 *        inconsistencies involving shading normals? See the description below
 *        for details.\default{no, i.e. \code{false}}
 *     }
 *     \parameter{hideEmitters}{\Boolean}{Hide directly visible emitters?
 *        See page~\pageref{sec:hideemitters} for details.
 *        \default{no, i.e. \code{false}}
 *     }
 * }
 *
 * This integrator implements a basic path tracer and is a \emph{good default choice}
 * when there is no strong reason to prefer another method.
 *
 * To use the path tracer appropriately, it is instructive to know roughly how
 * it works: its main operation is to trace many light paths using \emph{random walks}
 * starting from the sensor. A single random walk is shown below, which entails
 * casting a ray associated with a pixel in the output image and searching for
 * the first visible intersection. A new direction is then chosen at the intersection,
 * and the ray-casting step repeats over and over again (until one of several
 * stopping criteria applies).
 * \begin{center}
 * \includegraphics[width=.7\textwidth]{images/integrator_path_figure.pdf}
 * \end{center}
 * At every intersection, the path tracer tries to create a connection to
 * the light source in an attempt to find a \emph{complete} path along which
 * light can flow from the emitter to the sensor. This of course only works
 * when there is no occluding object between the intersection and the emitter.
 *
 * This directly translates into a category of scenes where
 * a path tracer can be expected to produce reasonable results: this is the case
 * when the emitters are easily ``accessible'' by the contents of the scene. For instance,
 * an interior scene that is lit by an area light will be considerably harder
 * to render when this area light is inside a glass enclosure (which
 * effectively counts as an occluder).
 *
 * Like the \pluginref{direct} plugin, the path tracer internally relies on multiple importance
 * sampling to combine BSDF and emitter samples. The main difference in comparison
 * to the former plugin is that it considers light paths of arbitrary length to compute
 * both direct and indirect illumination.
 *
 * For good results, combine the path tracer with one of the
 * low-discrepancy sample generators (i.e. \pluginref{ldsampler},
 * \pluginref{halton}, or \pluginref{sobol}).
 *
 * \paragraph{Strict normals:}\label{sec:strictnormals}
 * Triangle meshes often rely on interpolated shading normals
 * to suppress the inherently faceted appearance of the underlying geometry. These
 * ``fake'' normals are not without problems, however. They can lead to paradoxical
 * situations where a light ray impinges on an object from a direction that is classified as ``outside''
 * according to the shading normal, and ``inside'' according to the true geometric normal.
 *
 * The \code{strictNormals}
 * parameter specifies the intended behavior when such cases arise. The default (\code{false}, i.e. ``carry on'')
 * gives precedence to information given by the shading normal and considers such light paths to be valid.
 * This can theoretically cause light ``leaks'' through boundaries, but it is not much of a problem in practice.
 *
 * When set to \code{true}, the path tracer detects inconsistencies and ignores these paths. When objects
 * are poorly tesselated, this latter option may cause them to lose a significant amount of the incident
 * radiation (or, in other words, they will look dark).
 *
 * The bidirectional integrators in Mitsuba (\pluginref{bdpt}, \pluginref{pssmlt}, \pluginref{mlt} ...)
 * implicitly have \code{strictNormals} set to \code{true}. Hence, another use of this parameter
 * is to match renderings created by these methods.
 *
 * \remarks{
 *    \item This integrator does not handle participating media
 *    \item This integrator has poor convergence properties when rendering
 *    caustics and similar effects. In this case, \pluginref{bdpt} or
 *    one of the photon mappers may be preferable.
 * }
 */

#include <cmath>

float sigmoid(float x){
  return 1 / (1 + exp(-x));
}
float rectifier(float x){
  //return x > 1? 1 : x<-1 ? -1 : x;
  return x < 0 ? 0 : x;
}

static Point global_L;
static bool printstuff = true;

class MIPathTracerEval : public MonteCarloIntegrator {
public:

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
	MIPathTracerEval(const Properties &props)
		: MonteCarloIntegrator(props) { 
	  printf("Has light: %d\n", props.hasProperty("light"));
	  global_L = props.getPoint("light");
	  ifstream file;
	  file.open("/home/bengioe/data/udem/3d/weights_40.dat");
	  int nWeights = 0;
	  file.read((char*)&nWeights, sizeof(int));
	  float** wps[] = {&W1, &b1, &W2, &b2, &W3, &b3};
	  int* dims[] = {(int*)&W1_dim, &b1_dim, 
			 (int*)&W2_dim, &b2_dim,
			 (int*)&W3_dim, &b3_dim};
	  int ndims[] = {2, 1, 2, 1, 2, 1};

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
	  }
	  file.close();
	  {
	  float r,g,b;
	  Point TargetX(1,1,1);
	  Vector TargetN(1,1,1);
	  Vector TargetV(1,1,1);
	  Vector TargetL(1,1,1);
	  evalNet(r,g,b,TargetX, TargetN, TargetV, TargetL);
	  printf("z3: %f %f %f\n",r,g,b);
	  }
	  {

	  float r,g,b;
	  Point TargetX(0,0,0);
	  Vector TargetN(0,0,0);
	  Vector TargetV(0,0,0);
	  Vector TargetL(0,0,0);
	  evalNet(r,g,b,TargetX, TargetN, TargetV, TargetL);
	  printf("z3: %f %f %f\n",r,g,b);
	  }
	  printstuff = false;
	}
        
        void evalNet(float& r, float& g, float& b, Point X, Vector N, Vector V, Vector L) const{
	  /*printf("%f %f %f - %f %f %f - %f %f %f\n", X.x,X.y,X.z, 
			  N.x,N.y,N.z, 
			  V.x,V.y,V.z);*/
#define ACTIVE1 rectifier
#define ACTIVE2 rectifier
#define ACTIVE3 sigmoid
	  float data[] = {X.x/600.,X.y/600.,X.z/600., 
			  N.x,N.y,N.z, 
			  V.x,V.y,V.z,
			  L.x/600.,L.y/600.,L.z/600.};

	  float z1[W1_dim[1]];
	  for (int i=0;i<W1_dim[1];i++){
	    float s=0;
	    for (int j=0;j<W1_dim[0];j++){
	      s += data[j]*W1[W1_dim[1]*j+i];
	    }
	    z1[i] = ACTIVE1(s + b1[i]);
	    if (printstuff){
	      printf("%f ",z1[i]);
	    }
	  }
	  if (printstuff)
	    printf(" (z1) \n");

	  float z2[W2_dim[1]];
	  for (int i=0;i<W2_dim[1];i++){
	    float s=0;
	    for (int j=0;j<W2_dim[0];j++){
	      s += z1[j]*W2[W2_dim[1]*j+i];
	    }
	    z2[i] = ACTIVE2(s + b2[i]);
	    if (printstuff){
	      printf("%f ",z2[i]);
	    }
	  }
	  if (printstuff)
	    printf(" (z2) \n");
	  //printf("z2: %f %f %f\n",z2[0],z2[1],z2[2]);
	  float z3[W3_dim[1]];
	  for (int i=0;i<W3_dim[1];i++){
	    float s=0;
	    for (int j=0;j<W3_dim[0];j++){
	      s += z2[j]*W3[W3_dim[1]*j+i];
	    }
	    z3[i] = ACTIVE3(s + b3[i]);
	  }
	  r = z3[0];
	  g = z3[1];
	  b = z3[2];
	  //printf("%f %f %f\n",r,g,b);
#undef ACTIVE1
#undef ACTIVE2
#undef ACTIVE3
	}
	/// Unserialize from a binary data stream
	MIPathTracerEval(Stream *stream, InstanceManager *manager)
		: MonteCarloIntegrator(stream, manager) { }

	Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
		/* Some aliases and local variables */
		const Scene *scene = rRec.scene;
		Intersection &its = rRec.its;
		RayDifferential ray(r);
		Spectrum Li(0.0f);
		bool scattered = false;

		Point TargetX;
		Vector TargetN;
		Vector TargetV;
		Vector TargetL;
		bool isValidData = false;

		/* Perform the first ray intersection (or ignore if the
		   intersection has already been provided). */
		rRec.rayIntersect(ray);
		ray.mint = Epsilon;

		Spectrum throughput(1.0f);
		Float eta = 1.0f;


		do{
			if (!its.isValid()) {
				/* If no intersection could be found, potentially return
				   radiance from a environment luminaire if it exists */
				if ((rRec.type & RadianceQueryRecord::EEmittedRadiance)
					&& (!m_hideEmitters || scattered))
					Li += throughput * scene->evalEnvironment(ray);
				break;
			}

			const BSDF *bsdf = its.getBSDF(ray);

			/* Possibly include emitted radiance if requested */
			if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
				&& (!m_hideEmitters || scattered))
				Li += throughput * its.Le(-ray.d);

			/* Include radiance from a subsurface scattering model if requested */
			if (its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance))
				Li += throughput * its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth);

			if ((rRec.depth >= m_maxDepth && m_maxDepth > 0)
				|| (m_strictNormals && dot(ray.d, its.geoFrame.n)
					* Frame::cosTheta(its.wi) >= 0)) {

				/* Only continue if:
				   1. The current path length is below the specifed maximum
				   2. If 'strictNormals'=true, when the geometric and shading
				      normals classify the incident direction to the same side */
				break;
			}

			/* ==================================================================== */
			/*                     Direct illumination sampling                     */
			/* ==================================================================== */

			/* Estimate the direct illumination if this is requested */
			DirectSamplingRecord dRec(its);

			if (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance &&
				(bsdf->getType() & BSDF::ESmooth)) {
				Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
				if (!value.isZero()) {
					const Emitter *emitter = static_cast<const Emitter *>(dRec.object);
					/*if (rRec.depth == 1){
					  TargetL = dRec.d;
					  isValidData = true;
					  }*/
					/* Allocate a record for querying the BSDF */
					BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);

					/* Evaluate BSDF * cos(theta) */
					const Spectrum bsdfVal = bsdf->eval(bRec);

					/* Prevent light leaks due to the use of shading normals */
					if (!bsdfVal.isZero() && (!m_strictNormals
							|| dot(its.geoFrame.n, dRec.d) * Frame::cosTheta(bRec.wo) > 0)) {

						/* Calculate prob. of having generated that direction
						   using BSDF sampling */
						Float bsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
							? bsdf->pdf(bRec) : 0;

						/* Weight using the power heuristic */
						Float weight = miWeight(dRec.pdf, bsdfPdf);
						Li += throughput * value * bsdfVal * weight;
					}
				}
			}

			TargetN = its.geoFrame.n;
			TargetV = ray.d;
			TargetX = its.p;
			isValidData = true;
			/* ==================================================================== */
			/*                            BSDF sampling                             */
			/* ==================================================================== */

			/* Sample BSDF * cos(theta) */
			Float bsdfPdf;
			BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
			Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());
			if (bsdfWeight.isZero())
				break;

			scattered |= bRec.sampledType != BSDF::ENull;

			/* Prevent light leaks due to the use of shading normals */
			const Vector wo = its.toWorld(bRec.wo);
			Float woDotGeoN = dot(its.geoFrame.n, wo);
			if (m_strictNormals && woDotGeoN * Frame::cosTheta(bRec.wo) <= 0)
				break;

			bool hitEmitter = false;
			Spectrum value;

			/* Trace a ray in this direction */
			ray = Ray(its.p, wo, ray.time);
			if (scene->rayIntersect(ray, its)) {
				/* Intersected something - check if it was a luminaire */
				if (its.isEmitter()) {
					value = its.Le(-ray.d);
					dRec.setQuery(ray, its);
					hitEmitter = true;
				}
			} else {
				/* Intersected nothing -- perhaps there is an environment map? */
				const Emitter *env = scene->getEnvironmentEmitter();

				if (env) {
					if (m_hideEmitters && !scattered)
						break;

					value = env->evalEnvironment(ray);
					if (!env->fillDirectSamplingRecord(dRec, ray))
						break;
					hitEmitter = true;
				} else {
					break;
				}
			}

			/* Keep track of the throughput and relative
			   refractive index along the path */
			throughput *= bsdfWeight;
			eta *= bRec.eta;

			/* If a luminaire was hit, estimate the local illumination and
			   weight using the power heuristic */
			if (hitEmitter &&
				(rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)) {
				/* Compute the prob. of generating that direction using the
				   implemented direct illumination sampling technique */
				const Float lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ?
					scene->pdfEmitterDirect(dRec) : 0;
				Li += throughput * value * miWeight(bsdfPdf, lumPdf);
			}
		} while(0);

		if (isValidData){
		  Spectrum k; 
		  float r,g,b;
		  evalNet(r,g,b,TargetX, TargetN, TargetV, Vector(global_L));
		  k.fromLinearRGB(r,g,b);
		  Li = k;
		}

		/* Store statistics */
		avgPathLength.incrementBase();
		avgPathLength += rRec.depth;

		return Li;
	}

	inline Float miWeight(Float pdfA, Float pdfB) const {
		pdfA *= pdfA;
		pdfB *= pdfB;
		return pdfA / (pdfA + pdfB);
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		MonteCarloIntegrator::serialize(stream, manager);
	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "MIPathTracerEval[" << endl
			<< "  maxDepth = " << m_maxDepth << "," << endl
			<< "  rrDepth = " << m_rrDepth << "," << endl
			<< "  strictNormals = " << m_strictNormals << endl
			<< "]";
		return oss.str();
	}

	MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS_S(MIPathTracerEval, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(MIPathTracerEval, "MI path tracer eval");
MTS_NAMESPACE_END
