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

static Float diff = 0.0;
static long dcount = 0;
static FILE* dset_file = NULL;
static Point global_L;
static bool hasLight;

class MIPathTracerLR : public MonteCarloIntegrator {
public:
	MIPathTracerLR(const Properties &props)
	  : MonteCarloIntegrator(props) { 
	  printf("Build PTLR\n");
	  printf("Has light: %d\n", props.hasProperty("light"));
	  //global_L = props.getPoint("light");
	  diff = 0; dcount = 0; hasLight = false;
	  if (dset_file == NULL){
	    dset_file = fopen("dataset.dat","w");
	  }
	}
    
        ~MIPathTracerLR(){
	  printf("Destroy PTLR\n");
	  fflush(dset_file);
	  printf("Done.\n");
	}
	/// Unserialize from a binary data stream
	MIPathTracerLR(Stream *stream, InstanceManager *manager)
		: MonteCarloIntegrator(stream, manager) { }

	Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const{
		/* Some aliases and local variables */
		const Scene *scene = rRec.scene;
		Spectrum Li(0.0f);
		bool scattered = false;

		Spectrum TargetLi(0.0f);
		Point TargetX;
		Vector TargetN;
		Vector TargetV;
		//Vector TargetL;
		
		bool isValidData = true;

		RadianceQueryRecord initRRec = rRec;
		int NITER = 80;
		for (int iter=0;iter<NITER;iter++){
		  rRec = initRRec;
		  Intersection &its = rRec.its;
		  RayDifferential ray(r);
		  /* Perform the first ray intersection (or ignore if the
		     intersection has already been provided). */
		  rRec.rayIntersect(ray);
		  ray.mint = Epsilon;
		  Spectrum throughput(1.0f);
		  Float eta = 1.0f;
		while (rRec.depth <= m_maxDepth || m_maxDepth < 0) {
			if (!its.isValid()) {
				/* If no intersection could be found, potentially return
				   radiance from a environment luminaire if it exists */
				if ((rRec.type & RadianceQueryRecord::EEmittedRadiance)
					&& (!m_hideEmitters || scattered))
					Li += throughput * scene->evalEnvironment(ray);
				if (rRec.depth == 1){
				  isValidData = false;
				}
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
				(bsdf->getType() & BSDF::ESmooth) &&
			    (rRec.depth != 1 || iter==0)) {
				Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
				
				if (!value.isZero()) {
					const Emitter *emitter = static_cast<const Emitter *>(dRec.object);
					
					if (!hasLight){
					  Point ref = emitter->getAABB().getCorner(0);
					  global_L = ref;
					  printf("new light: %f %f %f\n", ref.x, ref.y, ref.z);
					  hasLight = true;
					}
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

			// At this point (in the first iteration) is
			// the Li value we want to learn to add to
			if (rRec.depth == 1 && iter==0){
			  //TargetLi = Li;
			  TargetN = its.geoFrame.n;
			  TargetV = ray.d;
			  TargetX = its.p;
			}

			if (its.isValid()){
			  if (0){
			    Float r,g,b;
			    Li.toLinearRGB(r,g,b);
			    printf("Li %d %f %f %f\n",rRec.depth,r,g,b);
			  }
			  if (0){
			    Float r,g,b;
			    Li.toLinearRGB(r,g,b);
			    printf("Li %f %f %f -- %f %f %f -- %f %f %f\n",its.p.x,its.p.y,its.p.z,
				   its.geoFrame.n.x,its.geoFrame.n.y,its.geoFrame.n.z,
				   r,g,b);
			  }
			}
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

			/* ==================================================================== */
			/*                         Indirect illumination                        */
			/* ==================================================================== */

			/* Set the recursive query type. Stop if no surface was hit by the
			   BSDF sample or if indirect illumination was not requested */
			if (!its.isValid() || !(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
				break;
			rRec.type = RadianceQueryRecord::ERadianceNoEmission;

			if (rRec.depth++ >= m_rrDepth) {
				/* Russian roulette: try to keep path weights equal to one,
				   while accounting for the solid angle compression at refractive
				   index boundaries. Stop with at least some probability to avoid
				   getting stuck (e.g. due to total internal reflection) */

				Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
				if (rRec.nextSample1D() >= q)
					break;
				throughput /= q;
			}
		}

		}
		Li *= 1./NITER;

		/* Store statistics */
		avgPathLength.incrementBase();
		avgPathLength += rRec.depth;
		
		if (rand() < RAND_MAX/2048){
		  Float r,g,b;
		  Li.toLinearRGB(r,g,b);
		  Float x,y,z;
		  TargetLi.toLinearRGB(x,y,z);
		  diff += (r-x)*(r-x)+(g-y)*(g-y)+(b-z)*(b-z);
		  //printf("return: %p %f %f %f -- %f %f %f -- %f %f %f\n",this, r,g,b,x,y,z,diff,diff/++dcount,(r-x)*(r-x)+(g-y)*(g-y)+(b-z)*(b-z));
		}
		if (isValidData){
		  Float x,y,z;
		  Float r,g,b;
		  Li.toLinearRGB(x,y,z);
		  TargetLi.toLinearRGB(r,g,b);
		  //printf("%f %f %f -- %f %f %f\n",x,y,z,r,g,b);
		  //x -= r; y -= g; z -= b;
		  //Li.fromLinearRGB(x,y,z);
		  float data[] = {x,y,z, 
				  TargetX.x,TargetX.y,TargetX.z, 
				  TargetN.x,TargetN.y,TargetN.z, 
				  TargetV.x,TargetV.y,TargetV.z,
				  global_L.x,global_L.y,global_L.z};
		  //printf("%f %f %f (%f %f %f)\n",x,y,z,TargetX.x,TargetX.y,TargetX.z);
		  fwrite(data, 4, 15, dset_file);
		}
  
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
		oss << "MIPathTracerLR[" << endl
			<< "  maxDepth = " << m_maxDepth << "," << endl
			<< "  rrDepth = " << m_rrDepth << "," << endl
			<< "  strictNormals = " << m_strictNormals << endl
			<< "]";
		return oss.str();
	}

	MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS_S(MIPathTracerLR, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(MIPathTracerLR, "MI LR path tracer");
MTS_NAMESPACE_END
