#include "framebuffer.h"
#include <iostream>
#include <math.h>
#include <algorithm>
#include <QGLWidget>
#include <QDir>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Partition_traits_2.h>
#include <CGAL/partition_2.h>
#include <CGAL/point_generators_2.h>
#include <CGAL/random_polygon_2.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/create_offset_polygons_2.h>

using namespace std;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Partition_traits_2<K>                         Traits;
typedef Traits::Point_2                                     Point_2;
typedef Traits::Polygon_2                                   Polygon_2;
typedef Polygon_2::Vertex_iterator                          Vertex_iterator;
typedef std::list<Polygon_2>                                Polygon_list;
typedef CGAL::Creator_uniform_2<int, Point_2>               Creator;
typedef CGAL::Random_points_in_square_2< Point_2, Creator > Point_generator;
typedef boost::shared_ptr<Polygon_2>						PolygonPtr;

AABB::AABB() {
	corners[0][0] = (numeric_limits<float>::max)();
	corners[0][1] = (numeric_limits<float>::max)();
	corners[0][2] = (numeric_limits<float>::max)();

	corners[1][0] = -(numeric_limits<float>::max)();
	corners[1][1] = -(numeric_limits<float>::max)();
	corners[1][2] = -(numeric_limits<float>::max)();
}

AABB::AABB(const glm::vec3& p) {
	corners[0] = corners[1] = p;
}

void AABB::AddPoint(const glm::vec3& p) {
	if (p.x < corners[0].x) {
		corners[0][0] = p.x;
	}
	if (p.y < corners[0].y) {
		corners[0][1] = p.y;
	}
	if (p.z < corners[0].z) {
		corners[0][2] = p.z;
	}

	if (p.x > corners[1].x) {
		corners[1][0] = p.x;
	}
	if (p.y > corners[1].y) {
		corners[1][1] = p.y;
	}
	if (p.z > corners[1].z) {
		corners[1][2] = p.z;
	}
}

const glm::vec3& AABB::minCorner() const {
	return corners[0];
}

const glm::vec3& AABB::maxCorner() const {
	return corners[1];
}

glm::vec3 AABB::Size() const {
	return corners[1] - corners[0];
}

Stroke::Stroke(const std::string& filename) {
	stroke_image = cv::imread(filename.c_str());
}

glm::vec3 Stroke::getColor(float x, float y) const {
	// return white color if the pixel is outside the stroke image
	if (x < 0 || x >= stroke_image.cols || y < 0 || y >= stroke_image.rows) return glm::vec3(1, 1, 1);

	int b = stroke_image.at<cv::Vec3b>(y, x)[0];
	int g = stroke_image.at<cv::Vec3b>(y, x)[1];
	int r = stroke_image.at<cv::Vec3b>(y, x)[2];

	return glm::vec3(r, g, b);
}

FrameBuffer::FrameBuffer(int _w, int _h) {
	w = _w;
	h = _h;
	pix = new unsigned int[w*h];
	zb  = new float[w*h];

	// initialize stylized polylines
	style_polylines.resize(5);
	style_polylines[0].push_back(glm::vec2(-0.025, -0.025));
	style_polylines[0].push_back(glm::vec2(0.3, 0.035));
	style_polylines[0].push_back(glm::vec2(0.6, 0.05));
	style_polylines[0].push_back(glm::vec2(0.85, 0.04));
	style_polylines[0].push_back(glm::vec2(1.01, 0.02));

	style_polylines[1].push_back(glm::vec2(-0.01, 0.01));
	style_polylines[1].push_back(glm::vec2(0.13, -0.01));
	style_polylines[1].push_back(glm::vec2(0.27, -0.02));
	style_polylines[1].push_back(glm::vec2(0.7, -0.02));
	style_polylines[1].push_back(glm::vec2(0.81, 0));
	style_polylines[1].push_back(glm::vec2(1.02, 0));

	style_polylines[2].push_back(glm::vec2(-0.02, 0.0));
	style_polylines[2].push_back(glm::vec2(0.12, 0.01));
	style_polylines[2].push_back(glm::vec2(0.37, 0.02));
	style_polylines[2].push_back(glm::vec2(0.6, 0.02));
	style_polylines[2].push_back(glm::vec2(0.77, 0.01));
	style_polylines[2].push_back(glm::vec2(0.91, 0.005));
	style_polylines[2].push_back(glm::vec2(0.99, -0.01));

	style_polylines[3].push_back(glm::vec2(-0.02, 0.0));
	style_polylines[3].push_back(glm::vec2(0.57, -0.01));
	style_polylines[3].push_back(glm::vec2(0.8, -0.01));
	style_polylines[3].push_back(glm::vec2(1.01, 0.01));

	style_polylines[4].push_back(glm::vec2(-0.01, 0.0));
	style_polylines[4].push_back(glm::vec2(0.13, -0.01));
	style_polylines[4].push_back(glm::vec2(0.23, -0.02));
	style_polylines[4].push_back(glm::vec2(0.31, -0.02));
	style_polylines[4].push_back(glm::vec2(0.38, -0.01));
	style_polylines[4].push_back(glm::vec2(0.46, 0.0));
	style_polylines[4].push_back(glm::vec2(0.61, 0.02));
	style_polylines[4].push_back(glm::vec2(0.68, 0.03));
	style_polylines[4].push_back(glm::vec2(0.8, 0.03));
	style_polylines[4].push_back(glm::vec2(0.88, 0.02));
	style_polylines[4].push_back(glm::vec2(0.97, 0.01));
}

FrameBuffer::~FrameBuffer() {
	delete [] pix;
	delete [] zb;
}

void FrameBuffer::resize(int _w, int _h) {
	delete [] pix;
	delete [] zb;

	w = _w;
	h = _h;
	pix = new unsigned int[w*h];
	zb  = new float[w*h];
}

void FrameBuffer::draw() {
	glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, pix);
}

void FrameBuffer::setClearColor(const glm::vec3& clear_color) {
	this->clear_color = clear_color;
}

/**
 * Set all pixels to given color.
 *
 * @param bgr	the given color
 */
void FrameBuffer::clear() {
	unsigned int clr = GetColor(clear_color);
	for (int uv = 0; uv < w*h; uv++) {
		pix[uv] = clr;
		zb[uv] = 100.0f;
	}
}

/**
 * Set one pixel to given color.
 * If the specified pixel is out of the screen, it does nothing.
 *
 * @param u		x coordinate of the pixel
 * @param v		y coordinate of the pixel
 * @param clr	the color
 */
void FrameBuffer::Set(int u, int v, const glm::vec3& clr, float z) {
	if (u < 0 || u > w-1 || v < 0 || v > h-1) return;

	//Set(u, v, clr, z);
	pix[(h-1-v)*w+u] = GetColor(clr);
	zb[(h-1-v)*w+u] = z;
}

void FrameBuffer::Add(int u, int v, const glm::vec3& color) {
	if (u < 0 || u > w-1 || v < 0 || v > h-1) return;

	pix[(h-1-v)*w+u] = ~(~pix[(h-1-v)*w+u] | ~GetColor(color));
}

/**
 * Draw 2D segment with color interpolation.
 *
 * @param p0	the first point of the segment
 * @param c0	the color of the first point
 * @param p1	the second point of the segment
 * @param c1	the color of the second point
 */
void FrameBuffer::Draw2DSegment(const glm::vec3& p0, const glm::vec3& c0, const glm::vec3& p1, const glm::vec3& c1) {
	float dx = fabsf(p0.x - p1.x);
	float dy = fabsf(p0.y - p1.y);

	int n;
	if (dx < dy) {
		n = 1 + (int)dy;
	} else {
		n = 1 + (int)dx;
	}

	for (int i = 0; i <= n; i++) {
		float frac = (float) i / (float)n;
		glm::vec3 curr = p0 + (p1-p0) * frac;
		glm::vec3 currc = c0 + (c1-c0) * frac;
		int u = (int)curr[0];
		int v = (int)curr[1];
		Set(u, v, currc, curr.z);
	}
}

/**
 * Draw 3D segment with color interpolation.
 *
 * @param ppc	the camera
 * @param p0	the first point of the segment
 * @param c0	the color of the first point
 * @param p1	the second point of the segment
 * @param c1	the color of the second point
 */
void FrameBuffer::Draw3DSegment(Camera* camera, const glm::vec3& p0, const glm::vec3& c0, const glm::vec3& p1, const glm::vec3& c1) {
	glm::vec3 pp0, pp1;
	if (!camera->Project(p0, pp0)) return;
	if (!camera->Project(p1, pp1)) return;

	pp0 = convertScreenCoordinate(pp0);
	pp1 = convertScreenCoordinate(pp1);

	Draw2DSegment(pp0, c0, pp1, c1);
}

/**
 * Draw 3D segment with color interpolation.
 *
 * @param ppc	the camera
 * @param p0	the first point of the segment
 * @param c0	the color of the first point
 * @param p1	the second point of the segment
 * @param c1	the color of the second point
 */
void FrameBuffer::Draw3DStroke(Camera* camera, const glm::vec3& p0, const glm::vec3& p1, int seed) {
	glm::vec3 q0 = p0;
	glm::vec3 q1 = p1;

	if (q0.x > q1.x) {
		swap(q0, q1);
	} else if (q0.x == q1.x) {
		if (q0.y > q1.y) {
			swap(q0, q1);
		} else if (q0.y == q1.y) {
			if (q0.z > q1.z) {
				swap(q0, q1);
			}
		}
	}

	glm::vec3 pp0, pp1;
	if (!camera->Project(q0, pp0)) return;
	if (!camera->Project(q1, pp1)) return;

	pp0 = convertScreenCoordinate(pp0);
	pp1 = convertScreenCoordinate(pp1);

	srand(seed + q0.x * 100 + q0.y * 50 + q0.z * 10 + q1.x * 20 + q1.y * 30 + q1.z * 40);

	int polyline_index = rand() % style_polylines.size();

	Draw2DPolyline(pp0, pp1, polyline_index);
}

/**
 * Draw 2D polyline.
 *
 * @param p0	the first point of the segment
 * @param p1	the second point of the segment
 */
void FrameBuffer::Draw2DPolyline(const glm::vec3& p0, const glm::vec3& p1, int polyline_index) {
	float theta = atan2(p1.y - p0.y, p1.x - p0.x);
	float scale = glm::length(p1 - p0);

	cv::Mat_<float> R(2, 2);
	R(0, 0) = scale * cosf(theta);
	R(0, 1) = -scale * sinf(theta);
	R(1, 0) = scale * sinf(theta);
	R(1, 1) = scale * cosf(theta);

	cv::Mat_<float> A(2, 1);
	A(0, 0) = p0.x;
	A(1, 0) = p0.y;

	for (int i = 0; i < style_polylines[polyline_index].size() - 1; ++i) {
		cv::Mat_<float> X0(2, 1);
		X0(0, 0) = style_polylines[polyline_index][i].x;
		X0(1, 0) = style_polylines[polyline_index][i].y;
		cv::Mat_<float> T0 = R * X0 + A;

		cv::Mat_<float> X1(2, 1);
		X1(0, 0) = style_polylines[polyline_index][i+1].x;
		X1(1, 0) = style_polylines[polyline_index][i+1].y;
		cv::Mat_<float> T1 = R * X1 + A;

		Draw2DSegment(glm::vec3(T0(0, 0), T0(1, 0), 0), glm::vec3(0, 0, 0), glm::vec3(T1(0, 0), T1(1, 0), 0), glm::vec3(0, 0, 0));
	}
}

/**
 * objectを描画する。
 */
void FrameBuffer::rasterize(Camera* camera, const std::vector<std::vector<Vertex> >& vertices, int seed) {
	std::multimap<float, std::vector<Vertex> > sortedVertices;

	for (int i = 0; i < vertices.size(); ++i) {
		float depth = maxDepth(camera, vertices[i]);

		sortedVertices.insert(std::make_pair(depth, vertices[i]));
	}

	for (auto it = sortedVertices.rbegin(); it != sortedVertices.rend(); ++it) {
		if (it->second.size() == 3) {
			rasterizePolygon(camera, it->second, seed);
		} else {
			rasterizeConcavePolygon(camera, it->second, seed);
		}
	}
}

/**
 * １つのfaceを描画する。
 */
void FrameBuffer::rasterizePolygon(Camera* camera, const std::vector<Vertex>& vertices, int seed) {
	for (int i = 1; i < vertices.size() - 1; ++i) {
		rasterizeTriangle(camera, vertices[0].position, vertices[i].position, vertices[i + 1].position);
	}

	for (int i = 0; i < vertices.size(); ++i) {
		int next = (i + 1) % vertices.size();

		Draw3DStroke(camera, vertices[i].position, vertices[next].position, seed);
	}
}

void FrameBuffer::rasterizeConcavePolygon(Camera* camera, const std::vector<Vertex>& vertices, int seed) {
	Polygon_2 polygon;
	glm::vec3 prev_pp;
	glm::vec3 first_pp;
	for (int i = 0; i < vertices.size(); ++i) {
		glm::vec3 pp;
		camera->Project(vertices[i].position, pp);
		pp = convertScreenCoordinate(pp);
		
		if (i == 0) {
			first_pp = pp;
		}

		if (i > 0 && pp.x == prev_pp.x && pp.y == prev_pp.y) continue;
		if (i > 0 && pp.x == first_pp.x && pp.y == first_pp.y) continue;

		prev_pp = pp;
		polygon.push_back(Point_2(pp.x, pp.y));
	}
	
	if (polygon.size() > 4) {
		// tesselate the concave polygon
		Polygon_list partition_polys;
		Traits       partition_traits;
		if (polygon.is_clockwise_oriented()) {
			polygon.reverse_orientation();
		}
		CGAL::greene_approx_convex_partition_2(polygon.vertices_begin(), polygon.vertices_end(), std::back_inserter(partition_polys), partition_traits);

		for (auto fit = partition_polys.begin(); fit != partition_polys.end(); ++fit) {
			std::vector<glm::vec2> pts;
			for (auto vit = fit->vertices_begin(); vit != fit->vertices_end(); ++vit) {
				pts.push_back(glm::vec2(vit->x(), vit->y()));
			}

			for (int i = 1; i < pts.size() - 1; ++i) {
				rasterizeTriangle(glm::vec3(pts[0], 0), glm::vec3(pts[i], 0), glm::vec3(pts[i+1], 0));
			}
		}
	} else {
		for (int i = 1; i < vertices.size() - 1; ++i) {
			rasterizeTriangle(camera, vertices[0].position, vertices[i].position, vertices[i + 1].position);
		}
	}

	for (int i = 0; i < vertices.size(); ++i) {
		int next = (i + 1) % vertices.size();

		Draw3DStroke(camera, vertices[i].position, vertices[next].position, seed);
	}
}

/**
 * １つの三角形の領域を、背景色で塗りつぶす。
 */
void FrameBuffer::rasterizeTriangle(Camera* camera, const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2) {
	// if the area is too small, skip this triangle.
	if (glm::length(glm::cross(p1 - p0, p2 - p0)) < 1e-7) return;

	bool isFront = false;
	glm::vec3 pp0, pp1, pp2;
	if (!camera->Project(p0, pp0)) return;
	if (!camera->Project(p1, pp1)) return;
	if (!camera->Project(p2, pp2)) return;

	pp0 = convertScreenCoordinate(pp0);
	pp1 = convertScreenCoordinate(pp1);
	pp2 = convertScreenCoordinate(pp2);

	rasterizeTriangle(pp0, pp1, pp2);
}

void FrameBuffer::rasterizeTriangle(const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2) {
	AABB box;

	// compute the bounding box
	box.AddPoint(p0);
	box.AddPoint(p1);
	box.AddPoint(p2);

	// the bounding box should be inside the screen
	int u_min = (int)(box.minCorner().x + 0.5f);
	if (u_min < 0) u_min = 0;;
	int u_max = (int)(box.maxCorner().x - 0.5f);
	if (u_max >= w) u_max = w - 1;
	int v_min = (int)(box.minCorner().y + 0.5f);
	if (v_min < 0) v_min = 0;
	int v_max = (int)(box.maxCorner().y - 0.5f);
	if (v_max >= h) v_max = h - 1;

	float denom = (p1.y - p0.y) * (p2.x - p0.x) - (p1.x - p0.x) * (p2.y - p0.y);

	for (int u = u_min; u <= u_max; u++) {
		for (int v = v_min; v <= v_max; v++) {
			glm::vec3 p(u + 0.5f, v + 0.5f, 0.0f);
			
			float s = ((p2.x - p0.x) * (p.y - p0.y) - (p2.y - p0.y) * (p.x - p0.x)) / denom;
			float t = ((p1.y - p0.y) * (p.x - p0.x) - (p1.x - p0.x) * (p.y - p0.y)) / denom;

			// if the point is outside the triangle, skip it.
			if (s < 0 || s > 1 || t < 0 || t > 1 || s + t > 1) continue;

			// interpolate in screen space
			p.z = p0.z * (1.0f - s - t) + p1.z * s + p2.z * t;

			// if the point is behind the camera, skip this pixel.
			if (p.z < 0) continue;
			
			// check if the point is occluded by other triangles.
			//if (zb[(h-1-v)*w+u] <= p.z) continue;

			// set bg color
			Set(u, v, clear_color, p.z);
		}
	}
}

float FrameBuffer::maxDepth(Camera* camera, const std::vector<Vertex>& vertices) {
	float max_z = 0.0f;

	for (int i = 0; i < vertices.size(); ++i) {
		glm::vec3 pp;
		camera->Project(vertices[i].position, pp);
		if (pp.z > max_z) {
			max_z = pp.z;
		}
	}

	return max_z;
}

unsigned int FrameBuffer::GetColor(const glm::vec3& clr) const {
	unsigned int ret = 0xFF000000;

	int red = (int) (clr.x * 255.0f);
	if (red > 255) red = 255;

	int green = (int) (clr.y * 255.0f);
	if (green > 255) green = 255;
	if (green < 0) green = 0;

	int blue = (int) (clr.z * 255.0f);
	if (blue > 255) blue = 255;
	if (blue < 0) blue = 0;

	return 0xFF000000 + red + (green << 8) + (blue << 16);
}

glm::vec3 FrameBuffer::convertScreenCoordinate(const glm::vec3& p) const {
	glm::vec3 a;
	a.x = w * 0.5 + p.x * w * 0.5;
	a.y = h * 0.5 - p.y * h * 0.5;
	a.z = p.z;

	return a;
}
