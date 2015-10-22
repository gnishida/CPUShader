#include "framebuffer.h"
//#include <libtiff/tiffio.h>
#include <iostream>
//#include "scene.h"
#include <math.h>
#include <algorithm>
#include <QGLWidget>

using namespace std;


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


// makes an OpenGL window that supports SW, HW rendering, that can be displayed on screen
//        and that receives UI events, i.e. keyboard, mouse, etc.
FrameBuffer::FrameBuffer(int _w, int _h) {
	w = _w;
	h = _h;
	pix = new unsigned int[w*h];
	zb  = new float[w*h];
}

FrameBuffer::~FrameBuffer() {
	delete [] pix;
	delete [] zb;
}

// rendering callback; see header file comment
void FrameBuffer::draw() {
	// SW window, just transfer computed pixels from pix to HW for display
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
 * This function does not check neigher the range and the zbuffer.
 *
 * @param u		x coordinate of the pixel
 * @param v		y coordinate of the pixel
 * @param clr	the color
 */
void FrameBuffer::Set(int u, int v, const glm::vec3& clr) {
	pix[(h-1-v)*w+u] = GetColor(clr);
}

/**
 * Set one pixel to given color.
 * This function does not check the range, but check the zbuffer.
 *
 * @param u		x coordinate of the pixel
 * @param v		y coordinate of the pixel
 * @param clr	the color
 * @param z		z buffer
 */
void FrameBuffer::Set(int u, int v, const glm::vec3& clr, float z) {
	//if (zb[(h-1-v)*w+u] <= z) return;

	pix[(h-1-v)*w+u] = GetColor(clr);
	zb[(h-1-v)*w+u] = z;
}

/**
 * Set one pixel to given color.
 * If the specified pixel is out of the screen, it does nothing.
 *
 * @param u		x coordinate of the pixel
 * @param v		y coordinate of the pixel
 * @param clr	the color
 */
void FrameBuffer::SetGuarded(int u, int v, const glm::vec3& clr, float z) {
	if (u < 0 || u > w-1 || v < 0 || v > h-1) return;

	Set(u, v, clr, z);
}

// set all z values in SW ZB to z0
/*
void FrameBuffer::SetZB(float z0) {
	for (int i = 0; i < w*h; i++) {
		zb[i] = z0;
	}
}
*/

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
		SetGuarded(u, v, currc, curr.z);
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
 * Draw 2D segment with color interpolation.
 *
 * @param p0	the first point of the segment
 * @param c0	the color of the first point
 * @param p1	the second point of the segment
 * @param c1	the color of the second point
 */
void FrameBuffer::Draw2DStroke(const glm::vec3& p0, const glm::vec3& c0, const glm::vec3& p1, const glm::vec3& c1) {
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
		SetGuarded(u, v, currc, curr.z);
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
void FrameBuffer::Draw3DStroke(Camera* camera, const glm::vec3& p0, const glm::vec3& c0, const glm::vec3& p1, const glm::vec3& c1) {
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

	srand(q0.x * 100 + q0.y * 50 + q0.z * 10 + q1.x * 20 + q1.y * 30 + q1.z * 40);

	glm::vec3 offset = pp0 - pp1;
	offset *= 0.01;
	for (int i = 0; i < 2; ++i) {
		pp0[i] += ((float)rand() / RAND_MAX - 0.5) * 5 + offset[i];
		pp1[i] += ((float)rand() / RAND_MAX - 0.5) * 5 - offset[i];
	}

	Draw2DStroke(pp0, c0, pp1, c1);
}

bool FrameBuffer::isHidden(int u, int v, float z) {
	if (zb[(h-1-v)*w+u] >= z) return true;
	else return false;
}

/**
 * objectを描画する。
 */
void FrameBuffer::rasterize(Camera* camera, const std::vector<std::vector<Vertex> >& vertices) {
	std::multimap<float, std::vector<Vertex> > sortedVertices;

	for (int i = 0; i < vertices.size(); ++i) {
		float depth = maxDepth(camera, vertices[i]);

		sortedVertices.insert(std::make_pair(depth, vertices[i]));
	}

	for (auto it = sortedVertices.rbegin(); it != sortedVertices.rend(); ++it) {
		rasterize(camera, it->second);
	}
}

/**
 * １つのfaceを描画する。
 */
void FrameBuffer::rasterize(Camera* camera, const std::vector<Vertex>& vertices) {
	for (int i = 1; i < vertices.size() - 1; ++i) {
		rasterize(camera, vertices[0], vertices[i], vertices[i + 1]);
	}

	for (int i = 0; i < vertices.size(); ++i) {
		int next = (i + 1) % vertices.size();

		//Draw3DSegment(camera, vertices[i].position, glm::vec3(0, 0, 0), vertices[next].position, glm::vec3(0, 0, 0));
		Draw3DStroke(camera, vertices[i].position, glm::vec3(0, 0, 0), vertices[next].position, glm::vec3(0, 0, 0));
	}
}

/**
 * １つの三角形の描画する。
 */
void FrameBuffer::rasterize(Camera* camera, const Vertex& p0, const Vertex& p1, const Vertex& p2) {
	AABB box;

	// if the area is too small, skip this triangle.
	if (glm::length(glm::cross(p1.position - p0.position, p2.position - p0.position)) < 1e-7) return;

	bool isFront = false;
	glm::vec3 pp0, pp1, pp2;
	if (!camera->Project(p0.position, pp0)) return;
	if (!camera->Project(p1.position, pp1)) return;
	if (!camera->Project(p2.position, pp2)) return;

	pp0 = convertScreenCoordinate(pp0);
	pp1 = convertScreenCoordinate(pp1);
	pp2 = convertScreenCoordinate(pp2);

	// compute the bounding box
	box.AddPoint(pp0);
	box.AddPoint(pp1);
	box.AddPoint(pp2);

	// the bounding box should be inside the screen
	int u_min = (int)(box.minCorner().x + 0.5f);
	if (u_min < 0) u_min = 0;;
	int u_max = (int)(box.maxCorner().x - 0.5f);
	if (u_max >= w) u_max = w - 1;
	int v_min = (int)(box.minCorner().y + 0.5f);
	if (v_min < 0) v_min = 0;
	int v_max = (int)(box.maxCorner().y - 0.5f);
	if (v_max >= h) v_max = h - 1;

	float denom = (pp1.y - pp0.y) * (pp2.x - pp0.x) - (pp1.x - pp0.x) * (pp2.y - pp0.y);

	for (int u = u_min; u <= u_max; u++) {
		for (int v = v_min; v <= v_max; v++) {
			glm::vec3 pp(u + 0.5f, v + 0.5f, 0.0f);
			
			float s = ((pp2.x - pp0.x) * (pp.y - pp0.y) - (pp2.y - pp0.y) * (pp.x - pp0.x)) / denom;
			float t = ((pp1.y - pp0.y) * (pp.x - pp0.x) - (pp1.x - pp0.x) * (pp.y - pp0.y)) / denom;

			// if the point is outside the triangle, skip it.
			if (s < 0 || s > 1 || t < 0 || t > 1 || s + t > 1) continue;

			// interpolate in screen space
			pp.z = pp0.z * (1.0f - s - t) + pp1.z * s + pp2.z * t;

			// if the point is behind the camera, skip this pixel.
			if (pp.z <= 0) continue;
			
			// check if the point is occluded by other triangles.
			//if (zb[(h-1-v)*w+u] <= pp.z) continue;

			// set bg color
			Set(u, v, clear_color, pp.z);
			//Set(u, v, glm::vec3(1, 1, 1), pp.z);
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
