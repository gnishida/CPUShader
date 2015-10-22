#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include "Camera.h"
#include "Vertex.h"
#include <vector>
#include <opencv/cv.h>
#include <opencv/highgui.h>

// axis aligned bounding box class
class AABB {
private:
	glm::vec3 corners[2];

public:
	AABB();
	AABB(const glm::vec3& p);
	void AddPoint(const glm::vec3& p);
	const glm::vec3& minCorner() const;
	const glm::vec3& maxCorner() const;
	glm::vec3 Size() const;
};

class Stroke {
public:
	cv::Mat stroke_image;

public:
	Stroke(const std::string& filename);
	glm::vec3 getColor(float x, float y) const;
};

class FrameBuffer {
public:
	/** software color buffer (The first pixel is the bottom left corner.) */
	unsigned int *pix;

	/** software Z buffer */
	float *zb;

	/** image wdith resolution */
	int w;
		
	/** image height resolution */
	int h;

	glm::vec3 clear_color;

	std::vector<Stroke> strokes;
	std::vector<Stroke> strokes_mini;

public:
	FrameBuffer(int _w, int _h);
	~FrameBuffer();

	void resize(int _w, int _h);
	void loadStrokes(const std::string& dirname1, const std::string& dirname2);
	void draw();

	void setClearColor(const glm::vec3& clear_color);
	void clear();
	void Set(int u, int v, const glm::vec3& clr, float z);
	void Add(int u, int v, const glm::vec3& color);
	void Draw2DSegment(const glm::vec3& p0, const glm::vec3& c0, const glm::vec3& p1, const glm::vec3& c1);
	void Draw3DSegment(Camera* camera, const glm::vec3& p0, const glm::vec3& c0, const glm::vec3& p1, const glm::vec3& c1);
	void Draw2DStroke(const glm::vec3& p0, const glm::vec3& p1, int stroke_index);
	void Draw3DStroke(Camera* camera, const glm::vec3& p0, const glm::vec3& p1);

	void rasterize(Camera* camera, const std::vector<std::vector<Vertex> >& vertices);
	void rasterizePolygon(Camera* camera, const std::vector<Vertex>& vertices);
	void rasterizeConcavePolygon(Camera* camera, const std::vector<Vertex>& vertices);
	void rasterizeTriangle(Camera* camera, const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2);
	void rasterizeTriangle(const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2);

	float maxDepth(Camera* camera, const std::vector<Vertex>& vertices);

	unsigned int GetColor(const glm::vec3& clr) const;
	glm::vec3 convertScreenCoordinate(const glm::vec3& p) const;
};


