#pragma once

#include <glm/glm.hpp>

class Camera {
public:
	Camera();

	void mousePress(int mouse_x, int mouse_y);
	void rotate(int mouse_x, int mouse_y);
	void zoom(int mouse_x, int mouse_y);
	void move(int mouse_x, int mouse_y);
	void updatePMatrix(int width,int height);
	void updateMVPMatrix();
	bool Project(const glm::vec3& p, glm::vec3& pp) const;

public:
	float fovy;
	glm::vec3 pos;
	float xrot;
	float yrot;
	float zrot;

	glm::vec2 mouse_pos;

public:
	glm::mat4 mvMatrix;
	glm::mat4 pMatrix;
	glm::mat4 mvpMatrix;
};

