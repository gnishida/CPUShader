#include <iostream>
#include "GLWidget3D.h"
#include "MainWindow.h"
#include <GL/GLU.h>
#include "GLUtils.h"

#define SQR(x)	((x) * (x))

GLWidget3D::GLWidget3D() {
}

/**
 * This event handler is called when the mouse press events occur.
 */
void GLWidget3D::mousePressEvent(QMouseEvent *e) {
	camera.mousePress(e->x(), e->y());
}

/**
 * This event handler is called when the mouse release events occur.
 */
void GLWidget3D::mouseReleaseEvent(QMouseEvent *e) {
}

/**
 * This event handler is called when the mouse move events occur.
 */
void GLWidget3D::mouseMoveEvent(QMouseEvent *e) {
	if (e->buttons() & Qt::LeftButton) { // Rotate
		camera.rotate(e->x(), e->y());
	} else if (e->buttons() & Qt::MidButton) { // Move
		camera.move(e->x(), e->y());
	} else if (e->buttons() & Qt::RightButton) { // Zoom
		camera.zoom(e->x(), e->y());
	}

	updateGL();
}

/**
 * This function is called once before the first call to paintGL() or resizeGL().
 */
void GLWidget3D::initializeGL() {
	glClearColor(0.443, 0.439, 0.458, 0.0);
	//glClearColor(1, 1, 1, 1);

	glEnable(GL_DEPTH_TEST);
	//glEnable(GL_LIGHTING);
	//glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);

	//static GLfloat lightPosition[4] = {0.0f, 0.0f, 100.0f, 0.0f};
	//glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);

	/*
	std::vector<std::vector<Vertex> > vertices;
	glutils::drawBox(1, 1, 1, glm::vec4(1, 1, 1, 1), glm::mat4(), vertices);
	renderManager.addObject("test", vertices);
	*/

	fb = new FrameBuffer(width(), height());
	fb->loadStrokes("..\\strokes", "..\\strokes_mini\\");
}

/**
 * This function is called whenever the widget has been resized.
 */
void GLWidget3D::resizeGL(int width, int height) {
	/*
	height = height?height:1;

	glViewport( 0, 0, (GLint)width, (GLint)height );
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60, (GLfloat)width/(GLfloat)height, 0.1f, 10000);
	glMatrixMode(GL_MODELVIEW);
	*/

	camera.updatePMatrix(width, height);

	fb->resize(width, height);
}

/**
 * This function is called whenever the widget needs to be painted.
 */
void GLWidget3D::paintGL() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	fb->setClearColor(glm::vec3(1, 1, 1));
	fb->clear();

	std::vector<std::vector<Vertex> > vertices;
	/*
	vertices.resize(2);
	vertices[0].push_back(Vertex(glm::vec3(-1, 0, 0), glm::vec3(0, 0, 1)));
	vertices[0].push_back(Vertex(glm::vec3(1, 0, 0), glm::vec3(0, 0, 1)));
	vertices[0].push_back(Vertex(glm::vec3(0, 1, 0), glm::vec3(0, 0, 1)));
	vertices[1].push_back(Vertex(glm::vec3(-1, 0, 0), glm::vec3(0, 0, 1)));
	vertices[1].push_back(Vertex(glm::vec3(-1, 0, -1), glm::vec3(0, 0, 1)));
	vertices[1].push_back(Vertex(glm::vec3(0, 1, 0), glm::vec3(0, 0, 1)));*/
	glutils::drawBox(1, 1, 1, glm::vec4(1, 1, 1, 1), glm::mat4(), vertices);	
	//glutils::drawBox(2, 2, 2, glm::vec4(1, 1, 1, 1), glm::translate(glm::mat4(), glm::vec3(4, 0, 0)), vertices);	
	fb->rasterize(&camera, vertices);

	//fb->Draw2DSegment(glm::vec3(100, 100, 0), glm::vec3(1, 0, 1), glm::vec3(200, 100, 0), glm::vec3(1, 0, 1));
	//fb->Draw2DStroke(glm::vec3(100, 100, 0), glm::vec3(200, 200, 0), fb->strokes[0]);

	fb->draw();
}

