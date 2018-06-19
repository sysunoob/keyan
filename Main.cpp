#define GLM_FORCE_RADIANS
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>


#include "Shader.h"


const unsigned int W = 1000;
const unsigned int H = 1000;

std::string paraname;
int NC, NR, NS;
int MODE;
int NCSTART, NRSTART, NSSTART;
int NX, NY, NZ;
float XLEN, YLEN, ZLEN;
int ALPHA, BETA, GAMMA;
int MAPC, MAPR, MAPS;
float AMIN, AMAX, AMEAN;
int ISPG, NSYMBT, LSKFLG;
int SKWMAT, SKWTRN;

// voxel num
const int VN = 155;

float volumeData[VN][VN][VN];

int drawcnt = 0;
float drawData[20000005];

std::ifstream data_in;
std::string file_in = "EMD-2788.txt";
float isovalue = 0.16;

struct XYZ {
	float x, y, z;
};

struct GRID {
	XYZ p[8];
	float val[8];
	XYZ grad[8];
} voxels[VN][VN][VN];

struct TRIANGLE {
	XYZ p[3];
};

/*struct GRAD {
	XYZ grad[8];
} grads[VN][VN][VN];*/

std::vector<TRIANGLE> Trigs; 
std::vector<TRIANGLE> Norms;

unsigned int VAO;
unsigned int VBO;

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

void readData();
void initTestData();
void initData();
void visualizeData();
void setVoxelPoint(GRID &g, int np, int nx, int ny, int nz, int nvi, int nvj, int nvk);
void marchingTetra(GRID g, float iso, int v0, int v1, int v2, int v3);
XYZ interpVertex(float iso, XYZ p1, XYZ p2, float val1, float val2);
void processInterp(float iso, GRID g, int idx1, int idx2, int idx3, int idx4, int idx5, int idx6);

glm::vec3 targetPos = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 200.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);


int main() {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(W, H, "MTAlgorithm", NULL, NULL);
	if (window == NULL) {
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetScrollCallback(window, scroll_callback);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	//initTestData();
	initData();

	visualizeData();

	for (int i = 0; i < Trigs.size(); i ++) {
		drawData[i * 18 + 0] = Trigs[i].p[0].x;
		drawData[i * 18 + 1] = Trigs[i].p[0].y;
		drawData[i * 18 + 2] = Trigs[i].p[0].z;
		drawData[i * 18 + 6] = Trigs[i].p[1].x;
		drawData[i * 18 + 7] = Trigs[i].p[1].y;
		drawData[i * 18 + 8] = Trigs[i].p[1].z;
		drawData[i * 18 + 12] = Trigs[i].p[2].x;
		drawData[i * 18 + 13] = Trigs[i].p[2].y;
		drawData[i * 18 + 14] = Trigs[i].p[2].z;

		drawData[i * 18 + 3] = Norms[i].p[0].x;
		drawData[i * 18 + 4] = Norms[i].p[0].y;
		drawData[i * 18 + 5] = Norms[i].p[0].z;
		drawData[i * 18 + 9] = Norms[i].p[1].x;
		drawData[i * 18 + 10] = Norms[i].p[1].y;
		drawData[i * 18 + 11] = Norms[i].p[1].z;
		drawData[i * 18 + 15] = Norms[i].p[2].x;
		drawData[i * 18 + 16] = Norms[i].p[2].y;
		drawData[i * 18 + 17] = Norms[i].p[2].z;

		/*for (int j = i * 9; j < i * 9 + 9; j++) {
			printf("%f ", drawData[j]);
			std::cout << std::endl;
		}
		std::cout << std::endl;*/
	}
	drawcnt = Trigs.size() * 18;
	std::cout << drawcnt << std::endl;

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, drawcnt*sizeof(float), drawData, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	glEnable(GL_DEPTH_TEST);

	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	Shader shader("Shader.vs", "Shader.fs");

	glm::mat4 model, model1, model2, model3;
	glm::mat4 viewmodel;
	model1 = glm::translate(model1, glm::vec3(float(NC) / -2.0f, float(NR) / -2.0f, -float(NS) / 2));

	cameraPos = glm::vec3(0, 0, 200.0f);
	//cameraPos = glm::vec3(float(NC) / 2.0f, float(NR) / 2.0f, float(NS) + 100.0f);
	glm::vec3 lightPos = glm::vec3(0, 0, 180);
	
	//glEnable(GL_CULL_FACE);
	//glCullFace(GL_BACK);
	//viewmodel = glm::rotate(model2, glm::radians(0.5f), glm::vec3(0.0f, 1.0f, 0.0f));

	while (!glfwWindowShouldClose(window)) {
		processInput(window);

		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		shader.use();

		glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)W / (float)H, 0.1f, 1000.0f);
		shader.setMat4("projection", projection);

		glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
		shader.setMat4("view", view);

		lightPos = glm::vec3(viewmodel * glm::vec4(lightPos, 1.0f));
		shader.setVec3("lightPos", lightPos);
		shader.setVec3("viewPos", cameraPos);
		
		//model2 = glm::rotate(model2, glm::radians(0.5f), glm::vec3(0.0f, 1.0f, 0.0f));
		//model3 = glm::translate(model3, glm::vec3(float(NC) / 2.0f, float(NR) / 2.0f, float(NS) / 2));

		model = model3 * model2 * model1;
		shader.setMat4("model", model);

		glBindVertexArray(VAO);
		glDrawArrays(GL_TRIANGLES, 0, drawcnt);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();
	return 0;
}

void processInput(GLFWwindow *window) {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	float cameraSpeed = 0.5f; // adjust accordingly
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		cameraPos += cameraSpeed * cameraUp;
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		cameraPos -= cameraSpeed * cameraUp;
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
}

void readData() {
	data_in >> paraname; // file_name
	
	data_in >> paraname >> NC;
	data_in >> paraname >> NR;
	data_in >> paraname >> NS;

	/*std::cout << NC << std::endl;
	std::cout << NR << std::endl;
	std::cout << NS << std::endl;*/

	data_in >> paraname >> MODE;
	data_in >> paraname >> NCSTART;
	data_in >> paraname >> NRSTART;
	data_in >> paraname >> NSSTART;
	data_in >> paraname >> NX;
	data_in >> paraname >> NY;
	data_in >> paraname >> NZ;
	data_in >> paraname >> XLEN;
	data_in >> paraname >> YLEN;
	data_in >> paraname >> ZLEN;
	data_in >> paraname >> ALPHA;
	data_in >> paraname >> BETA;
	data_in >> paraname >> GAMMA;
	data_in >> paraname >> MAPC;
	data_in >> paraname >> MAPR;
	data_in >> paraname >> MAPS;
	data_in >> paraname >> AMIN;
	data_in >> paraname >> AMAX;
	data_in >> paraname >> AMEAN;
	data_in >> paraname >> ISPG;
	data_in >> paraname >> NSYMBT;
	data_in >> paraname >> LSKFLG;

	// SKWMAT
	data_in >> paraname;
	for (int i = 0; i < 9; i++) {
		data_in >> SKWMAT;
	}

	// SKWTRN
	data_in >> paraname;
	for (int i = 0; i < 3; i++) {
		data_in >> SKWTRN;
	}

	// DATA
	data_in >> paraname;
	//std::cout << paraname << std::endl;
	for (int i = 0; i < NC; i++) {
		for (int j = 0; j < NR; j++) {
			for (int k = 0; k < NS; k++) {
				data_in >> volumeData[i][j][k];
				//if (volumeData[i][j][k] != 0) std::cout << volumeData[i][j][k] << std::endl;
			}
		}
	}
}

void initTestData() {
	NC = NR = NS = 3;
	NCSTART = NRSTART = NSSTART = 0;
	memset(volumeData, 0, sizeof(volumeData));
	volumeData[1][1][1] = 2.0f;
}

void initData() {
	data_in.open(file_in, std::ios::in);
	if (data_in.is_open()) {
		readData();
	}
	//isovalue = 0.025;
}

void visualizeData() {
	for (int i = 0; i < NC-1; i++) {
		for (int j = 0; j < NR-1; j++) {
			for (int k = 0; k < NS-1; k++) {

				GRID &g_base = voxels[i][j][k];
				int nx_base = NCSTART + i;
				int ny_base = NRSTART + j;
				int nz_base = NSSTART + k;

				setVoxelPoint(g_base, 0, nx_base, ny_base, nz_base, i, j, k);
				setVoxelPoint(g_base, 1, nx_base + 1, ny_base, nz_base, i+1, j, k);
				setVoxelPoint(g_base, 2, nx_base + 1, ny_base, nz_base + 1, i + 1, j, k + 1);
				setVoxelPoint(g_base, 3, nx_base, ny_base, nz_base + 1, i, j, k + 1);
				setVoxelPoint(g_base, 4, nx_base, ny_base + 1, nz_base, i, j + 1, k);
				setVoxelPoint(g_base, 5, nx_base + 1, ny_base + 1, nz_base, i + 1, j + 1, k);
				setVoxelPoint(g_base, 6, nx_base + 1, ny_base + 1, nz_base + 1, i + 1, j + 1, k + 1);
				setVoxelPoint(g_base, 7, nx_base, ny_base + 1, nz_base + 1, i, j + 1, k + 1);

				marchingTetra(g_base, isovalue, 0, 2, 3, 7);
				marchingTetra(g_base, isovalue, 0, 2, 6, 7);
				marchingTetra(g_base, isovalue, 0, 4, 6, 7);
				marchingTetra(g_base, isovalue, 0, 6, 1, 2);
				marchingTetra(g_base, isovalue, 0, 6, 1, 4);
				marchingTetra(g_base, isovalue, 5, 6, 1, 4);
			}
		}
	}
}

void setVoxelPoint(GRID &g, int np, int nx, int ny, int nz, int nvi, int nvj, int nvk) {
	// printf("%d %d %d %d %d %d\n", nx, ny, nz, nvi, nvj, nvk);
	g.p[np].x = nx;
	g.p[np].y = ny;
	g.p[np].z = nz;
	g.val[np] = volumeData[nvi][nvj][nvk];

	if (nvi + 1 >= NC || nvi - 1 < 0 || nvj + 1 >= NR || nvj - 1 < 0 || nvk + 1 >= NS || nvk - 1 < 0) {
		return;
	}
	g.grad[np].x = (volumeData[nvi + 1][nvj][nvk] - volumeData[nvi - 1][nvj][nvk]) / -1;
	g.grad[np].y = (volumeData[nvi][nvj + 1][nvk] - volumeData[nvi][nvj - 1][nvk]) / -1;
	g.grad[np].z = (volumeData[nvi][nvj][nvk + 1] - volumeData[nvi][nvj][nvk - 1]) / -1;
}

void marchingTetra(GRID g, float iso, int v0, int v1, int v2, int v3) {
	int index = 0;

	if (g.val[v0] < iso) index |= 1;
	if (g.val[v1] < iso) index |= 2;
	if (g.val[v2] < iso) index |= 4;
	if (g.val[v3] < iso) index |= 8;

	switch (index) {
	case 0x00:
	case 0x0F:
		break;

	case 0x01:
	case 0x0E:
		processInterp(iso, g, v0, v3, v0, v1, v0, v2);
		break;

	case 0x02:
	case 0x0D:
		processInterp(iso, g, v0, v1, v1, v2, v1, v3);
		break;

	case 0x03:
	case 0x0C:
		processInterp(iso, g, v0, v3, v1, v3, v0, v2);
		processInterp(iso, g, v2, v1, v1, v3, v0, v2);
		break;

	case 0x04:
	case 0x0B:
		processInterp(iso, g, v2, v3, v1, v2, v0, v2);
		break;

	case 0x05:
	case 0x0A:
		processInterp(iso, g, v0, v3, v2, v3, v1, v2);
		processInterp(iso, g, v0, v3, v1, v0, v1, v2);
		break;

	case 0x06:
	case 0x09:
		processInterp(iso, g, v0, v2, v2, v3, v1, v3);
		processInterp(iso, g, v0, v2, v0, v1, v1, v3);
		break;

	case 0x07:
	case 0x08:
		processInterp(iso, g, v0, v3, v2, v3, v1, v3);
		break;
	}

}

XYZ interpVertex(float iso, XYZ p1, XYZ p2, float val1, float val2) {
	XYZ ans;
	float itpVtxPos = (iso - val1) / (val2 - val1);
	ans.x = p1.x + itpVtxPos * (p2.x - p1.x);
	ans.y = p1.y + itpVtxPos * (p2.y - p1.y);
	ans.z = p1.z + itpVtxPos * (p2.z - p1.z);
	return ans;
}

void processInterp(float iso, GRID g, int idx1, int idx2, int idx3, int idx4, int idx5, int idx6) {
	TRIANGLE tri;
	TRIANGLE nor;

	tri.p[0] = interpVertex(iso, g.p[idx1], g.p[idx2], g.val[idx1], g.val[idx2]);
	tri.p[1] = interpVertex(iso, g.p[idx3], g.p[idx4], g.val[idx3], g.val[idx4]);
	tri.p[2] = interpVertex(iso, g.p[idx5], g.p[idx6], g.val[idx5], g.val[idx6]);
	Trigs.push_back(tri);
	nor.p[0] = interpVertex(iso, g.grad[idx1], g.grad[idx2], g.val[idx1], g.val[idx2]);
	nor.p[1] = interpVertex(iso, g.grad[idx3], g.grad[idx4], g.val[idx3], g.val[idx4]);
	nor.p[2] = interpVertex(iso, g.grad[idx5], g.grad[idx6], g.val[idx5], g.val[idx6]);
	Norms.push_back(nor);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
	cameraPos += 2 * float(yoffset) * cameraFront;
}
