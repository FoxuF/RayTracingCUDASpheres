#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"
#include <stdio.h>
#include "math_functions.h"
#include <cmath>
#include <stdio.h>

//Variables punto flotante hay perdida de color

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

struct sphere
{
	//Origen
	float x;
	float y;
	float z;
	//Sphere
	float radio;
	//material
	uchar r;
	uchar g;
	uchar b;
	float difrac;
	float refrac;
};

struct ray
{
	//Origen
	float x;
	float y;
	float z;
	//NORMALIZADOOOOOO!
	float d_x;
	float d_y;
	float d_z;
};

struct light
{
	//Origen
	float x;
	float y;
	float z;
	//Sphere
	float radio;
	//Color n stuff
	float intensity;
	uchar r;
	uchar g;
	uchar b;
};

struct vector3
{
	float x;
	float y;
	float z;
};

__device__ float dotP(vector3 vec1, vector3 vec2) {
	return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
}

__device__ void normalize(vector3* vector) {
	//Normalizar
	float aux = sqrtf(vector->x * vector->x + vector->y * vector->y + vector->z * vector->z);

	vector->x /= aux;
	vector->y /= aux;
	vector->z /= aux;
}

__device__ bool sphereIntersection(ray* ray_test, sphere* obj, float* dist)
{
	float vX, vY, vZ;
	float discrimiante;

	float a, b, c;

	vX = ray_test->x - obj->x;
	vY = ray_test->y - obj->y;
	vZ = ray_test->z - obj->z;

	a = (ray_test->d_x * ray_test->d_x + ray_test->d_y * ray_test->d_y + ray_test->d_z * ray_test->d_z);
	b = 2.0f * (vX * ray_test->d_x + vY * ray_test->d_y + vZ * ray_test->d_z);
	c = (vX * vX + vY * vY + vZ * vZ) - (obj->radio * obj->radio);
	discrimiante = (b * b) - (4 * a * c);
	if (discrimiante < 0.0f)
		return false;
	else
	{
		*dist = (-b - sqrtf(discrimiante)) / (2.0f * a);
		return true;

	}

	return false;

}

__device__ vector3 phongShading(light* luzes, int num_luzes, vector3* point, vector3* normal, vector3* camera, vector3* color)
{
	//Factores de Phong
	float ambiental = 0.2;
	float difuso = 0.5;
	float specular = 0.3f;
	float brillantez = 100;

	vector3 colorSalida;
	colorSalida.x = 0;
	colorSalida.y = 0;
	colorSalida.z = 0;

	// Calcular la contribución de cada luz
	for (int i = 0; i < num_luzes; ++i) {
		light luz = luzes[i];

		// Calcular componente ambiental
		colorSalida.x += color->x * ambiental * luz.intensity;
		colorSalida.y += color->y * ambiental * luz.intensity;
		colorSalida.z += color->z * ambiental * luz.intensity;

		// Calcular componente difusa
		vector3 vec_luz;
		vec_luz.x = luz.x - point->x;
		vec_luz.y = luz.y - point->y;
		vec_luz.z = luz.z - point->z;

		normalize(&vec_luz);

		float doc_prod = dotP(vec_luz, *normal);

		if (doc_prod > 0)
		{
			doc_prod *= difuso * luz.intensity;
			colorSalida.x += doc_prod * color->x;
			colorSalida.y += doc_prod * color->y;
			colorSalida.z += doc_prod * color->z;

			// Calcular componente especular
			vector3 rVect;
			rVect.x = vec_luz.x - 2.0f * (doc_prod)*normal->x;
			rVect.y = vec_luz.y - 2.0f * (doc_prod)*normal->y;
			rVect.z = vec_luz.z - 2.0f * (doc_prod)*normal->z;

			vector3 dirCam;
			dirCam.x = camera->x - point->x;
			dirCam.y = camera->y - point->y;
			dirCam.z = camera->z - point->z;

			normalize(&dirCam);
			//Necesitamos vector que va a la camara. Calculamos el vector que va a la camara.
			float dotVR = dotP(rVect, dirCam);
			dotVR *= powf(dotVR, brillantez) * specular * luz.intensity;

			colorSalida.x += dotVR * color->x;
			colorSalida.y += dotVR * color->y;
			colorSalida.z += dotVR * color->z;
		}
	}

	colorSalida.x = min(255, (int)roundf(colorSalida.x));
	colorSalida.y = min(255, (int)roundf(colorSalida.y));
	colorSalida.z = min(255, (int)roundf(colorSalida.z));

	return colorSalida;
}


__global__ void rayCasting(vector3* camera, light* luz, vector3* pi_corner, uchar* output, sphere* objects, int num_esferas, int num_luz, int width, int heigth, float inc_x, float inc_y)
{
	//columna
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	//fila
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	if (i < width && j < heigth)
	{
		int idx = j * width * 3 + i * 3;

		ray primary;
		// Origen de rayos
		primary.x = camera->x;
		primary.y = camera->y;
		primary.z = camera->z;

		// Ver la esquina izquierda de cada pixel para cada thread
		vector3 dest;
		dest.x = pi_corner->x + inc_x * i;
		dest.y = pi_corner->y - inc_y * j;
		dest.z = 1;

		// Rayo vector de dirección sin normalizar
		primary.d_x = dest.x - primary.x;
		primary.d_y = dest.y - primary.y;
		primary.d_z = dest.z - primary.z;
		// Normalizar rayos
		float aux = sqrtf(primary.d_x * primary.d_x + primary.d_y * primary.d_y + primary.d_z * primary.d_z);
		primary.d_x /= aux;
		primary.d_y /= aux;
		primary.d_z /= aux;

		//llenamos de un color
		float distance = 0; // Distancia para ver donde esta el choque. 
		//Foxu 3 spheres
		float min_dist = 1000;
		vector3 hit_point;
		sphere hit_sphere;
		bool intersected = false; //Si choco con algo es el interPoint
		//Foxu 3 spheres
		// Iterar sobre todas las esferas para encontrar la más cercana
		for (int k = 0; k < num_esferas; k++) {
			if (sphereIntersection(&primary, &objects[k], &distance)) {
				intersected = true;
				if (distance < min_dist) {
					min_dist = distance;
					hit_point.x = primary.d_x * distance + primary.x;
					hit_point.y = primary.d_y * distance + primary.y;
					hit_point.z = primary.d_z * distance + primary.z;
					hit_sphere = objects[k]; //Con que esfera choque. 
				}
			}
		}

		if (intersected) {
			// Calcular vector normal de la intersección del rayo con la esfera.
			vector3 normal;
			normal.x = hit_point.x - hit_sphere.x;
			normal.y = hit_point.y - hit_sphere.y;
			normal.z = hit_point.z - hit_sphere.z;
			// Normalizar el vector
			normalize(&normal);

			// Color de la esferas
			vector3 colorInicio;
			colorInicio.x = hit_sphere.r;
			colorInicio.y = hit_sphere.g;
			colorInicio.z = hit_sphere.b;

			vector3 cameraVect;
			cameraVect.x = camera->x - hit_point.x;
			cameraVect.y = camera->y - hit_point.y;
			cameraVect.z = camera->z - hit_point.z;

			normalize(&cameraVect);

			colorInicio = phongShading(luz, num_luz, &hit_point, &normal, &cameraVect, &colorInicio);
			//Generamos el reflect
			//Vemos si reflect choca con algo
			// Si choca le pasamos colorInicio y le ajustamos el color.
			// Si no choca le ponemos directo colorInicio.
			//

			output[idx] = colorInicio.z; // Blue
			output[idx + 1] = colorInicio.y; // Green
			output[idx + 2] = colorInicio.x; // Red
		}
		else {
			output[idx] = 30;
			output[idx + 1] = 30;
			output[idx + 2] = 30;
		}

	}

}


int main()
{
	//creamos camara
	vector3 camera;
	camera.x = 0;
	camera.y = 0;
	camera.z = 0;

	//Tamano de imagen en pixeles y tamano de plano
	int width = 500, height = 500;
	float tam_imgX = 2, tam_imgY = 2;

	int num_esferas = 3;
	sphere esferas_host[3];
	//Morada
	esferas_host[0].x = 0;
	esferas_host[0].y = 0;
	esferas_host[0].z = 20;
	esferas_host[0].r = 255;
	esferas_host[0].g = 0;
	esferas_host[0].b = 158;
	esferas_host[0].radio = 2;
	//Verde claro
	esferas_host[1].x = 0;
	esferas_host[1].y = -5;
	esferas_host[1].z = 10;
	esferas_host[1].r = 50;
	esferas_host[1].g = 168;
	esferas_host[1].b = 123;
	esferas_host[1].radio = 2;
	//Naranja
	esferas_host[2].x = 0;
	esferas_host[2].y = 5;
	esferas_host[2].z = 10;
	esferas_host[2].r = 200;
	esferas_host[2].g = 100;
	esferas_host[2].b = 50;
	esferas_host[2].radio = 2;
	//////////////////////

	//Creamos una luz en el mundo
	int num_luz = 3;
	light light_host[2];
	light_host[0].x = 25;
	light_host[0].y = 3;
	light_host[0].z = 1;
	light_host[0].radio = 1;
	light_host[0].intensity = 0.5;
	light_host[1].x = 10;
	light_host[1].y = 3;
	light_host[1].z = -1;
	light_host[1].radio = 1;
	light_host[1].intensity = 1;
	//light luz1;
	//luz1.x = 25;
	//luz1.y = 3;
	//luz1.z = 1;
	//luz1.radio = 1;
	//// Segunda Luz
	//light luz2;
	//luz2.x = 5;
	//luz2.y = 3;
	//luz2.z = -1;
	//luz2.radio = 1;


	//esquina superior izquierda de la pantalla
	vector3 esquina_img;
	esquina_img.x = -1;
	esquina_img.y = tam_imgY / 2.0f;
	esquina_img.z = 1;

	/*Calcular el tamano de cuanto va a medir cuanto pixel en el espacio 3d. (Pasar los 500x500 pixeles a los 2x2 del mundo) es el valor de cuanto en cuanto me
	voy moviendo de la esquina*/
	float inc_x = tam_imgX / width;
	float inc_y = tam_imgY / height;

	//agregamos desf al cent
	esquina_img.y -= inc_y / 2.0f;
	esquina_img.z += inc_x / 2.0f;

	//Pasar memoria de cuda
	vector3* camera_dev;
	sphere* esferas_dev;
	vector3* esquina_dev;
	//Imagen destino donde guardar
	uchar* img_dev;
	//Luz destino
	//light* luz_dev;
	//light* luz_dev2;
	light* lights_dev;

	dim3 threads(16, 16);
	dim3 blocks(ceil((float)width / (float)threads.x), ceil((float)height / (float)threads.y));

	cudaMalloc(&camera_dev, sizeof(vector3));
	cudaMalloc(&img_dev, width * height * 3);
	cudaMalloc(&esferas_dev, 3 * sizeof(sphere));
	cudaMalloc(&esquina_dev, sizeof(vector3));
	cudaMalloc(&lights_dev, 2 * sizeof(light));
	//cudaMalloc(&luz_dev, sizeof(light));
	//cudaMalloc(&luz_dev2, sizeof(light));
	// Copiar datos del host al dispositivo

	cudaMemcpy(camera_dev, &camera, sizeof(vector3), cudaMemcpyHostToDevice);
	//cudaMemcpy(esferas_dev, &esf1, sizeof(esfera), cudaMemcpyHostToDevice);
	cudaMemcpy(esferas_dev, esferas_host, 3 * sizeof(sphere), cudaMemcpyHostToDevice);
	cudaMemcpy(esquina_dev, &esquina_img, sizeof(vector3), cudaMemcpyHostToDevice);
	/*cudaMemcpy(luz_dev, &luz1, sizeof(light), cudaMemcpyHostToDevice);
	cudaMemcpy(luz_dev2, &luz2, sizeof(light), cudaMemcpyHostToDevice);*/
	cudaMemcpy(lights_dev, light_host, 2 * sizeof(light), cudaMemcpyHostToDevice);
	rayCasting << <blocks, threads >> > (camera_dev, lights_dev, esquina_dev, img_dev, esferas_dev, num_esferas, num_luz, width, height, inc_x, inc_y);

	cv::Mat frame = cv::Mat(cv::Size(width, height), CV_8UC3);
	cudaMemcpy(frame.ptr(), img_dev, width * height * 3, cudaMemcpyDeviceToHost);

	cv::imshow("salida", frame);
	cv::waitKey(0);

	// Liberar memoria en el dispositivo CUDA
	cudaFree(camera_dev);
	cudaFree(img_dev);
	cudaFree(esferas_dev);
	cudaFree(esquina_dev);
	cudaFree(lights_dev);

	return 0;
}