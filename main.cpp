#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <iostream>

#include <GL/glew.h>

#include <glfw3.h>

GLFWwindow *window;

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>

using namespace glm;


#include <common/shader.hpp>
#include <common/texture.hpp>
#include <common/controls.hpp>
#include <common/lodepng.h>
#include <sstream>

// CPU representation of a particle
struct Particle {
    glm::vec3 pos, old_pos, velocity, deltaP, force;
    unsigned char r, g, b, a; // Color
    float size, angle, weight;
    float cameradistance; // *Squared* distance to the camera. if dead : -1.0f
    std::vector<int> neighbors;
    float lambda;

    bool operator<(const Particle &that) const {
        // Sort in reverse order : far particles drawn first.
        return this->cameradistance > that.cameradistance;
    }
};

class RGB {
public:
    unsigned char R;
    unsigned char G;
    unsigned char B;

    RGB(unsigned char r, unsigned char g, unsigned char b) {
        R = r;
        G = g;
        B = b;
    }

    bool Equals(RGB rgb) {
        return (R == rgb.R) && (G == rgb.G) && (B == rgb.B);
    }
};

class HSL {
public:
    int H;
    float S;
    float L;

    HSL(int h, float s, float l) {
        H = h;
        S = s;
        L = l;
    }

    bool Equals(HSL hsl) {
        return (H == hsl.H) && (S == hsl.S) && (L == hsl.L);
    }
};

const bool SAVE_PHOTOS = false;
int image_counter = 1000;
const int NUM_IMAGES = 1000 + 32 * 5;

const int WATER_r = 0;
const int WATER_g = 191;
const int WATER_b = 255;
const int WATER_a = 255;
const HSL WATER_hsl = HSL(195, 1.0f, 0.5f);
const float WATER_size = 0.05f; // Size of Water Particles

// Dimensions for particle grid
vec3 startHeight = vec3(-0.5f, .0f, -0.5f);
int width(10), height(10), depth(10);
float spacing = 0.1f;
vec3 GRAVITY = vec3(0.0, -9.81, 0.0) * 0.5f;

// Hyperparams
float deltaT = 1.0f / 32.0f;
float h = spacing * 1.01f; // Threshold to consider a particle a neighbor
float rho0 = 0.001f;   // Rest Density, emp: 1
float epsLambda = 6000.0f; // Relaxation Param 6000
float epsVort = 0.000000001f;
float cVisc = 0.0001f;
int NUM_ITERS = 8;

std::vector<Particle> particlesContainer = std::vector<Particle>();

// Spatial hashing
std::unordered_map<float, std::vector<int> > map;

static float HueToRGB(float v1, float v2, float vH) {
    if (vH < 0)
        vH += 1;

    if (vH > 1)
        vH -= 1;

    if ((6 * vH) < 1)
        return (v1 + (v2 - v1) * 6 * vH);

    if ((2 * vH) < 1)
        return v2;

    if ((3 * vH) < 2)
        return (v1 + (v2 - v1) * ((2.0f / 3) - vH) * 6);

    return v1;
}

static RGB HSLToRGB(HSL hsl) {
    unsigned char r = 0;
    unsigned char g = 0;
    unsigned char b = 0;

    if (hsl.S == 0) {
        r = g = b = (unsigned char) (hsl.L * 255);
    } else {
        float v1, v2;
        float hue = (float) hsl.H / 360;

        v2 = (hsl.L < 0.5) ? (hsl.L * (1 + hsl.S)) : ((hsl.L + hsl.S) - (hsl.L * hsl.S));
        v1 = 2 * hsl.L - v2;

        r = (unsigned char) (255 * HueToRGB(v1, v2, hue + (1.0f / 3)));
        g = (unsigned char) (255 * HueToRGB(v1, v2, hue));
        b = (unsigned char) (255 * HueToRGB(v1, v2, hue - (1.0f / 3)));
    }

    return RGB(r, g, b);
}

float norm(vec3 vec) {
    return std::sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

vec3 grad_W_spiky(vec3 pi_min_pj, float h) {
    // ∇W(pi - pj, h) = -(45/(PI*h^6*r))*(h-r)^2 * (pi-pj)
    // TODO: Test Me
    float r = norm(pi_min_pj);

    if (r > 0 and r <= h) {
        float scalar = (float) (-(45.0 / (M_PI * pow(h, 6) * r)) * pow(h - r, 2));
        return scalar * pi_min_pj / r;
    } else {
        return vec3(0.0);
    }
}

float W_poly6(float r, float h) {
    // Poly6 Kernel
    //     W(r,h) = (315/(64*PI*h^9))(h^2-r^2)^3 if 0 <= r <= h
    //     W(r,h) = 0 otherwise
    if (r >= 0 and r <= h) {
        return (float) ((315.0 / (64.0 * M_PI * pow(h, 9))) * pow(h * h - r * r, 3.0));
    } else {
        return 0.0f;
    }
}

void SortParticles() {
//    std::sort(&ParticlesContainer[0], &ParticlesContainer[numParticles]);
    std::sort(particlesContainer.begin(), particlesContainer.end());
}

int initWindow() {
    // Initialise GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        getchar();
        return -1;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Open a window and create its OpenGL context
    window = glfwCreateWindow(1024, 768, "Water Simulation", NULL, NULL);
    if (window == NULL) {
        fprintf(stderr,
                "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
        getchar();
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    glewExperimental = true; // Needed for core profile
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        getchar();
        glfwTerminate();
        return -1;
    }

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    // Hide the mouse and enable unlimited mouvement
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Set the mouse at the center of the screen
    glfwPollEvents();
    glfwSetCursorPos(window, 1024 / 2, 768 / 2);

    // Dark blue background
//	glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    // Enable depth test
    glEnable(GL_DEPTH_TEST);
    // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS);

//    // Enable blending
//    glEnable(GL_BLEND);
//    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    return 0;
}

void flushParticleToGPU(Particle &p, GLfloat *g_particule_position_size_data, GLubyte *g_particule_color_data,
                        glm::vec3 CameraPosition, int ParticlesCount) {
    p.cameradistance = glm::length2(p.pos - CameraPosition);
    //ParticlesContainer[i].pos += glm::vec3(0.0f,10.0f, 0.0f) * (float)delta;

    // Fill the GPU buffer
    g_particule_position_size_data[4 * ParticlesCount + 0] = p.pos.x;
    g_particule_position_size_data[4 * ParticlesCount + 1] = p.pos.y;
    g_particule_position_size_data[4 * ParticlesCount + 2] = p.pos.z;

    g_particule_position_size_data[4 * ParticlesCount + 3] = p.size;

    g_particule_color_data[4 * ParticlesCount + 0] = p.r;
    g_particule_color_data[4 * ParticlesCount + 1] = p.g;
    g_particule_color_data[4 * ParticlesCount + 2] = p.b;
    g_particule_color_data[4 * ParticlesCount + 3] = p.a;
}

float hashFunction(int a, int b, int c) {
    return (float) (1.0 * a + pow(1.0 * b, 2.0) + pow(1.0 * c, 3.0));
}

float spatialHashFunction(int a, int b, int c) {
    int p1 = 73856093, p2 = 19349669, p3 = 83492791;
    int n = (int) ((width * spacing / h) * (height * spacing / h) * (depth * spacing / h));
    return (a * p1) ^ (b * p2) * (c * p3) % n;
}

float getHashValue(vec3 pos) {
    // Hash a 3D position into a unique float identifier that represents
    // membership in some uniquely identified 3D box volume.
    int a = ((int) (pos.x / h));
    int b = ((int) (pos.y / h));
    int c = ((int) (pos.z / h));

    return spatialHashFunction(a, b, c);
}

// Build Map from position to bucket
void buildSpatialMap() {
//    for (const auto &entry : map) {
//        delete (entry.second);
//    }
    map.clear();

    // Build a spatial map out of all of the point masses.
    for (int i = 0; i < particlesContainer.size(); i++) {
        Particle &p = particlesContainer[i];
        float hashVal = getHashValue(p.pos);
        if (!map.count(hashVal)) { // New Key
            map[hashVal] = std::vector<int>();
        }
        map[hashVal].push_back(i);
    }
}

// Creates a list for each particle, of the particles in its bucket and the 27 adjacent buckets
void findNeighboringParticles(Particle &p) {
    // Build a spatial map out of all of the point masses.
    std::vector<int> neighbors = std::vector<int>();
    int centerX = (int) (p.pos.x / h);
    int centerY = (int) (p.pos.y / h);
    int centerZ = (int) (p.pos.z / h);
    for (int x = centerX - 1; x <= centerX + 1; x++) {
        for (int y = centerY - 1; y <= centerY + 1; y++) {
            for (int z = centerZ - 1; z <= centerZ + 1; z++) {
                float hashVal = spatialHashFunction(x, y, z);
                if (map.count(hashVal)) {
                    std::vector<int> bucketVals = map[hashVal];
                    for (int j: bucketVals) {
                        Particle &pj = particlesContainer[j];
                        float dist = norm(p.pos - pj.pos);
                        if (dist > 0 and dist <= h) {
                            neighbors.push_back(j);
                        }
                    }
                }
            }
        }
    }
    p.neighbors = neighbors;
}

float getSPHdensityEstimation(Particle &pi) {
    // SPH Density Estimation = sum_over_j{W(|pi-pj|, h)}
    // W - Poly6 kernel, h - cuttoff distance for particles neighbors
    // TODO: Test Me
    float density = 0.0f;
    for (int j: pi.neighbors) {
        Particle &pj = particlesContainer[j];
        float r = norm(pi.pos - pj.pos);
        density += W_poly6(r, h);
    }
    return density;
}

float C(Particle &pi) {
    // TODO: Test Me
    // Calculate Ci(p1,...,pn) = ϱi - ϱ0
    // Where ϱi = SPH density estimation for point i
    // ϱ0 = rest density
    float rhoi = getSPHdensityEstimation(pi);
    return (rhoi / rho0) - 1;
}

vec3 gradC(Particle &pi) {
    // TODO: Test Me
    // Calculate ∇C = (1/ϱ0)*sum_over_j(grad_W_spiky(pi-pj,h))
    vec3 total = vec3(0.0);
    for (int j: pi.neighbors) {
        Particle &pj = particlesContainer[j];
        total += grad_W_spiky(pi.pos - pj.pos, h);
    }
    return (1.0f / rho0) * total;
}

vec3 gradCi(Particle &pi, Particle &pk) {
    // TODO: Test Me
    // Calculate ∇C = (1/ϱ0)*sum_over_j(grad_W_spiky(pi-pj,h))
    vec3 total = grad_W_spiky(pi.pos - pk.pos, h);
    return (1.0f / rho0) * total;
}

float calculateLambda(Particle &pi) {
    // TODO: Test me
    // Calculate lambdai = -(C(pi)/(sum_over_all_points_pk{|gradC(pk)|^2} + epsilon))
    float denom = epsLambda;
    vec3 gradI = vec3(0.0);
    for (int k: pi.neighbors) {
        Particle &pk = particlesContainer[k];
        vec3 grad2 = gradCi(pi, pk);
        denom += glm::length2(grad2);
        gradI += grad2;
    }
    denom += glm::length2(gradI);
    float c = C(pi);
    return -c / denom;
}

float sCorr(float r) {
    // TODO: Test Me
    // Corrective term for tensile instability
    // sCorr = -k*(W(r,h)/W(q,h))^n
    // Empirical vals: k = 0.001, n = 4, q = 0.01*h
    float k = 0.001f;
    int n = 4;
    float q = 0.01f * h;
    return (float) (-k * pow(W_poly6(r, h) / W_poly6(q, h), n));
}

vec3 getDeltaP(Particle &pi) {
    // TODO: Test me
    // Calculate lambda_i = (1/ϱ0)*sum_over_j{(lambda_i + lambda_j)*grad_W_spiky(pi-pj, h)}
    vec3 deltaP = vec3(0.0);
    for (int j: pi.neighbors) {
        Particle &pj = particlesContainer[j];
        vec3 gradW = grad_W_spiky(pi.pos - pj.pos, h); // TODO: Add sCorr
    }
    return deltaP / rho0;
}

vec3 getEta(Particle &pi, vec3 omega) {
    vec3 eta = vec3(0.0);
    float normOmega = norm(omega);
    for (int j: pi.neighbors) {
        Particle &pj = particlesContainer[j];
        eta += grad_W_spiky(pi.pos - pj.pos, h) * normOmega;
    }
    return eta;
}

vec3 getFVort(Particle &pi) {
    vec3 omega = vec3(0.0);
    for (int j: pi.neighbors) {
        Particle &pj = particlesContainer[j];
        vec3 vij = pi.velocity - pj.velocity;
        vec3 gradW = -1.0f * grad_W_spiky(pi.pos - pj.pos, h);
        omega += glm::cross(vij, gradW);
    }
    vec3 eta = getEta(pi, omega);
    float normEta = norm(eta);
    if (normEta == 0) {
        return vec3(0.0);
    }
    vec3 n = eta / normEta;
    return epsVort * glm::cross(n, omega);
}

vec3 getXSPHViscosity(Particle &pi) {
    vec3 viscosity = vec3(0.0);
    for (int j: pi.neighbors) {
        Particle &pj = particlesContainer[j];
        vec3 vij = pi.velocity - pj.velocity;
        float r = norm(pi.pos - pj.pos);
        viscosity += W_poly6(r, h) * vij;
    }
    return viscosity * cVisc;
}

void projectToSurface(Particle &p) {
    // TODO: Use real bounds
    float floor = -height * spacing;
    float ceiling = height * spacing;
    float leftWall = -width * spacing;
    float rightWall = width * spacing;
    float frontWall = depth * spacing;
    float backWall = -depth * spacing;
    float offset = 0.001f;

    // Check flow
    if (p.pos.y < floor) {
        p.pos.y = floor + offset;
        p.velocity.y = 0;
    } else if (p.pos.y > ceiling) {
        p.pos.y = ceiling - offset;
        p.velocity.y = 0;
    }
    // Check Left and Right Wall
    if (p.pos.x < leftWall) {
        p.pos.x = leftWall + offset;
        p.velocity.x = 0;
    } else if (p.pos.x > rightWall) {
        p.pos.x = rightWall - offset;
        p.velocity.x = 0;
    }
    // Check Front and Back Wall
    if (p.pos.z < backWall) {
        p.pos.z = backWall + offset;
        p.velocity.z = 0;
    } else if (p.pos.z > frontWall) {
        p.pos.z = frontWall - offset;
        p.velocity.z = 0;
    }
}

void updateParticleColor(Particle &p) {
    vec3 v = p.velocity;
    float s = powf(1.0f / (v.x * v.x + v.y * v.y + v.z * v.z), 2);
    HSL hsl = WATER_hsl;
    hsl.L = fmaxf(hsl.L, 1.0f - s);
    RGB rgb = HSLToRGB(hsl);
    p.r = rgb.R;
    p.g = rgb.G;
    p.b = rgb.B;
}

// Handle collisions
void handleAllCollisions() {
    std::vector<int> countBuffer(particlesContainer.size(), 0);
    std::vector<vec3> deltaPBuffer(particlesContainer.size(), vec3(0.0));

    for (int i = 0; i < particlesContainer.size(); i++) {
        Particle &p = particlesContainer[i];
        for (int j: p.neighbors) {
            Particle &pj = particlesContainer[j];
            vec3 dir = p.pos - pj.pos;
            float len = norm(dir);
            vec3 dp;
            if (len > h or len == 0) {
                dp = vec3(0.0);
            } else {
                dp = 0.5f * (len - h) * dir / len;
            }
            deltaPBuffer[i] -= dp;
            deltaPBuffer[j] += dp;

            countBuffer[i]++;
            countBuffer[j]++;

        }
    }
    for (int i = 0; i < particlesContainer.size(); i++) {
        Particle &p = particlesContainer[i];
        int count = countBuffer[i];
        vec3 dp = deltaPBuffer[i];
        if (count > 0) {
            p.pos += dp * (1.0f / count);
        }
    }
}


void initParticleGrid() {
    // Generate 10 new particule each millisecond,
    // but limit this to 16 ms (60 fps), or if you have 1 long frame (1sec),
    // newparticles will be huge and the next frame even longer.
    particlesContainer.clear();
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int z = 0; z < depth; z++) {
                Particle p = Particle();
                p.pos = startHeight + vec3(x, y, z) * spacing;
                p.old_pos = p.pos;
                p.deltaP = vec3(0.0, 0.0, 0.0);
                p.lambda = 0.0;
                p.velocity = vec3(0.0, 0.0, 0.0);
                p.force = GRAVITY;

                p.r = WATER_r;// (rand() % 256)
                p.g = WATER_g;// (rand() % 256)
                p.b = WATER_b; // (rand() % 256)
                p.a = WATER_a;// (rand() % 256)/3

                p.size = WATER_size; // (rand()%1000)/2000.0f + 0.1f
                p.angle = 0.0f;
                p.weight = 1.0f;
                particlesContainer.push_back(p);
            }
        }
    }
}

void initSimpleParticleStack() {
    // Generate 10 new particule each millisecond,
    // but limit this to 16 ms (60 fps), or if you have 1 long frame (1sec),
    // newparticles will be huge and the next frame even longer.
    particlesContainer.clear();

    Particle p1 = Particle();
    Particle p2 = Particle();
    Particle p3 = Particle();

    p1.velocity = vec3(0.0);
    p2.velocity = vec3(0.0);
    p3.velocity = vec3(0.0);

    p1.force = GRAVITY;
    p2.force = GRAVITY;
    p3.force = GRAVITY;

    p1.r = 0;
    p1.g = 255;
    p1.b = 0;
    p1.a = WATER_a;
    p2.r = 255;
    p2.g = 0;
    p2.b = 0;
    p2.a = WATER_a;
    p3.r = 255;
    p3.g = 255;
    p3.b = 255;
    p3.a = WATER_a;

    p1.size = WATER_size;
    p2.size = WATER_size;
    p3.size = WATER_size;

    p1.pos = vec3(0.0);
    p2.pos = vec3(0.0, WATER_size, 0.0);
    p3.pos = vec3(0.0, 2 * WATER_size, 0.0);

    particlesContainer.push_back(p1);
    particlesContainer.push_back(p2);
    particlesContainer.push_back(p3);
}

void initParticleStacks() {
    // Generate 10 new particule each millisecond,
    // but limit this to 16 ms (60 fps), or if you have 1 long frame (1sec),
    // newparticles will be huge and the next frame even longer.
    particlesContainer.clear();

    Particle p1 = Particle();
    Particle p2 = Particle();
    Particle p3 = Particle();
    Particle p4 = Particle();
    Particle p5 = Particle();
    Particle p6 = Particle();

    p1.velocity = vec3(0.0);
    p2.velocity = vec3(0.0);
    p3.velocity = vec3(0.0);
    p4.velocity = vec3(0.0);
    p5.velocity = vec3(0.0);
    p6.velocity = vec3(0.0);

    p1.force = GRAVITY;
    p2.force = GRAVITY;
    p3.force = GRAVITY;
    p4.force = GRAVITY;
    p5.force = GRAVITY;
    p6.force = GRAVITY;

    p1.r = 0;
    p1.g = 255;
    p1.b = 0;
    p1.a = WATER_a;
    p2.r = 255;
    p2.g = 0;
    p2.b = 0;
    p2.a = WATER_a;
    p3.r = 255;
    p3.g = 255;
    p3.b = 255;
    p3.a = WATER_a;
    p4.r = 0;
    p4.g = 255;
    p4.b = 0;
    p4.a = WATER_a;
    p5.r = 255;
    p5.g = 0;
    p5.b = 0;
    p5.a = WATER_a;
    p6.r = 255;
    p6.g = 255;
    p6.b = 255;
    p6.a = WATER_a;

    p1.size = WATER_size;
    p2.size = WATER_size;
    p3.size = WATER_size;
    p4.size = WATER_size;
    p5.size = WATER_size;
    p6.size = WATER_size;

    p1.pos = vec3(0.0);
    p2.pos = vec3(0.0, WATER_size, 0.0);
    p3.pos = vec3(0.0, 2 * WATER_size, 0.0);
    p4.pos = vec3(0.3);
    p5.pos = vec3(0.3, WATER_size, 0.0);
    p6.pos = vec3(0.3, 2 * WATER_size, 0.0);

    particlesContainer.push_back(p1);
    particlesContainer.push_back(p2);
    particlesContainer.push_back(p3);
    particlesContainer.push_back(p4);
    particlesContainer.push_back(p5);
    particlesContainer.push_back(p6);
}

void initParticles() {
    initParticleGrid();
//    initSimpleParticleCircle();
//    initSimpleParticleStack();
//    initParticleStacks();
}

void simulateParticles() {
    // Simulate all particles
    for (Particle &p: particlesContainer) {
        p.velocity += p.force * deltaT;
        updateParticleColor(p);
        p.pos += p.velocity * deltaT;
        projectToSurface(p);
    }

    buildSpatialMap();
    for (Particle &p: particlesContainer) {
        findNeighboringParticles(p);    // 2
    }

    for (int iters = 0; iters < NUM_ITERS; iters++) {
        for (Particle &p: particlesContainer) {
            p.lambda = calculateLambda(p);
        }
        for (Particle &p: particlesContainer) {
            p.deltaP = getDeltaP(p);
        }
        handleAllCollisions();
        for (Particle &p: particlesContainer) {
            p.pos += p.deltaP;
        }
    }
    int i = 0;
    for (Particle &p: particlesContainer) {
        p.velocity = (p.pos - p.old_pos) / deltaT;

        p.force = GRAVITY + getFVort(p);
        p.velocity += getXSPHViscosity(p);
        p.old_pos = p.pos;
        p.deltaP = vec3(0.0); // clear deltaP
    }
}

void savePhoto(std::string &filename, unsigned int width, unsigned int height) {
    void *image = malloc(4 * width * height);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, image);
    unsigned error = lodepng::encode(filename, (const unsigned char *) image, width, height, LCT_RGBA);
    if (error) {
        printf("error: %d\n", error);
    } else {
        std::cout << "Saved to " << filename << std::endl;
    }

    free(image);
}

int main(void) {
    if (initWindow() == -1) {
        return -1;
    }
    initParticles();

    // The framebuffer, which regroups 0, 1, or more textures, and 0 or 1 depth buffer.
    GLuint FramebufferName = 0;
    glGenFramebuffers(1, &FramebufferName);
    glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);

    // The texture we're going to render to
    GLuint renderedTexture;
    glGenTextures(1, &renderedTexture);

    // "Bind" the newly created texture : all future texture functions will modify this texture
    glBindTexture(GL_TEXTURE_2D, renderedTexture);

    // Give an empty image to OpenGL ( the last "0" )
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1024, 768, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);

    // Poor filtering. Needed !
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    // The depth buffer
    GLuint depthrenderbuffer;
    glGenRenderbuffers(1, &depthrenderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 1024, 768);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer);

    // Set "renderedTexture" as our colour attachement #0
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderedTexture, 0);

    // Set the list of draw buffers.
    GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, DrawBuffers); // "1" is the size of DrawBuffers

    // Always check that our framebuffer is ok
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        printf("Something went wrong with the framebuffer!");
        return false;
    }


    GLuint VertexArrayID;
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);

    // Create and compile our GLSL program from the shaders
    GLuint programID = LoadShaders("../shaders/Particle.vert", "../shaders/Particle.frag");
    if (programID == 0) {
        return -1;
    }
    // Vertex shader
    GLuint CameraRight_worldspace_ID = glGetUniformLocation(programID, "CameraRight_worldspace");
    GLuint CameraUp_worldspace_ID = glGetUniformLocation(programID, "CameraUp_worldspace");
    GLuint ViewProjMatrixID = glGetUniformLocation(programID, "VP");

    // fragment shader
    GLuint TextureID = glGetUniformLocation(programID, "myTextureSampler");


    static GLfloat *g_particule_position_size_data = new GLfloat[particlesContainer.size() * 4];
    static GLubyte *g_particule_color_data = new GLubyte[particlesContainer.size() * 4];


    GLuint Texture = loadDDS("../textures/particle.DDS");

    // The VBO containing the 4 vertices of the particles.
    // Thanks to instancing, they will be shared by all particles.
    static const GLfloat g_vertex_buffer_data[] = {
            -0.5f, -0.5f, 0.0f,
            0.5f, -0.5f, 0.0f,
            -0.5f, 0.5f, 0.0f,
            0.5f, 0.5f, 0.0f,
    };
    GLuint billboard_vertex_buffer;
    glGenBuffers(1, &billboard_vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, billboard_vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

    // The VBO containing the positions and sizes of the particles
    GLuint particles_position_buffer;
    glGenBuffers(1, &particles_position_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
    // Initialize with empty (NULL) buffer : it will be updated later, each frame.
    glBufferData(GL_ARRAY_BUFFER, particlesContainer.size() * 4 * sizeof(GLfloat), NULL, GL_STREAM_DRAW);

    // The VBO containing the colors of the particles
    GLuint particles_color_buffer;
    glGenBuffers(1, &particles_color_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, particles_color_buffer);
    // Initialize with empty (NULL) buffer : it will be updated later, each frame.
    glBufferData(GL_ARRAY_BUFFER, particlesContainer.size() * 4 * sizeof(GLubyte), NULL, GL_STREAM_DRAW);


    bool pause = false;
    do {
        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


        computeMatricesFromInputs();
        glm::mat4 ProjectionMatrix = getProjectionMatrix();
        glm::mat4 ViewMatrix = getViewMatrix();

        // We will need the camera's position in order to sort the particles
        // w.r.t the camera's distance.
        // There should be a getCameraPosition() function in common/controls.cpp,
        // but this works too.
        glm::vec3 CameraPosition(glm::inverse(ViewMatrix)[3]);

        glm::mat4 ViewProjectionMatrix = ProjectionMatrix * ViewMatrix;

        if (!pause) {
            simulateParticles(); // TODO: Add back
            // Flush to buffer
            for (int i = 0; i < particlesContainer.size(); i++) {
                Particle &p = particlesContainer[i];
                flushParticleToGPU(p, g_particule_position_size_data, g_particule_color_data, CameraPosition, i);
            }
            SortParticles();
        }



        // Update the buffers that OpenGL uses for rendering.
        // There are much more sophisticated means to stream data from the CPU to the GPU,
        // but this is outside the scope of this tutorial.
        // http://www.opengl.org/wiki/Buffer_Object_Streaming

        glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
        glBufferData(GL_ARRAY_BUFFER, particlesContainer.size() * 4 * sizeof(GLfloat), NULL,
                     GL_STREAM_DRAW); // Buffer orphaning, a common way to improve streaming perf. See above link for details.
        glBufferSubData(GL_ARRAY_BUFFER, 0, particlesContainer.size() * sizeof(GLfloat) * 4,
                        g_particule_position_size_data);

        glBindBuffer(GL_ARRAY_BUFFER, particles_color_buffer);
        glBufferData(GL_ARRAY_BUFFER, particlesContainer.size() * 4 * sizeof(GLubyte), NULL,
                     GL_STREAM_DRAW); // Buffer orphaning, a common way to improve streaming perf. See above link for details.
        glBufferSubData(GL_ARRAY_BUFFER, 0, particlesContainer.size() * sizeof(GLubyte) * 4, g_particule_color_data);


        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // Use our shader
        glUseProgram(programID);

        // Bind our texture in Texture Unit 0
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, Texture);
        // Set our "myTextureSampler" sampler to user Texture Unit 0
        glUniform1i(TextureID, 0);

        // Same as the billboards tutorial
        glUniform3f(CameraRight_worldspace_ID, ViewMatrix[0][0], ViewMatrix[1][0], ViewMatrix[2][0]);
        glUniform3f(CameraUp_worldspace_ID, ViewMatrix[0][1], ViewMatrix[1][1], ViewMatrix[2][1]);

        glUniformMatrix4fv(ViewProjMatrixID, 1, GL_FALSE, &ViewProjectionMatrix[0][0]);

        // 1rst attribute buffer : vertices
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, billboard_vertex_buffer);
        glVertexAttribPointer(
                0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
                3,                  // size
                GL_FLOAT,           // type
                GL_FALSE,           // normalized?
                0,                  // stride
                (void *) 0            // array buffer offset
        );

        // 2nd attribute buffer : positions of particles' centers
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
        glVertexAttribPointer(
                1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
                4,                                // size : x + y + z + size => 4
                GL_FLOAT,                         // type
                GL_FALSE,                         // normalized?
                0,                                // stride
                (void *) 0                          // array buffer offset
        );

        // 3rd attribute buffer : particles' colors
        glEnableVertexAttribArray(2);
        glBindBuffer(GL_ARRAY_BUFFER, particles_color_buffer);
        glVertexAttribPointer(
                2,                                // attribute. No particular reason for 1, but must match the layout in the shader.
                4,                                // size : r + g + b + a => 4
                GL_UNSIGNED_BYTE,                 // type
                GL_TRUE,                          // normalized?    *** YES, this means that the unsigned char[4] will be accessible with a vec4 (floats) in the shader ***
                0,                                // stride
                (void *) 0                          // array buffer offset
        );

        // These functions are specific to glDrawArrays*Instanced*.
        // The first parameter is the attribute buffer we're talking about.
        // The second parameter is the "rate at which generic vertex attributes advance when rendering multiple instances"
        // http://www.opengl.org/sdk/docs/man/xhtml/glVertexAttribDivisor.xml
        glVertexAttribDivisor(0, 0); // particles vertices : always reuse the same 4 vertices -> 0
        glVertexAttribDivisor(1, 1); // positions : one per quad (its center)                 -> 1
        glVertexAttribDivisor(2, 1); // color : one per quad                                  -> 1

        // Draw the particules !
        // This draws many times a small triangle_strip (which looks like a quad).
        // This is equivalent to :
        // for(i in ParticlesCount) : glDrawArrays(GL_TRIANGLE_STRIP, 0, 4),
        // but faster.
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, particlesContainer.size());

        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glDisableVertexAttribArray(2);

        // Render to our framebuffer
        if (SAVE_PHOTOS) {
            glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
            glViewport(0, 0, 1024,
                       768); // Render on the whole framebuffer, complete from the lower left corner to the upper right
            std::string filename = std::to_string(image_counter++) + ".png";
            savePhoto(filename, 1024, 768);
        } else {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }

        // Swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();

        // TODO: Callbacks are buggy and get called multiple times since
        // TODO: glfwGetKey reports last value of key
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {  // Reset when 'R' key is pressed
            initParticles();
            printf("Reset.\n");
        }
        if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS) { // Pause when 'P' pressed
            pause = !pause;
            printf("Paused.\n");
        }
    } // Check if the ESC key was pressed or the window was closed
    while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS
           && glfwWindowShouldClose(window) == 0
           && (!SAVE_PHOTOS || image_counter < NUM_IMAGES));


    delete[]
            g_particule_position_size_data;

// Cleanup VBO and shader
    glDeleteBuffers(1, &particles_color_buffer);
    glDeleteBuffers(1, &particles_position_buffer);
    glDeleteBuffers(1, &billboard_vertex_buffer);
    glDeleteProgram(programID);
    glDeleteTextures(1, &Texture);
    glDeleteVertexArrays(1, &VertexArrayID);


// Close OpenGL window and terminate GLFW
    glfwTerminate();

    return 0;
}

