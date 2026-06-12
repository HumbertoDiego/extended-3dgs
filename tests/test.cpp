#include <stdio.h>
#include <algorithm>
#include <glm/glm.hpp>
using namespace std; 

void getCurvatura(const glm::vec3 scale, float mod, float& curvatura) {
    // Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

    // Proposition
    float min_val = min(min(S[0][0], S[1][1]), S[2][2]);
    curvatura = min_val*min_val  / (S[0][0]*S[0][0] + S[1][1]*S[1][1] + S[2][2]*S[2][2]);
}

int main() {
    printf("curvatura...\n");
    float mod = 1.0f;
    float curvatura;
    glm::vec3 scale(1.0f,2.0f,3.0f);
    getCurvatura(scale, mod, curvatura);
    printf("%.9g",curvatura);
    return 0;
}