#version 330 core
out vec4 FragColor;
in vec3 Normal;

uniform vec3 lightPos;
uniform vec3 viewPos;
in vec3 FragPos;

void main() {
	vec3 lightColor = vec3(1.0f, 1.0f, 1.0f);
	vec3 objectColor = vec3(0.7f, 0.7f, 0.7f);

	float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

	vec3 norm = normalize(Normal);
	vec3 lightDir = normalize(lightPos - FragPos);
	float diff = max(dot(norm, lightDir), 0);
	vec3 diffuse = diff * lightColor;


	float specularStrength = 1;
	vec3 viewDir = normalize(viewPos - FragPos);
	vec3 halfwayDir = normalize(lightDir + viewDir);
	vec3 reflectDir = reflect(-lightDir, norm);
	float spec = pow(max(dot(norm, halfwayDir), 0.0), 32);
	vec3 specular = specularStrength * spec * lightColor;

	vec3 result = (ambient + diffuse + specular) * objectColor;
	FragColor = vec4(result, 1.0);
}