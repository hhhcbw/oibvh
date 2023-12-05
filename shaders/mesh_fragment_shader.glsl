#version 430 core

out vec4 FragColor;

in vec3 FragPos;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;
uniform vec3 objectColor;
uniform bool wireframe;
void main()
{
    vec3 lightDir = normalize(lightPos - FragPos);
    if (!wireframe)
    {
        FragColor = vec4(vec3(1), 1.0);
    }
    else
    {
        FragColor = vec4(vec3(0), 1.0);
    }
}