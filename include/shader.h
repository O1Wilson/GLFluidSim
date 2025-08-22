#ifndef SHADER_H
#define SHADER_H

#include <glad/glad.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

class Shader {
public:
	unsigned int ID;

	Shader(std::vector<std::pair<const char*, unsigned int>> shaders) {
		ID = glCreateProgram();
		std::vector<unsigned int> shaderIDs;

		for (const auto& shaderData : shaders) {
			unsigned int shaderID = createAndCompileShader(shaderData.first, shaderData.second);
			glAttachShader(ID, shaderID);
			shaderIDs.push_back(shaderID);
		}

		glLinkProgram(ID);
		checkCompileErrors(ID, "PROGRAM");

		for (unsigned int shaderID : shaderIDs) {
			glDetachShader(ID, shaderID);
			glDeleteShader(shaderID);
		}
	}

	void use() {
		glUseProgram(ID);
	}

	void setBool(const std::string& name, bool value) const {
		glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value);
	}
	void setInt(const std::string& name, int value) const {
		glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
	}
	void setFloat(const std::string& name, float value) const {
		glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
	}
	void setVec2(const std::string& name, float x, float y) const {
		glUniform2f(glGetUniformLocation(ID, name.c_str()), x, y);
	}

private:
	unsigned int createAndCompileShader(const char* shaderPath, unsigned int shaderType) {
		std::string shaderCode;
		std::ifstream shaderFile;
		shaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
		try {
			shaderFile.open(shaderPath);
			std::stringstream shaderStream;
			shaderStream << shaderFile.rdbuf();
			shaderFile.close();
			shaderCode = shaderStream.str();
		}
		catch (std::ifstream::failure e) {
			std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ: " << shaderPath << std::endl;
		}
		const char* sShaderCode = shaderCode.c_str();

		// shader compilation happens here
		unsigned int shaderID = glCreateShader(shaderType);
		glShaderSource(shaderID, 1, &sShaderCode, NULL);
		glCompileShader(shaderID);

		std::string type;
		switch (shaderType) {
			case GL_VERTEX_SHADER: type = "VERTEX"; break;
			case GL_FRAGMENT_SHADER: type = "FRAGMENT"; break;
			case GL_COMPUTE_SHADER: type = "COMPUTE"; break;
			default: type = "UNKOWN"; break;
		}
		checkCompileErrors(shaderID, type);

		return shaderID;
	}

private:
	void checkCompileErrors(unsigned int shader, std::string type) {
		int success;
		char infoLog[1024];
		if (type != "PROGRAM") {
			glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
			if (!success) {
				glGetShaderInfoLog(shader, 1024, NULL, infoLog);
				std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
			}
		}
		else {
			glGetProgramiv(shader, GL_LINK_STATUS, &success);
			if (!success) {
				glGetProgramInfoLog(shader, 1024, NULL, infoLog);
				std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
			}
		}
	}
};

#endif