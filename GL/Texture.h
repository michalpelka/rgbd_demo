#pragma once

#include "Renderer.h"
#include <opencv2/opencv.hpp>
class Texture
{
private:
	unsigned int m_RendererID;
    std::string m_FilePath;
	unsigned char * m_LocalBuffer;
	int m_Width, m_Height, m_BPP;
public:
//	Texture(const std::string& path);
    Texture(const cv::Mat& image);
    void update(const cv::Mat& image);

	~Texture();

	void Bind(unsigned int slot = 0) const;
	void Unbind() const;

	inline int GetWidth() const { return m_Width; }
	inline int GetHeight() const { return m_Width; }
    unsigned int getMRendererId() const {
        return m_RendererID;
    }

};