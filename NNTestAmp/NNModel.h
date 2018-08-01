//--------------------------------------------------------------------------------------
// By XU, Tianchen
//--------------------------------------------------------------------------------------

#pragma once

#include <amp.h>  
#include <amp_graphics.h>
#include <amp_math.h>

using AmpIndex3D = concurrency::index<3>;

template<typename T>
using AmpImage2DArray = concurrency::array<T, 3>;
template<typename T>
using AmpImage2DArrayView = concurrency::array_view<const T, 3>;
template<typename T>
using AmpRWImage2DArrayView = concurrency::array_view<T, 3>;
template<typename T>
using upAmpImage2DArray = std::unique_ptr<AmpImage2DArray<T>>;
template<typename T>
using spAmpImage2DArray = std::shared_ptr<AmpImage2DArray<T>>;
template<typename T>
using vpAmpImage2DArray = std::vector<spAmpImage2DArray<T>>;

template<typename T>
using AmpArray = concurrency::array<T, 1>;
template<typename T>
using AmpArrayView = concurrency::array_view<const T, 1>;
template<typename T>
using AmpRWArrayView = concurrency::array_view<T, 1>;
template<typename T>
using upAmpArray = std::unique_ptr<AmpArray<T>>;
template<typename T>
using spAmpArray = std::shared_ptr<AmpArray<T>>;
template<typename T>
using vpAmpArray = std::vector<spAmpArray<T>>;

class NNModel
{
public:
	struct FeatureMapInfo
	{
		uint32_t m_uNum;
		uint32_t m_uWidth;
		uint32_t m_uHeight;
	};

	NNModel(uint32_t uLayerCount = 0);
	virtual ~NNModel();

	void SetInputImages(spAmpImage2DArray<float> &pImages);
	void SetInputImages(const float *pImages, int iWidth, int iHeight, int iNum = 1);
	void SetInputImages(const std::vector<float> &vImages, int iWidth, int iHeight, int iNum = 1);

	void SetFeatureMaps(const FeatureMapInfo *pMapInfo);
	void SetFeatureMaps(const std::vector<FeatureMapInfo> &vMapInfo);

	void SetKernels(vpAmpImage2DArray<float> &vImgWeights, vpAmpArray<float> &vImgBias);
	void SetKernels(const float *const *ppWeights, const float * const *ppBias,
		const int *pKernelSizes, const uint32_t *pStrides = nullptr);
	void SetKernels(const std::vector<const float*> &vpWeights, const std::vector<const float*> &vpBias,
		const int *pKernelSizes, const uint32_t *pStrides = nullptr);
	void SetKernels(const std::vector<float> *pvWeights, const std::vector<float> *pvBias,
		const int *pKernelSizes, const uint32_t *pStrides = nullptr);
	void SetKernels(const std::vector<std::vector<float>> &vvWeights, const std::vector<std::vector<float>> &vvBias,
		const int *pKernelSizes, const uint32_t *pStrides = nullptr);
	void SetKernels(const std::vector<std::vector<float>> &vvWeights, const std::vector<std::vector<float>> &vvBias,
		const std::vector<int> &vKernelSizes, const std::vector<int> &vStrides = std::vector<int>(0));

	void SetFeatureMap(uint32_t i, int uiWidth, int iHeight, int iNum);
	void SetKernel(uint32_t i, spAmpImage2DArray<float> &pImgWeights, spAmpArray<float> &pImgBias);
	void SetKernel(uint32_t i, const float *pWeights, const float *pBias,
		int iKernelSize, uint32_t uStride = 2);
	void SetKernel(uint32_t i, const std::vector<float> &vWeights, const std::vector<float> &vBias,
		int iKernelSize, uint32_t uStride = 2);

	void Execute();
	void ExecuteLayer(uint32_t i);

	void PrintLayerData();

	std::vector<float> GetResult() const;

protected:
	struct ConstData
	{
		uint32_t m_uStride = 2;
		uint32_t m_uKernelSize;
		
		uint32_t m_uImageCount = 1;	// Input image count (number of input image/neurons from the previous layer)
		uint32_t m_uMapCount;
	};

	vpAmpImage2DArray<float>	m_vFeatureMaps;

	vpAmpImage2DArray<float>	m_vWeights;
	vpAmpArray<float>			m_vBias;

	spAmpImage2DArray<float>	m_pImages;

	uint32_t					m_uLayerCount;
	std::vector<ConstData>		m_layerData;
};
