//--------------------------------------------------------------------------------------
// By XU, Tianchen
//--------------------------------------------------------------------------------------

#pragma once

#include <amp.h>  
#include <amp_graphics.h>
#include <amp_math.h>

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

	void SetGroundTruths(spAmpImage2DArray<float> &pImgResults);
	void SetGroundTruths(const float *pResults, int iWidth = 0, int iHeight = 0, int iNum = 0);
	void SetGroundTruths(const std::vector<float> &vResults, int iWidth = 0, int iHeight = 0, int iNum = 0);

	void SetLayers(vpAmpImage2DArray<float> &vImgWeights, vpAmpArray<float> &vImgBias,
		const int *pNumFeatureMaps, const uint32_t *pStrides = nullptr);
	void SetLayers(vpAmpImage2DArray<float> &vImgWeights, vpAmpArray<float> &vImgBias,
		const std::vector<int> &vNumFeatureMaps, const std::vector<uint32_t> &vStrides = std::vector<uint32_t>(0));
	void SetLayers(const float *const *ppWeights, const float * const *ppBias,
		const int *pNumFeatureMaps, const int *pKernelSizes, const uint32_t *pStrides = nullptr);
	void SetLayers(const std::vector<const float*> &vpWeights, const std::vector<const float*> &vpBias,
		const int *pNumFeatureMaps, const int *pKernelSizes, const uint32_t *pStrides = nullptr);
	void SetLayers(const std::vector<float> *pvWeights, const std::vector<float> *pvBias,
		const int *pNumFeatureMaps, const int *pKernelSizes, const uint32_t *pStrides = nullptr);
	void SetLayers(const std::vector<std::vector<float>> &vvWeights, const std::vector<std::vector<float>> &vvBias,
		const int *pNumFeatureMaps, const int *pKernelSizes, const uint32_t *pStrides = nullptr);
	void SetLayers(const std::vector<std::vector<float>> &vvWeights, const std::vector<std::vector<float>> &vvBias,
		const std::vector<int> &vNumFeatureMaps, const std::vector<int> &vKernelSizes,
		const std::vector<int> &vStrides = std::vector<int>(0));

	void SetFeatureMaps(const FeatureMapInfo *pMapInfo);
	void SetFeatureMaps(const std::vector<FeatureMapInfo> &vMapInfo);

	void SetKernels(vpAmpImage2DArray<float> &vImgWeights, vpAmpArray<float> &vImgBias,
		const uint32_t *pStrides = nullptr);
	void SetKernels(vpAmpImage2DArray<float> &vImgWeights, vpAmpArray<float> &vImgBias,
		const std::vector<uint32_t> &vStrides = std::vector<uint32_t>(0));
	void SetKernels(const float *const *ppWeights, const float * const *ppBias,
		const int *pKernelSizes, const uint32_t *pStrides = nullptr);
	void SetKernels(const std::vector<const float*> &vpWeights, const std::vector<const float*> &vpBias,
		const int *pKernelSizes, const uint32_t *pStrides = nullptr);
	void SetKernels(const std::vector<float> *pvWeights, const std::vector<float> *pvBias,
		const int *pKernelSizes, const uint32_t *pStrides = nullptr);
	void SetKernels(const std::vector<std::vector<float>> &vvWeights, const std::vector<std::vector<float>> &vvBias,
		const int *pKernelSizes, const uint32_t *pStrides = nullptr);
	void SetKernels(const std::vector<std::vector<float>> &vvWeights, const std::vector<std::vector<float>> &vvBias,
		const std::vector<int> &vKernelSizes, const std::vector<uint32_t> &vStrides = std::vector<uint32_t>(0));

	void SetMaxPools(const int *pSizes);
	void SetMaxPools(const std::vector<int> &vSizes);

	void SetLayer(uint32_t i, spAmpImage2DArray<float> &pImgWeights, spAmpArray<float> &pImgBias,
		int iNumFeatureMap, uint32_t uStride = 2);
	void SetLayer(uint32_t i, const float *pWeights, const float *pBias,
		int iNumFeatureMap, int iKernelSize, uint32_t uStride = 2);
	void SetLayer(uint32_t i, const std::vector<float> &vWeights, const std::vector<float> &vBias,
		int iNumFeatureMap, int iKernelSize, uint32_t uStride = 2);

	void SetFeatureMap(uint32_t i, int uiWidth, int iHeight, int iNum);
	void SetKernel(uint32_t i, spAmpImage2DArray<float> &pImgWeights, spAmpArray<float> &pImgBias,
		uint32_t uStride = 2);
	void SetKernel(uint32_t i, const float *pWeights, const float *pBias,
		int iKernelSize, uint32_t uStride = 2);
	void SetKernel(uint32_t i, const std::vector<float> &vWeights, const std::vector<float> &vBias,
		int iKernelSize, uint32_t uStride = 2);
	void SetMaxPool(uint32_t i, int iSize);

	void Execute();
	void ExecuteLayer(uint32_t i, bool bMaxPool = false);
	void MaxPool(uint32_t i);

	void BackLayer(uint32_t i);

	void PrintLayerData();

	std::vector<float> GetResult() const;

protected:
	struct ConstData
	{
		uint32_t m_uStride = 2;
		uint32_t m_uKernelSize;
		uint32_t m_uMaxPoolSize;
		
		uint32_t m_uImageCount = 1;	// Input image count (number of input image/neurons from the previous layer)
		uint32_t m_uMapCount;
	};

	concurrency::graphics::int_2 calculateFeatureMapSizeFromKernel(uint32_t i, int iKernelSize, uint32_t uStride) const;

	void maxPool(uint32_t i);
	void maxPool2x2(uint32_t i);

	vpAmpImage2DArray<float>	m_vFeatureMaps;
	vpAmpImage2DArray<float>	m_vMaxPools;
	vpAmpImage2DArray<float>	m_vErrors;

	vpAmpImage2DArray<float>	m_vWeights;
	vpAmpArray<float>			m_vBias;

	spAmpImage2DArray<float>	m_pImages;
	spAmpImage2DArray<float>	m_pGroundTruths;

	uint32_t					m_uLayerCount;
	std::vector<ConstData>		m_layerData;
};
