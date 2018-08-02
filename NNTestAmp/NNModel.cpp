//--------------------------------------------------------------------------------------
// By XU, Tianchen
//--------------------------------------------------------------------------------------

#include "NNModel.h"
//#include "Common\amp_vector_math.h"
#include <iostream>
#include <cassert>

using namespace std;
using namespace concurrency;
using namespace precise_math;

NNModel::NNModel(uint32_t uLayerCount) :
	m_uLayerCount(uLayerCount),
	m_vFeatureMaps(uLayerCount),
	m_vWeights(uLayerCount),
	m_vBias(uLayerCount),
	m_layerData(uLayerCount)
{
}

NNModel::~NNModel()
{
}

void NNModel::SetInputImages(spAmpImage2DArray<float> &pImages)
{
	m_pImages = pImages;
	m_layerData[0].m_uImageCount = pImages->get_extent()[0];
}

void NNModel::SetInputImages(const float *pImages, int iWidth, int iHeight, int iNum)
{
	m_pImages = make_shared<AmpImage2DArray<float>>(iNum, iHeight, iWidth, pImages);
	m_layerData[0].m_uImageCount = iNum;
}

void NNModel::SetInputImages(const vector<float> &vImages, int iWidth, int iHeight, int iNum)
{
	m_pImages = make_shared<AmpImage2DArray<float>>(iNum, iHeight, iWidth, vImages.cbegin());
	m_layerData[0].m_uImageCount = iNum;
}

void NNModel::SetFeatureMaps(const FeatureMapInfo *pMapInfo)
{
	for (auto i = 0u; i < m_uLayerCount; ++i)
		SetFeatureMap(i, pMapInfo[i].m_uWidth, pMapInfo[i].m_uHeight, pMapInfo[i].m_uNum);
}

void NNModel::SetFeatureMaps(const vector<FeatureMapInfo> &vMapInfo)
{
	for (auto i = 0u; i < m_uLayerCount; ++i)
		SetFeatureMap(i, vMapInfo[i].m_uWidth, vMapInfo[i].m_uHeight, vMapInfo[i].m_uNum);
}

void NNModel::SetKernels(vpAmpImage2DArray<float> &vImgWeights, vpAmpArray<float> &vImgBias)
{
	m_vWeights = vImgWeights;
	m_vBias = vImgBias;
	
	m_uLayerCount = static_cast<uint32_t>(vImgWeights.size());
	assert(vImgWeights.size() == vImgBias.size());

	m_layerData.resize(m_uLayerCount);

	for (auto i = 0u; i < m_uLayerCount; ++i)
		m_layerData[i].m_uKernelSize = vImgWeights[i]->get_extent()[1];
}

void NNModel::SetKernels(const float *const *ppWeights, const float *const *ppBias,
	const int *pKernelSizes, const uint32_t *pStrides)
{
	for (auto i = 0u; i < m_uLayerCount; ++i)
		SetKernel(i, ppWeights[i], ppBias[i], pKernelSizes[i], pStrides ? pStrides[i] : 2);
}

void NNModel::SetKernels(const vector<const float*> &vpWeights, const vector<const float*> &vpBias,
	const int *pKernelSizes, const uint32_t *pStrides)
{
	SetKernels(vpWeights.data(), vpBias.data(), pKernelSizes, pStrides);
}

void NNModel::SetKernels(const vector<float> *pvWeights, const vector<float> *pvBias,
	const int *pKernelSizes, const uint32_t *pStrides)
{
	for (auto i = 0u; i < m_uLayerCount; ++i)
		SetKernel(i, pvWeights[i], pvBias[i], pKernelSizes[i], pStrides ? pStrides[i] : 2);
}

void NNModel::SetKernels(const vector<vector<float>> &vvWeights, const vector<vector<float>> &vvBias,
	const int *pKernelSizes, const uint32_t *pStrides)
{
	for (auto i = 0u; i < m_uLayerCount; ++i)
		SetKernel(i, vvWeights[i], vvBias[i], pKernelSizes[i], pStrides ? pStrides[i] : 2);
}

void NNModel::SetKernels(const vector<vector<float>> &vvWeights, const vector<vector<float>> &vvBias,
	const vector<int> &vKernelSizes, const vector<int> &vStrides)
{
	for (auto i = 0u; i < m_uLayerCount; ++i)
		SetKernel(i, vvWeights[i], vvBias[i], vKernelSizes[i], vStrides.empty() ? 2 : vStrides[i]);
}

void NNModel::SetMaxPools(const int *pSizes)
{
	for (auto i = 0u; i < m_uLayerCount - 1; ++i) SetMaxPool(i, pSizes[i]);
}

void NNModel::SetMaxPools(const vector<int> &vSizes)
{
	SetMaxPools(vSizes.data());
}

void NNModel::SetFeatureMap(uint32_t i, int iWidth, int iHeight, int iNum)
{
	m_vFeatureMaps[i] = make_shared<AmpImage2DArray<float>>(iNum, iHeight, iWidth);
	m_layerData[i].m_uMapCount = iNum;

	if (++i < m_uLayerCount) m_layerData[i].m_uImageCount = iNum;
}

void NNModel::SetKernel(uint32_t i, spAmpImage2DArray<float> &pImgWeights, spAmpArray<float> &pImgBias)
{
	m_vWeights[i] = pImgWeights;
	m_vBias[i] = pImgBias;

	m_layerData[i].m_uKernelSize = pImgWeights->get_extent()[1];
}

void NNModel::SetKernel(uint32_t i, const float *pWeights, const float *pBias,
	int iKernelSize, uint32_t uStride)
{
	const auto iKernelCount = static_cast<int>(m_layerData[i].m_uImageCount * m_layerData[i].m_uMapCount);

	m_vWeights[i] = make_shared<AmpImage2DArray<float>>(iKernelCount, iKernelSize, iKernelSize, pWeights);
	m_vBias[i] = make_shared<AmpArray<float>>(iKernelCount, pBias);

	m_layerData[i].m_uStride = uStride;
	m_layerData[i].m_uKernelSize = iKernelSize;
}

void NNModel::SetKernel(uint32_t i, const vector<float> &vWeights, const vector<float> &vBias,
	int iKernelSize, uint32_t uStride)
{
	SetKernel(i, vWeights.data(), vBias.data(), iKernelSize, uStride);
}

void NNModel::SetMaxPool(uint32_t i, int iSize)
{
	const auto iNum = m_vFeatureMaps[i]->get_extent()[0];
	const auto iHeight = m_vFeatureMaps[i]->get_extent()[1] / iSize;
	const auto iWidth = m_vFeatureMaps[i]->get_extent()[2] / iSize;

	m_vMaxPools[i] = make_shared<AmpImage2DArray<float>>(iNum, iHeight, iWidth);
	m_layerData[i].m_uMaxPoolSize = iSize;
}

void NNModel::Execute()
{
	for (auto i = 0u; i < m_uLayerCount; ++i) ExecuteLayer(i);
}

void NNModel::ExecuteLayer(uint32_t i, bool bMaxPool)
{
	auto& iNeuronsRW = *m_vFeatureMaps[i];
	const auto ivNeuronsRO = AmpImage2DArrayView<float>(*(i > 0 ? (bMaxPool ? m_vMaxPools : m_vFeatureMaps)[i - 1] : m_pImages));

	const auto ivWeights = AmpImage2DArrayView<float>(*m_vWeights[i]);
	const auto avBias = AmpArrayView<float>(*m_vBias[i]);

	const auto uStride = m_layerData[i].m_uStride;
	const auto uKernelSize = m_layerData[i].m_uKernelSize;
	const auto uImageCount = m_layerData[i].m_uImageCount;

	parallel_for_each(
		// Define the compute domain, which is the set of threads that are created.
		iNeuronsRW.extent,
		// Define the code to run on each thread on the accelerator.
		[=, &iNeuronsRW](const index<3> idx) restrict(amp)
	{
		const auto vWinPos = index<3>(idx * uStride);

		auto fResult = avBias[idx[0]];

		for (auto i = 0u; i < uImageCount; ++i)
		{
			// Convolution
			for (auto j = 0u; j < uKernelSize; ++j)
			{
				for (auto k = 0u; k < uKernelSize; ++k)
				{
					const auto vCKPos = index<3>(idx[0] * uImageCount + i, j, k);
					const auto vPos = vWinPos + vCKPos;
					const auto vNPos = index<3>(i, vPos[1], vPos[2]);
					fResult += ivNeuronsRO[vNPos] * ivWeights[vCKPos];
				}
			}
		}

		// Activation function
		fResult = 1.7159f * tanh(fResult * 2.0f / 3.0f);

		iNeuronsRW[idx] = fResult;
	}
	);
}

void NNModel::MaxPool(uint32_t i)
{
	if (m_layerData[i].m_uMaxPoolSize == 2) maxPool2x2(i);
	// else maxPool(i);
}

void NNModel::PrintLayerData()
{
	for (auto i = 0u; i < m_uLayerCount; ++i)
		cout << "Layer " << i + 1 << " - " <<
			"Stride: " << m_layerData[i].m_uStride << ", " <<
			"Kernel size: " << m_layerData[i].m_uKernelSize << ", " <<
			"Image count: " << m_layerData[i].m_uImageCount << ", " <<
			"Feature map count: " << m_layerData[i].m_uMapCount << endl;
}

vector<float> NNModel::GetResult() const
{
	return *m_vFeatureMaps[m_uLayerCount - 1];
}

void NNModel::maxPool2x2(uint32_t i)
{
	auto& iNeuronsRW = *m_vMaxPools[i];
	const auto ivNeuronsRO = AmpImage2DArrayView<float>(*m_vFeatureMaps[i - 1]);

	// Fast 2x2 down sampling with max-filter
	parallel_for_each(
		// Define the compute domain, which is the set of threads that are created.
		ivNeuronsRO.extent.tile<1, 2, 2>(),
		// Define the code to run on each thread on the accelerator.
		[=, &iNeuronsRW](const tiled_index<1, 2, 2> t_idx) restrict(amp)
	{
		// Group (tile)-shared memory
		tile_static float fPool[2][2];
		auto &fPoolCur = fPool[t_idx.local[1]][t_idx.local[2]];

		fPoolCur = ivNeuronsRO[t_idx];
		t_idx.barrier.wait();

		// Take max in binary
		fPoolCur = t_idx.local[2] ? fmax(fPoolCur, fPool[t_idx.local[1]][t_idx.local[2] - 1]) : fPoolCur;
		fPoolCur = t_idx.local[1] ? fmax(fPoolCur, fPool[t_idx.local[1] - 1][t_idx.local[2]]) : fPoolCur;

		if (t_idx.local[1] && t_idx.local[2])
		{
			iNeuronsRW[t_idx.tile] = fPoolCur;
		}
	}
	);
}
