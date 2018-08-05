//--------------------------------------------------------------------------------------
// By XU, Tianchen
//--------------------------------------------------------------------------------------

#include "NNModel.h"
#include "Common\amp_vector_math.h"
#include <iostream>
#include <cassert>

using namespace std;
using namespace concurrency;
using namespace precise_math;

NNModel::NNModel(uint32_t uLayerCount) :
	m_uLayerCount(uLayerCount),
	m_vFeatureMaps(uLayerCount),
	m_vMaxPools(uLayerCount),
	m_vdEdys(uLayerCount),
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

void NNModel::SetGroundTruths(spAmpImage2DArray<float> &pImgResults)
{
	m_pGroundTruths = pImgResults;
}

void NNModel::SetGroundTruths(const float *pResults, int iWidth, int iHeight, int iNum)
{
	if (m_uLayerCount < 1) return;

	const auto i = m_uLayerCount - 1;
	iNum = iNum > 0 ? iNum : m_vFeatureMaps[i]->get_extent()[0];
	iHeight = iHeight > 0 ? iHeight : m_vFeatureMaps[i]->get_extent()[1];
	iWidth = iWidth > 0 ? iWidth : m_vFeatureMaps[i]->get_extent()[2];

	m_pGroundTruths = make_shared<AmpImage2DArray<float>>(iNum, iHeight, iWidth, pResults);
}

void NNModel::SetGroundTruths(const vector<float> &vResults, int iWidth, int iHeight, int iNum)
{
	SetGroundTruths(vResults.data(), iWidth, iHeight, iNum);
}

void NNModel::SetLayers(vpAmpImage2DArray<float> &vImgWeights, vpAmpArray<float> &vImgBias,
	const int *pNumFeatureMaps, const uint32_t *pStrides)
{
	for (auto i = 0u; i < m_uLayerCount; ++i)
	{
		const auto vMapSize = calculateFeatureMapSizeFromKernel(i, vImgWeights[i]->get_extent()[1], pStrides[i]);
		SetFeatureMap(i, vMapSize.x, vMapSize.y, pNumFeatureMaps[i]);
	}

	SetKernels(vImgWeights, vImgBias, pStrides);
}

void NNModel::SetLayers(vpAmpImage2DArray<float> &vImgWeights, vpAmpArray<float> &vImgBias,
	const vector<int> &vNumFeatureMaps, const vector<uint32_t> &vStrides)
{
	SetLayers(vImgWeights, vImgBias, vNumFeatureMaps.data(), vStrides.empty() ? nullptr : vStrides.data());
}

void NNModel::SetLayers(const float *const *ppWeights, const float *const *ppBias,
	const int *pNumFeatureMaps, const int *pKernelSizes, const uint32_t *pStrides)
{
	for (auto i = 0u; i < m_uLayerCount; ++i)
		SetLayer(i, ppWeights[i], ppBias[i], pNumFeatureMaps[i], pKernelSizes[i], pStrides ? pStrides[i] : 2);
}

void NNModel::SetLayers(const vector<const float*> &vpWeights, const vector<const float*> &vpBias,
	const int *pNumFeatureMaps, const int *pKernelSizes, const uint32_t *pStrides)
{
	SetLayers(vpWeights.data(), vpBias.data(), pNumFeatureMaps, pKernelSizes, pStrides);
}

void NNModel::SetLayers(const vector<vector<float>> &vvWeights, const vector<vector<float>> &vvBias,
	const vector<int> &vNumFeatureMaps, const std::vector<int> &vKernelSizes, const std::vector<int> &vStrides)
{
	for (auto i = 0u; i < m_uLayerCount; ++i)
		SetLayer(i, vvWeights[i], vvBias[i], vNumFeatureMaps[i], vKernelSizes[i], vStrides.empty() ? 2 : vStrides[i]);
}

void NNModel::SetLayers(const vector<float> *pvWeights, const vector<float> *pvBias,
	const int *pNumFeatureMaps, const int *pKernelSizes, const uint32_t *pStrides)
{
	for (auto i = 0u; i < m_uLayerCount; ++i)
		SetLayer(i, pvWeights[i], pvBias[i], pNumFeatureMaps[i], pKernelSizes[i], pStrides ? pStrides[i] : 2);
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

void NNModel::SetKernels(vpAmpImage2DArray<float> &vImgWeights, vpAmpArray<float> &vImgBias,
	const uint32_t *pStrides)
{
	m_vWeights = vImgWeights;
	m_vBias = vImgBias;
	
	m_uLayerCount = static_cast<uint32_t>(vImgWeights.size());
	assert(vImgWeights.size() == vImgBias.size());

	m_layerData.resize(m_uLayerCount);

	for (auto i = 0u; i < m_uLayerCount; ++i)
	{
		m_layerData[i].m_uStride = pStrides[i];
		m_layerData[i].m_uKernelSize = vImgWeights[i]->get_extent()[1];
	}
}

void NNModel::SetKernels(vpAmpImage2DArray<float>& vImgWeights, vpAmpArray<float>& vImgBias,
	const vector<uint32_t>& vStrides)
{
	SetKernels(vImgWeights, vImgBias, vStrides.empty() ? nullptr : vStrides.data());
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
	const vector<int> &vKernelSizes, const vector<uint32_t> &vStrides)
{
	for (auto i = 0u; i < m_uLayerCount; ++i)
		SetKernel(i, vvWeights[i], vvBias[i], vKernelSizes[i], vStrides.empty() ? 2 : vStrides[i]);
}

void NNModel::SetMaxPools(const int *pSizes)
{
	for (auto i = 0u; i + 1 < m_uLayerCount; ++i) SetMaxPool(i, pSizes[i]);
}

void NNModel::SetMaxPools(const vector<int> &vSizes)
{
	SetMaxPools(vSizes.data());
}

void NNModel::SetLayer(uint32_t i, spAmpImage2DArray<float> &pImgWeights, spAmpArray<float> &pImgBias,
	int iNumFeatureMap, uint32_t uStride)
{
	const auto vMapSize = calculateFeatureMapSizeFromKernel(i, pImgWeights->get_extent()[1], uStride);

	SetFeatureMap(i, vMapSize.x, vMapSize.y, iNumFeatureMap);
	SetKernel(i, pImgWeights, pImgBias, uStride);
}

void NNModel::SetLayer(uint32_t i, const float *pWeights, const float *pBias,
	int iNumFeatureMap, int iKernelSize, uint32_t uStride)
{
	const auto vMapSize = calculateFeatureMapSizeFromKernel(i, iKernelSize, uStride);

	SetFeatureMap(i, vMapSize.x, vMapSize.y, iNumFeatureMap);
	SetKernel(i, pWeights, pBias, iKernelSize, uStride);
}

void NNModel::SetLayer(uint32_t i, const vector<float> &vWeights, const vector<float> &vBias,
	int iNumFeatureMap, int iKernelSize, uint32_t uStride)
{
	SetLayer(i, vWeights.data(), vBias.data(), iNumFeatureMap, iKernelSize, uStride);
}

void NNModel::SetFeatureMap(uint32_t i, int iWidth, int iHeight, int iNum)
{
	m_vFeatureMaps[i] = make_shared<AmpImage2DArray<float>>(iNum, iHeight, iWidth);
	m_vdEdys[i] = make_shared<AmpImage2DArray<float>>(iNum, iHeight, iWidth);
	m_layerData[i].m_uMapCount = iNum;

	if (++i < m_uLayerCount) m_layerData[i].m_uImageCount = iNum;
}

void NNModel::SetKernel(uint32_t i, spAmpImage2DArray<float> &pImgWeights, spAmpArray<float> &pImgBias,
	uint32_t uStride)
{
	m_vWeights[i] = pImgWeights;
	m_vBias[i] = pImgBias;

	m_layerData[i].m_uStride = uStride;
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
	for (auto i = 0u; i < m_uLayerCount; ++i)
	{
		Convolution(i);
		MaxPool(i);
	}
}

void NNModel::BackPropagate(float fEta)
{
	if (m_uLayerCount < 1) return;

	// Evaluate the errors(dE / dy) for the final layer
	const auto i = m_uLayerCount - 1;
	auto &idEdysRW = *m_vdEdys[i];
	const auto &iNeuronsRO = *m_vFeatureMaps[i];
	const auto &iGroundTruthsRO = *m_pGroundTruths;

	parallel_for_each(
		// Define the compute domain, which is the set of threads that are created.
		idEdysRW.extent,
		// Define the code to run on each thread on the accelerator.
		[=, &idEdysRW, &iNeuronsRO, &iGroundTruthsRO](const index<3> idx) restrict(amp)
	{
		const auto fx = iNeuronsRO[idx];
		const auto fdEdx = fx - iGroundTruthsRO[idx];

		idEdysRW[idx] = (1.0f - fx * fx) * fdEdx;
	}
	);

	// Back propagate
	for (auto i = 0u; i < m_uLayerCount; ++i) BackConvolution(i, fEta);
}

void NNModel::Convolution(uint32_t i)
{
	const auto uStride = m_layerData[i].m_uStride;
	const auto uKernelSize = m_layerData[i].m_uKernelSize;
	const auto uImageCount = m_layerData[i].m_uImageCount;

	auto &iNeuronsRW = *m_vFeatureMaps[i];
	const auto &iNeuronsRO = *(i > 0 ? (m_layerData[i - 1].m_uMaxPoolSize > 1 ? m_vMaxPools : m_vFeatureMaps)[i - 1] : m_pImages);

	const auto &iWeightsRO = *m_vWeights[i];
	const auto &aBiasRO = *m_vBias[i];

	parallel_for_each(
		// Define the compute domain, which is the set of threads that are created.
		iNeuronsRW.extent,
		// Define the code to run on each thread on the accelerator.
		[=, &iNeuronsRW, &iNeuronsRO, &iWeightsRO, &aBiasRO](const index<3> idx) restrict(amp)
	{
		const auto vWinPos = index<2>(idx[1], idx[2]) * uStride;

		auto fResult = aBiasRO[idx[0]];

		for (auto i = 0u; i < uImageCount; ++i)
		{
			// Convolution
			for (auto j = 0u; j < uKernelSize; ++j)
			{
				for (auto k = 0u; k < uKernelSize; ++k)
				{
					const auto vPos = vWinPos + index<2>(j, k);
					const auto vNeuronPos = index<3>(i, vPos[0], vPos[1]);
					const auto vKernelPos = index<3>(idx[0] * uImageCount + i, j, k);

					fResult += iNeuronsRO[vNeuronPos] * iWeightsRO[vKernelPos];
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
	else if (m_layerData[i].m_uMaxPoolSize > 1) maxPool(i);
}

void NNModel::BackConvolution(uint32_t i, float fEta)
{
	const auto uStride = m_layerData[i].m_uStride;
	const auto uKernelSize = m_layerData[i].m_uKernelSize;
	const auto uImageCount = m_layerData[i].m_uImageCount;
	const auto uMapCount = m_layerData[i].m_uMapCount;
	const auto uMaxPoolSize = i > 0 ? m_layerData[i - 1].m_uMaxPoolSize : 1;

	///////////////////////////////////////////////////////////////
	// Pass 1: Evaluate the errors (dE/dy) for the previous layer
	///////////////////////////////////////////////////////////////
	auto &idEdysRW = *m_vdEdys[i - 1];
	const auto &idEdysRO = *m_vdEdys[i];
	const auto &iWeightsRO = *m_vWeights[i];

	const auto vExtent = concurrency::extent<3>(
		idEdysRW.extent[0],
		idEdysRW.extent[1] / uMaxPoolSize,
		idEdysRW.extent[2] / uMaxPoolSize
		);

	parallel_for_each(
		// Define the compute domain, which is the set of threads that are created.
		vExtent,
		// Define the code to run on each thread on the accelerator.
		[=, &idEdysRW, &idEdysRO, &iWeightsRO](const index<3> idx) restrict(amp)
	{
		auto idx2D = index<2>(idx[1], idx[2]);

		auto fResult = 0.0f;

		for (auto i = 0u; i < uMapCount; ++i)
		{
			// Convolution
			for (auto j = 0u; j < uKernelSize; ++j)
			{
				for (auto k = 0u; k < uKernelSize; ++k)
				{
					const auto vKernelPos2D = index<2>(j, k);
					const auto vPos = (idx2D - vKernelPos2D) / uStride;

					if (vPos * uStride + vKernelPos2D == idx2D)
					{
						const auto vErrorPos = index<3>(i, vPos[0], vPos[1]);
						const auto vKernelPos = index<3>(i * uImageCount + idx[0], j, k);

						fResult += idEdysRO[vErrorPos] * iWeightsRO[vKernelPos];
					}
				}
			}
		}

		// If there is max pooling, up-sampling is neccessary
		idx2D *= uMaxPoolSize;
		for (auto i = 0u; i < uMaxPoolSize; ++i)
		{
			for (auto j = 0u; j < uMaxPoolSize; ++j)
			{
				const auto vPos = idx2D + index<2>(i, j);
				const auto vEPos = index<3>(idx[0], vPos[0], vPos[1]);

				idEdysRW[vEPos] = fResult;
			}
		}
	}
	);

	///////////////////////////////////////////////////////////////
	// Pass 2: Update weights
	///////////////////////////////////////////////////////////////
	auto &iWeightsRW = *m_vWeights[i];
	const auto &iNeuronsRO = *(i > 0 ? (uMaxPoolSize > 1 ? m_vMaxPools : m_vFeatureMaps)[i - 1] : m_pImages);

	parallel_for_each(
		// Define the compute domain, which is the set of threads that are created.
		iWeightsRW.extent,
		// Define the code to run on each thread on the accelerator.
		[=, &iWeightsRW, &idEdysRO, &iNeuronsRO](const index<3> idx) restrict(amp)
	{
		const auto uNeuronIdx = idx[0] % uImageCount;
		const auto uErrorIdx = idx[0] / uImageCount;

		auto fdEdW = 0.0f;

		for (auto i = 0; i < idEdysRO.get_extent()[1]; ++i)
		{
			for (auto j = 0; j < idEdysRO.get_extent()[2]; ++j)
			{
				const auto vWinPos = index<2>(i, j) * uStride;
				const auto vPos = vWinPos + index<2>(idx[1], idx[2]);
				const auto vNeuronPos = index<3>(uNeuronIdx, vPos[0], vPos[1]);
				const auto vErrorPos = index<3>(uErrorIdx, i, j);

				fdEdW += iNeuronsRO[vNeuronPos] * idEdysRO[vErrorPos];
			}
		}
		
		iWeightsRW[idx] -= fEta * fdEdW;
	}
	);
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

int2 NNModel::calculateFeatureMapSizeFromKernel(uint32_t i, int iKernelSize, uint32_t uStride) const
{
	const auto &pImage = i > 0 ? (m_layerData[i - 1].m_uMaxPoolSize > 1 ? m_vMaxPools : m_vFeatureMaps)[i - 1] : m_pImages;
	const auto iHeight = (pImage->get_extent()[1] - iKernelSize) / uStride + 1;
	const auto iWidth = (pImage->get_extent()[2] - iKernelSize) / uStride + 1;

	return int2(iWidth, iHeight);
}

void NNModel::maxPool(uint32_t i)
{
	const auto uMaxPoolSize = m_layerData[i].m_uMaxPoolSize;

	auto& iNeuronsRW = *m_vMaxPools[i];
	const auto &iNeuronsRO = *m_vFeatureMaps[i];

	// nxn down sampling with max-filter
	parallel_for_each(
		// Define the compute domain, which is the set of threads that are created.
		iNeuronsRW.extent,
		// Define the code to run on each thread on the accelerator.
		[=, &iNeuronsRW, &iNeuronsRO](const index<3> idx) restrict(amp)
	{
		const auto vWinPos = index<2>(idx[1], idx[2]) * uMaxPoolSize;

		auto fResult = -FLT_MAX;

		for (auto i = 0u; i < uMaxPoolSize; ++i)
		{
			for (auto j = 0u; j < uMaxPoolSize; ++j)
			{
				const auto vPos = vWinPos + index<2>(i, j);
				const auto vNeuronPos = index<3>(idx[0], vPos[0], vPos[1]);

				fResult = fmax(iNeuronsRO[vNeuronPos], fResult);
			}
		}

		iNeuronsRW[idx] = fResult;
	}
	);
}

void NNModel::maxPool2x2(uint32_t i)
{
	auto& iNeuronsRW = *m_vMaxPools[i];
	const auto &iNeuronsRO = *m_vFeatureMaps[i];

	// Fast 2x2 down sampling with max-filter
	parallel_for_each(
		// Define the compute domain, which is the set of threads that are created.
		iNeuronsRO.extent.tile<1, 2, 2>(),
		// Define the code to run on each thread on the accelerator.
		[=, &iNeuronsRW, &iNeuronsRO](const tiled_index<1, 2, 2> t_idx) restrict(amp)
	{
		// Group (tile)-shared memory
		tile_static float fPool[2][2];
		auto &fResult = fPool[t_idx.local[1]][t_idx.local[2]];

		fResult = iNeuronsRO[t_idx];
		t_idx.barrier.wait();

		// Take max in binary
		fResult = t_idx.local[2] ? fmax(fResult, fPool[t_idx.local[1]][t_idx.local[2] - 1]) : fResult;
		fResult = t_idx.local[1] ? fmax(fResult, fPool[t_idx.local[1] - 1][t_idx.local[2]]) : fResult;

		if (t_idx.local[1] && t_idx.local[2])
		{
			iNeuronsRW[t_idx.tile] = fResult;
		}
	}
	);
}
