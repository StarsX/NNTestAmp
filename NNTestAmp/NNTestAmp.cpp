//--------------------------------------------------------------------------------------
// By XU, Tianchen
//--------------------------------------------------------------------------------------

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

#include "NNModel.h"

using namespace std;

bool readKernel(vector<float> &vWeights, vector<float> &vBias,
	const char *szFileName, uint32_t uKSize,
	uint32_t uImageCount, uint32_t uMapCount, uint32_t uPadding = 0)
{
	vWeights.resize(uKSize * uImageCount * uMapCount);
	vBias.resize(uMapCount);

	ifstream pFile(szFileName, ios::in | ios::binary);
	if (pFile)
	{
		for (auto i = 0u; i < uMapCount; ++i)
		{
			pFile.read(reinterpret_cast<char*>(&vBias[i]), sizeof(float));
			pFile.read(reinterpret_cast<char*>(&vWeights[i * uKSize * uImageCount]), sizeof(float) * uKSize * uImageCount);

			if (uPadding > 0) pFile.ignore(sizeof(float) * uPadding);
		}
		pFile.close();

		return true;
	}

	return false;
}

bool writeKernel(const vector<float> &vWeights, const vector<float> &vBias,
	const char *szFileName, uint32_t uKSize,
	uint32_t uImageCount, uint32_t uMapCount)
{
	ofstream pFile(szFileName, ios::out | ios::binary);
	if (pFile)
	{
		for (auto i = 0u; i < uMapCount; ++i)
		{
			pFile.write(reinterpret_cast<const char*>(&vBias[i]), sizeof(float));
			pFile.write(reinterpret_cast<const char*>(&vWeights[i * uKSize * uImageCount]), sizeof(float) * uKSize * uImageCount);
		}
		pFile.close();

		return true;
	}

	return false;
}

int main()
{
	const auto uLayers = 4u;

	const float input[] =
	{
		1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
		1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
	};

	const NNModel::FeatureMapInfo vMapInfo[] =
	{
		NNModel::FeatureMapInfo { 6, 13, 13 },
		NNModel::FeatureMapInfo { 50, 5, 5 },
		NNModel::FeatureMapInfo { 100, 1, 1 },
		NNModel::FeatureMapInfo { 10, 1, 1 }
	};

	const int pKernelSizes[] = { 5, 5, 5, 1 };
	const uint32_t pStrides[] = { 2, 2, 0, 0 };

	vector<float> pvWeights[uLayers];
	vector<float> pvBias[uLayers];

	for (auto i = 0u; i < uLayers; ++i)
	{
		const string szFileName = "lw" + to_string(i + 1) + ".wei";
		if (!readKernel(pvWeights[i], pvBias[i], szFileName.c_str(), pKernelSizes[i] * pKernelSizes[i],
			i > 0 ? vMapInfo[i - 1].m_uNum : 1, vMapInfo[i].m_uNum))
		{
			cout << "Failed to read file " << szFileName << "!" << endl;
			system("pause");

			return 0;
		}
	}

	// Start CNN model
	NNModel model(uLayers);
	model.SetInputImages(input, 29, 29);
	model.SetFeatureMaps(vMapInfo);
	model.SetKernels(pvWeights, pvBias, pKernelSizes, pStrides);

	model.PrintLayerData();
	model.Execute();

	// Collect results
	const auto vResults = model.GetResult();
	const auto result = max_element(vResults.begin(), vResults.end());
	cout << endl << "The digit is recognized as " << distance(vResults.begin(), result) << "." << endl;

	auto n = 0;
	cout << endl << "Score list:" << endl;
	for (const auto &fResult : vResults)
		cout << n++ << " - " << fResult << endl;

	system("pause");

	return 0;
}
