/**
 * @file    base_heat_solver.h
 * @authors Filip Vaverka <ivaverka@fit.vutbr.cz>
 *          Jiri Jaros <jarosjir@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2019/2020 - Project 1
 *
 * @date    2018-03-24
 */

#include <iostream>
#include <iomanip>
#include <limits>

#include "base_heat_solver.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

BaseHeatSolver::BaseHeatSolver(SimulationProperties &simulationProps,
                               MaterialProperties &materialProps)
    : m_simulationProperties(simulationProps), m_materialProperties(materialProps)
{

}

BaseHeatSolver::~BaseHeatSolver()
{
}

void BaseHeatSolver::PrintArray(const float *data, size_t edgeSize)
{
    for(size_t i = 0; i < edgeSize; ++i)
    {
        std::cout << "[Row " << i << "]: ";

        std::cout << std::scientific;
        for(size_t j = 0; j < edgeSize; ++j)
            std::cout << data[i * edgeSize + j] << " ";
        std::cout << std::endl;
        std::cout << std::defaultfloat;
    }
}

bool BaseHeatSolver::ShouldPrintProgress(size_t iteration) const
{
    const size_t nTotalIters = m_simulationProperties.GetNumIterations();
    if((iteration % (nTotalIters > 10 ? (nTotalIters / 10) : 1) != 0 && iteration + 1 != nTotalIters) ||
            m_simulationProperties.IsBatchMode())
        return false;

    return true;
}

void BaseHeatSolver::StoreAsImage(const std::string &fileName, const float *data,
                                  unsigned width, unsigned height, std::pair<float, float> *range)
{
    // Allocate buffer for pixel data in the image
    const unsigned paletteWidth = 18;
    std::vector<unsigned char> imageData(3 * (width + paletteWidth) * height, 0);

    // Find minimum and maximum values in the data for normalization
    float minValue = std::numeric_limits<float>::infinity();
    float maxValue = -std::numeric_limits<float>::infinity();

    if(!range)
    {
        for(size_t i = 0; i < width * height; ++i)
        {
            minValue = std::min(minValue, data[i]);
            maxValue = std::max(maxValue, data[i]);
        }
    }
    else
    {
        minValue = range->first;
        maxValue = range->second;
    }

    // Normalize the values and compute their color representation according to
    // the MATLAB HSV palette
    for(unsigned i = 0; i < height; ++i)
    {
        // Write values to image as usual and
        for(unsigned j = 0; j < width; ++j)
        {
            size_t srcIdx = i * width + j;
            size_t dstIdx = i * (width + paletteWidth) + j;

            float normalValue = (data[srcIdx] - minValue) / (maxValue - minValue);

            float red   = (normalValue < 0.5f) ? (-6.0f * normalValue + 67.0f / 32.0f) : ( 6.0f * normalValue -  79.0f / 16.0f);
            float green = (normalValue < 0.4f) ? ( 6.0f * normalValue -  3.0f / 32.0f) : (-6.0f * normalValue +  79.0f / 16.0f);
            float blue  = (normalValue < 0.7f) ? ( 6.0f * normalValue - 67.0f / 32.0f) : ( 6.0f * normalValue + 195.0f / 32.0f);

            imageData[3*dstIdx + 0] = static_cast<unsigned char>(std::max(std::min(red,   1.0f), 0.0f) * 255.0f);
            imageData[3*dstIdx + 1] = static_cast<unsigned char>(std::max(std::min(green, 1.0f), 0.0f) * 255.0f);
            imageData[3*dstIdx + 2] = static_cast<unsigned char>(std::max(std::min(blue,  1.0f), 0.0f) * 255.0f);
        }

        // add palette strip at the end (with 2px padding on left side).
        for(unsigned j = 2; j < paletteWidth; ++j)
        {
            size_t dstIdx = i * (width + paletteWidth) + j + width;

            float normalValue = 1.0f - float(i) * (1.0f / float(height));

            float red   = (normalValue < 0.5f) ? (-6.0f * normalValue + 67.0f / 32.0f) : ( 6.0f * normalValue -  79.0f / 16.0f);
            float green = (normalValue < 0.4f) ? ( 6.0f * normalValue -  3.0f / 32.0f) : (-6.0f * normalValue +  79.0f / 16.0f);
            float blue  = (normalValue < 0.7f) ? ( 6.0f * normalValue - 67.0f / 32.0f) : ( 6.0f * normalValue + 195.0f / 32.0f);

            imageData[3*dstIdx + 0] = static_cast<unsigned char>(std::max(std::min(red,   1.0f), 0.0f) * 255.0f);
            imageData[3*dstIdx + 1] = static_cast<unsigned char>(std::max(std::min(green, 1.0f), 0.0f) * 255.0f);
            imageData[3*dstIdx + 2] = static_cast<unsigned char>(std::max(std::min(blue,  1.0f), 0.0f) * 255.0f);
        }
    }

    // Write RGB pixel data into the *.png file.
    stbi_write_png(fileName.c_str(), int(width + paletteWidth), int(height), 3,
                   imageData.data(), int(3*(width + paletteWidth)));
}

bool BaseHeatSolver::VerifyResults(const float *seqResult, const float *parResult,
                                   float *outAbsDiff, ErrorInfo_t &outErrorInfo,
                                   size_t totalGridPoints, float epsilon)
{
    float maxErrorValue = 0.0f;
    size_t maxErrorIdx = 0;
    size_t errorsCount = 0;

    for(size_t i = 0; i < totalGridPoints; ++i)
    {
        outAbsDiff[i] = std::fabs(parResult[i] - seqResult[i]);
        if(outAbsDiff[i] > maxErrorValue)
        {
            maxErrorValue = outAbsDiff[i];
            maxErrorIdx = i;
        }

        if(outAbsDiff[i] > epsilon)
            errorsCount++;
    }

    outErrorInfo.maxErrorValue = maxErrorValue;
    outErrorInfo.maxErrorIdx = maxErrorIdx;

    return maxErrorValue > epsilon;
}

void BaseHeatSolver::PrintProgressReport(size_t iteration, float middleColAvgTemp)
{
    if(!ShouldPrintProgress(iteration))
        return;

    double progress = ((iteration + 1) * 100) / m_simulationProperties.GetNumIterations();

    std::cout << std::fixed;
    std::cout << "Progress " << std::setw(3) << unsigned(progress) <<
                 "% (Average Temperature " << std::fixed << middleColAvgTemp << " degrees)" << std::endl;
    std::cout << std::defaultfloat;
}

void BaseHeatSolver::PrintFinalReport(double totalTime, float middleColAvgTemp,
                                      const std::string &codeType)
{
    if(!m_simulationProperties.IsBatchMode())
    {
        std::cout << "====================================================" << std::endl;
        std::cout << "Execution time of \"" << codeType << "\" version: " << totalTime << "s" << std::endl;
        std::cout << "====================================================" << std::endl << std::endl;
    }
    else
    {
        std::cout << m_simulationProperties.GetOutputFileName(codeType) << ";" <<
                     codeType << ";" <<
                     middleColAvgTemp << ";" <<
                     totalTime << ";" <<
                     totalTime / m_simulationProperties.GetNumIterations() << std::endl;
    }
}

void BaseHeatSolver::StoreDataIntoFile(hid_t fileHandle, size_t iteration,
                                       const float *data)
{
    hsize_t gridSize[] = { m_materialProperties.GetEdgeSize(), m_materialProperties.GetEdgeSize() };

    // 1. Create new HDF5 file group named as "Timestep_N", where "N" is number
    //    of current snapshot. The group is placed into root of the file "/Timestep_N".
    std::string groupName = "Timestep_" + std::to_string(static_cast<unsigned long long>(iteration / m_simulationProperties.GetDiskWriteIntensity()));
    AutoHandle<hid_t> groupHandle(H5Gcreate(fileHandle, groupName.c_str(),
                                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose);
    // NOTE: AutoHandle<hid_t> is object wrapping HDF5 handle so that its automatically
    //       released when object goes out of scope (see RAII pattern).
    //       The class can be found in "base.h".

    {
        // 2. Create new dataset "/Timestep_N/Temperature" which is simulation-domain
        //    sized 2D array of "float"s.
        std::string dataSetName("Temperature");
        // 2.1 Define shape of the dataset (2D edgeSize x edgeSize array).
        AutoHandle<hid_t> dataSpaceHandle(H5Screate_simple(2, gridSize, NULL), H5Sclose);
        // 2.2 Create datased with specified shape.
        AutoHandle<hid_t> dataSetHandle(H5Dcreate(groupHandle, dataSetName.c_str(),
                                                  H5T_NATIVE_FLOAT, dataSpaceHandle,
                                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose);

        // Write the data from memory pointed by "data" into new datased.
        // Note that we are filling whole dataset and therefore we can specify
        // "H5S_ALL" for both memory and dataset spaces.
        H5Dwrite(dataSetHandle, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

        // NOTE: Both dataset and dataspace will be closed here automatically (due to RAII).
    }

    {
        // 3. Create Integer attribute in the same group "/Timestep_N/Time"
        //    in which we store number of current simulation iteration.
        std::string attributeName("Time");

        // 3.1 Dataspace is single value/scalar.
        AutoHandle<hid_t> dataSpaceHandle(H5Screate(H5S_SCALAR), H5Sclose);

        // 3.2 Create the attribute in the group as double.
        AutoHandle<hid_t> attributeHandle(H5Acreate2(groupHandle, attributeName.c_str(),
                                                     H5T_IEEE_F64LE, dataSpaceHandle,
                                                     H5P_DEFAULT, H5P_DEFAULT), H5Aclose);

        // 3.3 Write value into the attribute.
        double snapshotTime = double(iteration);
        H5Awrite(attributeHandle, H5T_IEEE_F64LE, &snapshotTime);

        // NOTE: Both dataspace and attribute handles will be released here.
    }

    // NOTE: The group handle will be released here.
}

void BaseHeatSolver::UpdateTile(const float *oldTemp, float *newTemp,
                                const float *params, const int *map,
                                size_t offsetX, size_t offsetY,
                                size_t sizeX, size_t sizeY, size_t strideX,
                                float airFlowRate, float coolerTemp) const
{
    #pragma omp parallel for firstprivate(oldTemp, newTemp) default(shared)
    for(size_t i = offsetY; i < offsetY + sizeY; ++i)
    {
        #pragma omp simd
        for(size_t j = offsetX; j < offsetX + sizeX; ++j)
        {
            ComputePoint(oldTemp, newTemp,
                    params,
                    map,
                    i, j, strideX,
                    airFlowRate,
                    coolerTemp);
        }
    }
}
