/**
 * @file    base_heat_solver.h
 * @authors Filip Vaverka <ivaverka@fit.vutbr.cz>
 *          Jiri Jaros <jarosjir@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2019/2020 - Project 1
 *          This file contains base class from which sequential and parallel
 *          versions of the solver are derived.
 *
 * @date    2018-03-28
 */

#ifndef BASE_HEAT_SOLVER_H
#define BASE_HEAT_SOLVER_H

#include <hdf5.h>
#include <limits>

#include "material_properties.h"
#include "simulation_properties.h"

/**
 * @brief The BaseHeatSolver class represents base class for implementation of
 * simple heat equation solver in heterogeneous medium.
 */
class BaseHeatSolver
{
public:
    struct ErrorInfo_t {
        float maxErrorValue;
        size_t maxErrorIdx;
    };

    /**
     * @brief Contructor
     * @param simulationProps Parameters of simulation read from command line arguments.
     * @param materialProps   Parameters of material read from the input file.
     */
    BaseHeatSolver(SimulationProperties &simulationProps, MaterialProperties &materialProps);

    /**
     * @brief Destructor
     */
    virtual ~BaseHeatSolver();

    /**
     * @brief Pure-virtual method that needs to be implemented by each solver
     * implementation and which is called to execute the simulation.
     * @param outResult Output array which is to be filled with computed temperature values.
     *                  The vector is pre-allocated and its size is given by dimensions
     *                  of the input file (edgeSize*edgeSize).
     *                  NOTE: Note that the vector can be 0-sized in case of the
     *                  MPI based implementation (where only ROOT - rank = 0 -
     *                  process gets needs this output array)!
     */
    virtual void RunSolver(std::vector<float, AlignedAllocator<float> > &outResult) = 0;

    /**
     * @brief Returns "true" every N-th iteration so that "true" is returned every
     *        10% of the progress.
     * @param iteration Integer representing current iteration.
     * @return Returns "true" when the code is supposed to print progress report.
     */
    bool ShouldPrintProgress(size_t iteration) const;

    /**
     * @brief Prints 2D square-shaped array of values stored in row-major order with
     *        edge length of "edgeSize-elements".
     * @param data     Pointer to the array values.
     * @param edgeSize Length of edge of the square array.
     */
    static void PrintArray(const float *data, size_t edgeSize);

    /**
     * @brief Writes "width"x"height" array of floats into *.png image using
     *        MATLAB HSV palette.
     *        NOTE: Note that the "data" array is first normalized so that
     *        full palette range is used every time!
     * @param fileName Name of the image file to be created.
     * @param data     "width"-by-"height" row-major ordered array of float values.
     * @param width    Width of the resulting image (and input 2D array).
     * @param height   Height of the resulting image (and input 2D array).
     * @param range    Range of values (used for normalization) of data ([min; max]).
     */
    static void StoreAsImage(const std::string &fileName, const float *data,
                             unsigned width, unsigned height,
                             std::pair<float, float> *range = nullptr);

    /**
     * @brief VerifyResults
     * @param seqResult
     * @param parResult
     * @param outAbsDiff
     * @param totalGridPoints
     * @param epsilon
     * @return
     */
    static bool VerifyResults(const float *seqResult, const float *parResult,
                              float *outAbsDiff, ErrorInfo_t &outErrorInfo,
                              size_t totalGridPoints, float epsilon = 0.001f);

protected:
    /**
     * @brief Evalute heat-equation solver stencil function at the specified point
     *        using 4-neighbourhood and 2 points in each direction. The method
     *        uses results of previous simulation step and writtes new value into
     *        the next one.
     *
     * @param oldTemp     [IN]  Array representing the domain state computed in PREVIOUS sim. step.
     * @param newTemp     [OUT] Array representing the domain in which computed point will be stored.
     * @param params      Parameters of the material at each point of the (sub-)domain.
     * @param map         Material map where "0" values represent air.
     * @param i           Row (Y-axis) position of the evaluated point (as in i-th row).
     * @param j           Column (X-axis) position of the evaluated point (as in j-th column).
     * @param width       Width (or row length) of "oldTemp" and "newTemp" 2D arrays.
     * @param airFlowRate Rate of heat dissipation due to air flow.
     * @param coolerTemp  Temperature of the cooler.
     */
    inline void ComputePoint(const float *oldTemp, float *newTemp, const float *params,
                             const int *map, size_t i, size_t j, size_t width,
                             float airFlowRate, float coolerTemp) const;

    /**
     * @brief Prints human readable simulation progress report every N-th iteration
     *        (see "ShouldPrintProgress" which is used internally).
     * @param iteration        Integer representing current iteration.
     * @param middleColAvgTemp Computed temperature average of the column
     *                         in the middle of the domain.
     */
    void PrintProgressReport(size_t iteration, float middleColAvgTemp);

    /**
     * @brief Print either human or machine (CSV) readable report of status of
     *        FINISHED simulation.
     *        NOTE: Should be called after last iteration of the simulation.
     * @param totalTime        Total time elapsed since the simulation begun [s] (ie. MPI_Wtime(...))
     * @param middleColAvgTemp Computed temperature average of the middle column in the domain.
     * @param codeType         Type of executed code: "seq" - for sequential code
     *                                                "par" - for parallel (MPI or Hybrid) code
     */
    void PrintFinalReport(double totalTime, float middleColAvgTemp,
                          const std::string &codeType);

    /**
     * @brief Stores the simulation state in "data" into HDF5 file using sequential HDF5
     *        this *HAS* to be called only from single (usually MASTER/RANK=0 process).
     *        The method assumes that the "data" is 2D square array of "edgeSize"x"edgeSize"
     *        elements (where "edgeSize" is read from the input file).
     * @param fileHandle Handle to opened HDF5 file (using sequential HDF5).
     * @param iteration  Integer representing current iteration.
     * @param data       2D square array of "edgeSize"x"edgeSize" elements.
     */
    void StoreDataIntoFile(hid_t fileHandle, size_t iteration, const float *data);

    /**
     * @brief Evaluate heat-equation over specified tile
     * @param oldTemp       [IN]  Array representing the domain state computed in PREVIOUS sim. step.
     * @param newTemp       [OUT] Array representing the updated domain.
     * @param params        Parameters of the material at each point of the (sub-)domain.
     * @param map           Material map where "0" values represent air.
     * @param offsetX       Offset of updated region of the array in X direction (>= 2).
     * @param offsetY       Offset of updated region of the array in Y direction (>= 2).
     * @param sizeX         Size of the updated region of the array in X direction.
     * @param sizeY         Size of the updated region of the array in Y direction.
     * @param strideX       Total size of the array in X direction.
     * @param airFlowRate   Rate of heat dissipation due to air flow.
     * @param coolerTemp    Temperature of the cooler.
     */
    void UpdateTile(const float *oldTemp, float *newTemp, const float *params,
                    const int *map, size_t offsetX, size_t offsetY, size_t sizeX, size_t sizeY,
                    size_t strideX, float airFlowRate, float coolerTemp) const;

    SimulationProperties &m_simulationProperties; ///< Parameters of the simulation (from arguments)
    MaterialProperties &m_materialProperties;     ///< Parameters of the material (from input file)
};

inline void BaseHeatSolver::ComputePoint(const float *oldTemp, float *newTemp, const float *params,
                                         const int *map, size_t i, size_t j, size_t width,
                                         float airFlowRate, float coolerTemp) const
{
    // 1. Precompute neighbor indices.
    const unsigned center    = unsigned(i * width + j);
    const unsigned top[2]    = { center - unsigned(width), center - 2*unsigned(width) };
    const unsigned bottom[2] = { center + unsigned(width), center + 2*unsigned(width) };
    const unsigned left[2]   = { center - 1, center - 2 };
    const unsigned right[2]  = { center + 1, center + 2 };

    // 2. The reciprocal value of the sum of domain parameters for normalization.
    const float frac = 1.0f / (params[top[0]]    + params[top[1]]    +
                               params[bottom[0]] + params[bottom[1]] +
                               params[left[0]]   + params[left[1]]   +
                               params[right[0]]  + params[right[1]]  +
                               params[center]);

    // 3. Compute new temperature at the specified grid point.
    float pointTemp =
            oldTemp[top[0]]    * params[top[0]]    * frac +
            oldTemp[top[1]]    * params[top[1]]    * frac +
            oldTemp[bottom[0]] * params[bottom[0]] * frac +
            oldTemp[bottom[1]] * params[bottom[1]] * frac +
            oldTemp[left[0]]   * params[left[0]]   * frac +
            oldTemp[left[1]]   * params[left[1]]   * frac +
            oldTemp[right[0]]  * params[right[0]]  * frac +
            oldTemp[right[1]]  * params[right[1]]  * frac +
            oldTemp[center]    * params[center]    * frac;

    // 4. Remove some of the heat due to air flow
    pointTemp = (map[center] == 0)
            ? (airFlowRate * coolerTemp) + ((1.0f - airFlowRate) * pointTemp)
            : pointTemp;

    newTemp[center] = pointTemp;
}

#endif // BASE_HEAT_SOLVER_H
