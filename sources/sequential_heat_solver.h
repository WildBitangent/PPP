/**
 * @file    sequential_heat_solver.h
 * @authors Filip Vaverka <ivaverka@fit.vutbr.cz>
 *          Jiri Jaros <jarosjir@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2019/2020 - Project 1
 *          This file contains implementation of sequential heat equation solver.
 *
 * @date    2018-03-28
 */

#ifndef SEQUENTIAL_HEAT_SOLVER_H
#define SEQUENTIAL_HEAT_SOLVER_H

#include "base_heat_solver.h"

/**
 * @brief The SequentialHeatSolver class implements reference sequential heat
 *        equation solver in 2D domain.
 */
class SequentialHeatSolver : public BaseHeatSolver
{
public:
    /**
     * @brief Constructor - Initializes the solver. This includes:
     *        - Allocate temporary working storage for simulation.
     *        - Open output HDF5 file (if filename was provided).
     * @param simulationProps Parameters of simulation - passed into base class.
     * @param materialProps   Parameters of material - passed into base class.
     */
    SequentialHeatSolver(SimulationProperties &simulationProps, MaterialProperties &materialProps);

    /**
     * @brief Run main simulation loop.
     * @param outResult Output array which is to be filled with computed temperature values.
     */
    virtual void RunSolver(std::vector<float, AlignedAllocator<float> > &outResult);

protected:
    /**
     * @brief Compute average temperature in middle column of the domain.
     * @param data 2D array containing simulation state.
     * @return Returns average temperature in the middle column of the domain.
     */
    float ComputeMiddleColAvgTemp(const float *data) const;

    std::vector<float, AlignedAllocator<float> > m_tempArray;   ///< Temporary work array.
    AutoHandle<hid_t> m_fileHandle;                             ///< Output HDF5 file handle.
};

#endif // SEQUENTIAL_HEAT_SOLVER_H
