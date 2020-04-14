/**
 * @file    parallel_heat_solver.h
 * @author  xkaras34 <xkaras34@stud.fit.vutbr.cz>
 *
 * @brief   Course: PPP 2019/2020 - Project 1
 *
 * @date    2020-04-DD
 */

#ifndef PARALLEL_HEAT_SOLVER_H
#define PARALLEL_HEAT_SOLVER_H

#include "base_heat_solver.h"

template<typename T>
struct Vec2
{
    T x,y;
};

using Vec2i = Vec2<int>;
using Vec2f = Vec2<float>;
using Vec2s = Vec2<size_t>;

/**
 * @brief The ParallelHeatSolver class implements parallel MPI based heat
 *        equation solver in 2D using 1D and 2D block grid decomposition.
 */
class ParallelHeatSolver : public BaseHeatSolver
{
    static const size_t ROOT = 0;
    using Bufferf = std::vector<float, AlignedAllocator<float>>;
    using Bufferi = std::vector<int, AlignedAllocator<int>>;

    enum class Direction
    {
        Up,
        Down,
        Left,
        Right,

        Count
    };

public:
    /**
     * @brief Constructor - Initializes the solver. This should include things like:
     *        - Construct 1D or 2D grid of tiles.
     *        - Create MPI datatypes used in the simulation.
     *        - Open SEQUENTIAL or PARALLEL HDF5 file.
     *        - Allocate data for local tile.
     *        - Initialize persistent communications?
     *
     * @param simulationProps Parameters of simulation - passed into base class.
     * @param materialProps   Parameters of material - passed into base class.
     */
    ParallelHeatSolver(SimulationProperties &simulationProps, MaterialProperties &materialProps);
    virtual ~ParallelHeatSolver();

    /**
     * @brief Run main simulation loop.
     * @param outResult Output array which is to be filled with computed temperature values.
     *                  The vector is pre-allocated and its size is given by dimensions
     *                  of the input file (edgeSize*edgeSize).
     *                  NOTE: The vector is allocated (and should be used) *ONLY*
     *                        by master process (rank 0 in MPI_COMM_WORLD)!
     */
    virtual void RunSolver(std::vector<float, AlignedAllocator<float> > &outResult);

private:
    void scatterData();
    void gatherData(const Bufferf& what, Bufferf& where);
    void createTypes();
    // std::vector<int> getNeighbors();
    void sendHaloZone(float* data, MPI_Win& win, Direction dir);
    void save(const Bufferf& data, size_t iteration);
    float computeMiddleTemp(const Bufferf& data);

    bool isRoot() const { return mRank == ROOT; }


protected:
    int mRank;     ///< Process rank in global (MPI_COMM_WORLD) communicator.
    int mSize;     ///< Total number of processes in MPI_COMM_WORLD.
    int mTempRank = -1;
    AutoHandle<hid_t> mFileHandle;

    MPI_Datatype mTWorkerTile;
    MPI_Datatype mTWorkerTileInt;
    MPI_Datatype mTTileRoot;
    MPI_Datatype mTTileResized;
    MPI_Datatype mTTileEastWest;
    MPI_Datatype mTTileNorthSouth;

    MPI_Comm mTemperatureComm;

    std::array<Bufferf, 2> mBuffers;
    std::array<MPI_Win, 2> mWindows;

    Bufferf mDomainParams;
    Bufferi mDomainMap;

    Vec2i mGridSize;
    Vec2s mTileSize;

    // scatter/gather data
    std::vector<int> mTileCounts;
	std::vector<int> mTileDisplacements;
};

#endif // PARALLEL_HEAT_SOLVER_H
