/**
 * @file    main.cpp
 * @authors Filip Vaverka <ivaverka@fit.vutbr.cz>
 *          Jiri Jaros <jarosjir@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2019/2020 - Project 1
 *
 * @date    2018-03-24
 */

#include "base.h"
#include "base_heat_solver.h"
#include "sequential_heat_solver.h"
#include "parallel_heat_solver.h"

int main(int argc, char *argv[])
{
#ifdef _OPENMP
    int providedParallelism = -1;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedParallelism);
    if(providedParallelism < MPI_THREAD_FUNNELED)
    {
        std::cerr << "MPI init error, atleast MPI_THREAD_FUNNELED not provided!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
#else
    MPI_Init(&argc, &argv);
#endif // _OPENMP

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    SimulationProperties simulationProps;
    simulationProps.ParseCommandLine(argc, argv);

    MaterialProperties materialProps;

    materialProps.LoadMaterialData(simulationProps.GetMaterialFileName(), rank == 0);

    if(rank == 0)
        simulationProps.PrintParameters(materialProps);

    std::vector<float, AlignedAllocator<float> > sequentialResult;
    std::vector<float, AlignedAllocator<float> > parallelResult;

    if(simulationProps.IsRunSequential() && rank == 0)
    {
        if(!simulationProps.IsBatchMode())
            std::cout << "============ Running sequential solver =============" << std::endl;

        sequentialResult.resize(materialProps.GetGridPoints());
        SequentialHeatSolver heatSolver(simulationProps, materialProps);
        heatSolver.RunSolver(sequentialResult);
    }

    if(simulationProps.IsRunParallel())
    {
        if(rank == 0)
        {
            if(!simulationProps.IsBatchMode())
                std::cout << "============= Running parallel solver ==============" << std::endl;

            parallelResult.resize(materialProps.GetGridPoints());
        }

        ParallelHeatSolver heatSolver(simulationProps, materialProps);
        heatSolver.RunSolver(parallelResult);
    }

    if(simulationProps.IsValidation() && rank == 0)
    {
        if(simulationProps.IsDebug())
        {            
            if(!simulationProps.GetDebugImageFileName().empty())
            {
                const std::string seqImageName = SimulationProperties::AppendFileNameExt(
                            simulationProps.GetDebugImageFileName(), "seq", ".png");
                const std::string parImageName = SimulationProperties::AppendFileNameExt(
                            simulationProps.GetDebugImageFileName(), "par", ".png");

                BaseHeatSolver::StoreAsImage(seqImageName, sequentialResult.data(),
                                             materialProps.GetEdgeSize(), materialProps.GetEdgeSize());

                BaseHeatSolver::StoreAsImage(parImageName, parallelResult.data(),
                                             materialProps.GetEdgeSize(), materialProps.GetEdgeSize());
            }
            else
            {
                std::cout << "=============== Sequential results =================" << std::endl;
                BaseHeatSolver::PrintArray(sequentialResult.data(), materialProps.GetEdgeSize());
                std::cout << std::endl;

                std::cout << "================ Parallel results ==================" << std::endl;
                BaseHeatSolver::PrintArray(parallelResult.data(), materialProps.GetEdgeSize());
                std::cout << std::endl;
            }
        }

        std::vector<float, AlignedAllocator<float> > absError(materialProps.GetGridPoints(), 0.0);

        BaseHeatSolver::ErrorInfo_t errorInfo;
        bool result = BaseHeatSolver::VerifyResults(sequentialResult.data(), parallelResult.data(),
                                                    absError.data(), errorInfo,
                                                    materialProps.GetGridPoints(), 0.001f);

        if(result)
        {
            std::cout << "Maximum error of " <<
                         std::scientific << errorInfo.maxErrorValue << std::defaultfloat <<
                         " is at [" << errorInfo.maxErrorIdx / materialProps.GetEdgeSize() <<
                         ", " << errorInfo.maxErrorIdx % materialProps.GetEdgeSize() << "]" << std::endl;
            std::cout << "Verification FAILED" << std::endl;
        }
        else
        {
            std::cout << "Max deviation is: " << std::scientific <<
                         errorInfo.maxErrorValue <<
                         std::defaultfloat << std::endl;
            std::cout << "Verification OK" << std::endl;
        }

        if(!simulationProps.GetDebugImageFileName().empty())
        {
            std::pair<float, float> range = std::make_pair(0.0f, 0.001f);
            const std::string errImageName = SimulationProperties::AppendFileNameExt(
                        simulationProps.GetDebugImageFileName(), "abs_diff", ".png");
            BaseHeatSolver::StoreAsImage(errImageName, absError.data(),
                                         materialProps.GetEdgeSize(), materialProps.GetEdgeSize(),
                                         &range);
        }
    }

    MPI_Finalize();
    return 0;
}
