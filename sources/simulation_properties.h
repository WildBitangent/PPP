/**
 * @file    simulation_properties.h
 * @authors Filip Vaverka <ivaverka@fit.vutbr.cz>
 *          Jiri Jaros <jarosjir@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2019/2020 - Project 1
 *          Parameters of the simulation passed as application arguments.
 *
 * @date    2018-03-28
 */

#ifndef SIMULATION_PROPERTIES_H
#define SIMULATION_PROPERTIES_H

#include <string>
#include "material_properties.h"

/**
 * @brief The SimulationProperties class represents parameters of the simulation
 *        passed as application arguments.
 */
class SimulationProperties
{
public:
    /**
     * @brief Mode of the simulation
     */
    enum SimulationMode_t {
        SIMULATION_MODE_SEQUENTIAL = 0, ///< Run sequential code only.
        SIMULATION_MODE_PARALLEL,       ///< Run parallel code only.

        SIMULATION_MODE_COUNT // *** KEEP AS LAST ITEM **
    };

    /**
     * @brief Type of the decomposition in MPI code
     */
    enum DecompositionMode_t {
        DECOMP_MODE_1D = 0, ///< Run in 1D decomposition (= Nx1).
        DECOMP_MODE_2D,     ///< Run in 2D decomposition (~ N^(1/2) x N^(1/2)).
    };

    /**
     * @brief Constructor
     */
    SimulationProperties();

    /**
     * @brief Parse command line arguments passed to the application.
     * @param argc
     * @param argv
     */
    void ParseCommandLine(int argc, char *argv[]);

    /**
     * @brief Print usage of the application and exit.
     */
    void PrintUsageAndExit() const;

    /**
     * @brief Print current simulation parameters.
     * @param materialProps Properties of the domain loaded from the input file.
     */
    void PrintParameters(const MaterialProperties &materialProps) const;

    /**
     * @brief Run sequential version?
     * @return Returns "true" if sequential version should be executed.
     */
    bool IsRunSequential() const {
        return (m_verificationFlag || m_debugFlag || m_mode == SIMULATION_MODE_SEQUENTIAL);
    }

    /**
     * @brief Run parallel version?
     * @return Returns "true" if parallel version should be executed.
     */
    bool IsRunParallel() const {
        return (m_verificationFlag || m_debugFlag || m_mode == SIMULATION_MODE_PARALLEL);
    }

    /**
     * @brief Should be validation performed?
     * @return Returns "true" if validation should be performed.
     */
    bool IsValidation() const {
        return (m_verificationFlag || m_debugFlag);
    }

    /**
     * @brief Get number of iterations requested.
     * @return Returns number of iterations specified by the user.
     */
    size_t GetNumIterations() const { return m_nIterations; }

    /**
     * @brief Get number of threads per MPI process requested.
     * @return Returns number of threads per MPI process specified by the user.
     */
    int GetNumThreads() const { return m_nThreads; }

    /**
     * @brief Get number of iterations between each time simulation state snapshot is taken.
     * @return Returns number of iterations to skip between state snapshost.
     */
    size_t GetDiskWriteIntensity() const { return m_diskWriteIntensity; }

    /**
     * @brief Get air flow rate around the heatsink
     * @return Returns the air flow rate.
     */
    float GetAirFlowRate() const { return m_airFlowRate; }

    /**
     * @brief Get path to the input material file.
     * @return Returns path to input material file.
     */
    const std::string &GetMaterialFileName() const { return m_materialFileName; }

    /**
     * @brief Get filename and path of the output file.
     *        NOTE: User should specify code type which is creating the file as:
     *              "seq" - for sequential code
     *              "par" - for parallel code
     * @param codeTypeExt "seq" or "par" depending on which code is creating the file.
     * @return Returns path to the output file including code specific extension.
     */
    std::string GetOutputFileName(const std::string &codeTypeExt = std::string()) const;

    /**
     * @brief Get base name for debug images.
     * @return Returns base name (if specified) of debug images.
     */
    const std::string &GetDebugImageFileName() const { return m_debugImageName; }

    /**
     * @brief Is application running in batch mode?
     * @return Returns "true" if application should output only in batch mode.
     */
    bool IsBatchMode() const { return m_batchMode; }

    /**
     * @brief Is parallel I/O enabled?
     * @return Returns "true" if parallel I/O is enabled.
     */
    bool IsUseParallelIO() const { return m_useParallelIO; }

    /**
     * @brief IsDebug
     * @return Returns "true" if debug flag was specified.
     */
    bool IsDebug() const { return m_debugFlag; }

    /**
     * @brief GetDecompType
     * @return Returns member of "DecompositionMode_t".
     */
    DecompositionMode_t GetDecompMode() const { return m_decompMode; }

    /**
     * @brief Get decomposition grid
     * @return Returns number of subdivisions (tiles) in X and Y dimensions.
     */
    void GetDecompGrid(int &outSizeX, int &outSizeY) { outSizeX = m_gridSize[0]; outSizeY = m_gridSize[1]; }

    /**
     * @brief Append extension to the filename before its file type extension.
     *        As in: "my_file.abc" to "my_file_EXT.abc"
     * @param fileName      Base filename (with or without extension).
     * @param fileNameExt   Extension "EXT" to be appended together with "_".
     * @param fileTypeExt   File type extension.
     * @return Returns extended filename.
     */
    static std::string AppendFileNameExt(const std::string &fileName,
                                         const std::string &fileNameExt,
                                         const std::string &fileTypeExt);

protected:
    size_t m_nIterations;           ///< Number of iteration of the simulation.
    int m_nThreads;                 ///< Number of OMP threads per process.
    size_t m_diskWriteIntensity;    ///< Every N-th iteration result is stored.
    float m_airFlowRate;            ///< Air flow rate of cooling air.

    std::string m_materialFileName; ///< Path to input file.
    std::string m_outputFileName;   ///< Path to output file.

    SimulationMode_t m_mode;          ///< Mode of the simulation.
    bool m_debugFlag;                 ///< Compare results of sequential and parallel codes.
    bool m_verificationFlag;          ///< Verify the result.
    bool m_sequentialFlag;            ///< Unused.
    bool m_batchMode;                 ///< Output only in CSV format.
    bool m_batchModeHeader;           ///< Output CSV header.
    bool m_useParallelIO;             ///< Whether to use parallel HDF5 I/O.
    DecompositionMode_t m_decompMode; ///< Decomposition of the simulation in parallel mode.
    std::array<int, 2> m_gridSize;    ///< Number of tiles in each grid dimensions (N_x, N_y).

    std::string m_debugImageName;   ///< Base filename of debug images.
};

#endif // SIMULATION_PROPERTIES_H
