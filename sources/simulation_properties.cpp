/**
 * @file    simulation_properties.cpp
 * @authors Filip Vaverka <ivaverka@fit.vutbr.cz>
 *          Jiri Jaros <jarosjir@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2019/2020 - Project 1
 *          Parameters of the simulation passed as application arguments.
 *
 * @date    2018-03-28
 */

#include <getopt.h>
#include <cmath>
#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include <omp.h>
#include <mpi.h>

#include "simulation_properties.h"


SimulationProperties::SimulationProperties()
    : m_nIterations(1)
    , m_diskWriteIntensity(1000)
    , m_airFlowRate(0.001f)
    , m_mode(SIMULATION_MODE_SEQUENTIAL)
    , m_debugFlag(false)
    , m_verificationFlag(false)
    , m_sequentialFlag(false)
    , m_batchMode(false)
    , m_batchModeHeader(false)
    , m_useParallelIO(false)
    , m_decompMode(DECOMP_MODE_1D)
{
}

void SimulationProperties::ParseCommandLine(int argc, char *argv[])
{
    std::map<char, bool> flags;

    int c;
    while((c = getopt(argc, argv, "n:w:a:dvi:o:bhm:pt:r:g")) != -1)
    {
        std::string optionArg;
        if(optarg)
            optionArg = std::string(optarg);

        std::stringstream optionArgParser(optionArg);

        switch(c)
        {
        case 'n':
            optionArgParser >> m_nIterations;
            flags['n'] = !optionArgParser.fail() && optionArgParser.eof();
            break;
        case 'w':
            optionArgParser >> m_diskWriteIntensity;
            flags['w'] = !optionArgParser.fail() && optionArgParser.eof();
            break;
        case 'a':
            optionArgParser >> m_airFlowRate;
            flags['a'] = !optionArgParser.fail() && optionArgParser.eof();
            break;
        case 'd':
            m_debugFlag = true;
            break;
        case 'm': {
                int mode = 0;
                optionArgParser >> mode;
                m_mode = static_cast<SimulationMode_t>(mode);
            }
            flags['m'] = !optionArgParser.fail() && optionArgParser.eof();
            break;
        case 'v':
            m_verificationFlag = true;
            break;
        case 'i':
            optionArgParser >> m_materialFileName;
            flags['i'] = !optionArgParser.fail() && optionArgParser.eof();
            break;
        case 'o':
            optionArgParser >> m_outputFileName;
            flags['o'] = !optionArgParser.fail() && optionArgParser.eof();
            break;
        case 'b':
            m_batchMode = true;
            break;
        case 'h':
            m_batchModeHeader = true;
            break;
        case 'p':
            m_useParallelIO = true;
            break;
        case 't':
            optionArgParser >> m_nThreads;
            flags['t'] = !optionArgParser.fail() && optionArgParser.eof();
            break;
        case 'r':
            optionArgParser >> m_debugImageName;
            break;
        case 'g':
            m_decompMode = DECOMP_MODE_2D;
            break;

        default:
            std::cerr << "Unknown parameter!" << std::endl;
            PrintUsageAndExit();
        }
    }

    if(!(flags['n'] && flags['i'] && flags['m']) ||
            (m_mode < 0 || m_mode >= SIMULATION_MODE_COUNT))
    {
        PrintUsageAndExit();
    }

    if(!flags['w'])
        m_diskWriteIntensity = 50;

    if(!flags['t'])
        m_nThreads = 1;

    omp_set_num_threads(m_nThreads);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Compute how arrangement of the processes into the grid
    if(GetDecompMode() == SimulationProperties::DECOMP_MODE_1D)
    {
        m_gridSize[0] = size;
        m_gridSize[1] = 1;
    }
    else
    {
        bool isEvenPower = (int(std::log2(size)) % 2 == 0);

        m_gridSize[0] = int(sqrt(size / (isEvenPower ? 1 : 2)));
        m_gridSize[1] = m_gridSize[0] * (isEvenPower ? 1 : 2);
    }
}

void SimulationProperties::PrintUsageAndExit() const
{
    std::cerr << "Usage:" << std::endl;
    std::cerr << "Mandatory arguments:" << std::endl;
    std::cerr << "  -m [0-1]    mode 0 - run sequential version" << std::endl <<
                 "              mode 1 - run parallel version" << std::endl;
    std::cerr << "  -n          number of iterations" << std::endl;
    std::cerr << "  -i          material HDF5 file" << std::endl;

    std::cerr << "Optional arguments:" << std::endl;
    std::cerr << "  -t          number of OpenMP threads (default 1)" << std::endl;
    std::cerr << "  -o          output HDF5 file" << std::endl;
    std::cerr << "  -w          disk write intensity (every N-th step)" << std::endl;
    std::cerr << "  -a          air flow rate (values in <0.0001, 0.5> make sense)" << std::endl;
    std::cerr << "  -d          debug mode (copare results of SEQ and PAR versions and print result)" << std::endl;
    std::cerr << "  -v          verification mode (copare results of SEQ and PAR versions)" << std::endl;
    std::cerr << "  -p          parallel I/O mode" << std::endl;
    std::cerr << "  -r          render results (with -d or -v) into *.png image." << std::endl;
    std::cerr << "  -g          Use 2D decomposition instead of 1D." << std::endl;

    std::cerr << "  -b          batch mode - output data into CSV format" << std::endl;
    std::cerr << "  -h          batch mode - print CSV header" << std::endl;

    MPI_Abort(MPI_COMM_WORLD, -1);
}

void SimulationProperties::PrintParameters(const MaterialProperties &materialProps) const
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(m_batchMode)
    {
        if(m_batchModeHeader)
        {
            std::cout << "mpi_procs" << ";"
                      << "grid_tiles_x" << ";" << "grid_tiles_y" << ";"
                      << "omp_threads" << ";"
                      << "domain_size" << ";"
                      << "n_iterations" << ";"
                      << "disk_write_intensity" << ";"
                      << "airflow" << ";"
                      << "material_file" << ";"
                      << "output_file" << ";"
                      << "simulation_mode" << ";"
                      << "middle_col_avg_temp" << ";"
                      << "total_time" << ";"
                      << "iteration_time" << std::endl;
        }

        std::cout << size << ";"
                  << m_gridSize[0] << ";" << m_gridSize[1] << ";"
                  << GetNumThreads() << ";"
                  << materialProps.GetEdgeSize() << ";"
                  << GetNumIterations() << ";"
                  << GetDiskWriteIntensity() << ";"
                  << GetAirFlowRate() << ";"
                  << GetMaterialFileName() << ";";
    }
    else
    {
        std::cout << ".......... Parameters of the simulation ..........." << std::endl;
        std::cout << "Domain size             : " << materialProps.GetEdgeSize() << "x" << materialProps.GetEdgeSize() << std::endl;
        std::cout << "Number of iterations    : " << GetNumIterations() << std::endl;
        std::cout << "Number of MPI processes : " << size << std::endl;
        std::cout << "Number of OpenMP threads: " << GetNumThreads() << std::endl;
        std::cout << "Disk write intensity    : " << GetDiskWriteIntensity() << std::endl;
        std::cout << "Air flow rate           : " << GetAirFlowRate() << std::endl;
        std::cout << "Input file name         : " << GetMaterialFileName() << std::endl;
        std::cout << "Output file name        : " << GetOutputFileName() << std::endl;
        std::cout << "Mode                    : " << m_mode << std::endl;
        std::cout << "Decomposition type      : " << ((m_decompMode == DECOMP_MODE_1D) ? "1D" : "2D")
                                                  << " (" << m_gridSize[0] << ", " << m_gridSize[1] << ")" << std::endl;
        std::cout << "..................................................." << std::endl << std::endl;
    }
}

std::string SimulationProperties::GetOutputFileName(const std::string &codeTypeExt) const
{
    if(m_outputFileName.empty())
        return std::string();
    else if(codeTypeExt.empty())
        return m_outputFileName;

    return AppendFileNameExt(m_outputFileName, codeTypeExt, ".h5");
}

std::string SimulationProperties::AppendFileNameExt(const std::string &fileName,
                                                    const std::string &fileNameExt,
                                                    const std::string &fileTypeExt)
{
    std::string result = fileName;

    if(fileName.find(fileTypeExt) == std::string::npos)
        result.append("_" + fileNameExt + fileTypeExt);
    else
        result.insert(result.find_last_of("."), "_" + fileNameExt);

    return result;
}
