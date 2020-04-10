/**
 * @file        main.cpp
 * @author      Jiri Jaros \n
 *              Faculty of Information Technology \n
 *              Brno University of Technology \n
 *              jarosjir@fit.vutbr.cz
 *
 * @brief       The implementation file creating a file with all the material
 *              properties of the domain.
 *
 * @version     2020
 * @date        19 February 2015, 16:22 (created) \n
 *              03 March    2020, 14:52 (revised)
 *
 * @detail
 * This is the data generator for PPP 2020 projects
 */

#include <string>
#include <cstdlib>
#include <getopt.h>

#include <hdf5.h>
#include <hdf5_hl.h>


using namespace std;

//----------------------------------------------------------------------------//
//------------------------- Data types declarations --------------------------//
//----------------------------------------------------------------------------//

/**
 * @struct TParameters
 * @brief  Parameters of the program
 */
struct TParameters
{
  string FileName;
  size_t Size;
  float  HeaterTemperature;
  float  CoolerTemperature;

  float dt;
  float dx;
};// end of Parameters
//------------------------------------------------------------------------------

/**
 * @struct MediumParameters
 * @brief Parameters of Medium
 */
struct TMediumParameters
{
  float k_s;     // W/(m K)  Thermal conductivity - conduction ciefficient
  float rho;     // kg.m^3   Density
  float Cp;      // J/kg K   Spefic heat constant pressure
  float alpha;   // m^2/s    Diffusivity

  TMediumParameters(const float k_s, const float rho, const float Cp)
                   : k_s(k_s), rho(rho), Cp(Cp)
  {
    alpha = k_s / (rho * Cp);
  }

  // Calculate coef F0 - heat diffusion parameter
  inline float GetF0(const float dx, const float dt) const
  {
    return alpha * dt / (dx * dx) ;
  }

  /// Check stability of the simulation for the medium
  inline bool CheckStability(const float dx, const float dt) const
  {
    return (GetF0(dx, dt) < 0.25f);
  }
};// end of TMediumParameters
//------------------------------------------------------------------------------

//----------------------------------------------------------------------------//
//-------------------------    Global variables        -----------------------//
//----------------------------------------------------------------------------//

constexpr size_t MaskSize   =  16;  // size of the mask
constexpr float  DomainSize =  1.f; // length of the edge 1m


///  Basic mask of the cooler (0 - Air, 1 -aluminum, 2 - copper)
int CoolerMask[MaskSize * MaskSize]
//1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
{ 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0,   //16
  0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0,   //15
  0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0,   //14
  0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0,   //13
  0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0,   //12
  0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0,   //11
  0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0,   //10
  0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0,   // 9
  0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0,   // 8
  0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0,   // 7
  0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0,   // 6
  0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0,   // 5
  0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0,   // 4
  0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,   // 3
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   // 2
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0    // 1
};


/// Parameters of the medium
TParameters Parameters {"ppp_input_data.h5", 16u, 100.f, 20.f};

/// Properties of Air
TMediumParameters Air(0.0024f, 1.207f, 1006.1f);

/// Properties of Aluminum
TMediumParameters Aluminum(205.f, 2700.f, 910.f);
//Aluminum.SetValues()

/// Properties of Copper
TMediumParameters Copper(387.f, 8940.f, 380.f);



//----------------------------------------------------------------------------//
//------------------------- Function declarations ----------------------------//
//----------------------------------------------------------------------------//

/// Print usage of the program and exit
void PrintUsageAndExit();

/// Set parameters
void ParseCommandline(int argc, char** argv);

/// Generate data for the matrix
void GenerateData(int DomainMap[], float DomainParameters[]);

/// Store data in the file
void StoreData();

// Get dx
float Getdx();

//----------------------------------------------------------------------------//
//------------------------- Function implementation  -------------------------//
//----------------------------------------------------------------------------//


/**
 * Print usage of the program
 */
void PrintUsageAndExit()
{
  printf("--------------------------- Usage --------------------------------\n");
  printf("  -o <string>             : Output file name with the medium data \n");
  printf("  -N <int>                : Size of the domain                    \n");
  printf("                          : Power of 2 only!                      \n");
  printf("  -H <float>              : Heater temperature °C                 \n");
  printf("  -C <float>              : Air    temperature °C                 \n");
  printf("  -h                      : help                                  \n");

  exit(EXIT_SUCCESS);
}// end of PrintUsage
//------------------------------------------------------------------------------
  
/**
 * Parse commandline and setup
 * @param [in] argc
 * @param [in] argv
 */
void ParseCommandline(int argc, char** argv)
{
  char c;
  const char * shortOpts = "o:N:H:C:h";

  while ((c = getopt (argc, argv, shortOpts)) != -1)
  {
    switch (c)
    {
      case 'o':
      { // input file
        if ((optarg == NULL))  PrintUsageAndExit();
        Parameters.FileName = optarg;
        break;
      } // i

      case 'N':
      {
        if ((optarg == NULL) || (atoi(optarg) <= 0)) PrintUsageAndExit();

        size_t size = atol(optarg);
        if (!((size != 0) && !(size & (size - 1))))
        {
          // not power of two
          printf ("Error: The size is not power of two (%lu)\n", size);
          PrintUsageAndExit();
        }
        Parameters.Size = size;
        if (Parameters.Size < 16)
        {
          printf("Minimum size is 16");
          PrintUsageAndExit();
        }
        break;
      }//

      case 'H':
      {
        if ((optarg == NULL) || (atoi(optarg) <= 0)) PrintUsageAndExit();
        Parameters.HeaterTemperature = atof(optarg);
        break;
      }

      case 'C':
      {
        if ((optarg == NULL) || (atoi(optarg) <= 0)) PrintUsageAndExit();
        Parameters.CoolerTemperature = atof(optarg);
        break;
      }
      case 'h':
      {
        PrintUsageAndExit();
        break;
      }

      default :
      {
        PrintUsageAndExit();
        break;
      }

    }// switch
  }// while


  if (Parameters.Size < 128  ) Parameters.dt       =    0.1f;
  else if (Parameters.Size < 512  ) Parameters.dt  =   0.01f;
  else if (Parameters.Size < 2048 ) Parameters.dt  =  0.001f;
  else if (Parameters.Size < 16384) Parameters.dt  = 0.0001f;
  else Parameters.dt  = 0.00001f;

  Parameters.dx = DomainSize / Parameters.Size;
}// end of ParseCommandline
//------------------------------------------------------------------------------


/**
 * Generate data for the domain
 * @param [out] DomainMap
 * @param [out] DomainParameters
 * @param [out] InitialTemperature
 */
void GenerateData(int * DomainMap, float * DomainParameters, float * InitialTemperature)
{
  const size_t ScaleFactor = Parameters.Size / MaskSize;


  // set the global medium map
  #pragma omp parallel
  {
    #pragma omp for
    for (size_t m_y = 0; m_y < MaskSize; m_y++)
    {
      for (size_t m_x = 0; m_x < MaskSize; m_x++)
      {
        // Scale
        for (size_t y = 0; y < ScaleFactor; y++)
        {
          for (size_t x = 0; x < ScaleFactor; x++)
          {
            size_t global = (m_y * ScaleFactor + y)* Parameters.Size + (m_x * ScaleFactor + x);
            size_t local = m_y * MaskSize + m_x;

            DomainMap[global]  = CoolerMask[local];
            //
          }// x
        }// y
      } // m_x
    }// m_y


    // set medium properties
    #pragma omp for
    for (size_t y = 0; y < Parameters.Size; y++)
    {
      for (size_t x = 0; x < Parameters.Size; x++)
      {
        switch(DomainMap[y * Parameters.Size + x])
        {
          case 0: DomainParameters[y * Parameters.Size + x] = Air.GetF0(Parameters.dx, Parameters.dt); break;
          case 1: DomainParameters[y * Parameters.Size + x] = Aluminum.GetF0(Parameters.dx, Parameters.dt); break;
          case 2: DomainParameters[y * Parameters.Size + x] = Copper.GetF0(Parameters.dx, Parameters.dt); break;
        }
      }
    }

    // set initial temperature (skip first two lines)  - that's the heater
    #pragma omp for
    for (size_t y = 2; y < Parameters.Size; y++)
    {
      for (size_t x = 0; x < Parameters.Size; x++)
      {
        InitialTemperature[y * Parameters.Size + x] = Parameters.CoolerTemperature;
      }
    }
  }// end of parallel

  //set temperature for heater
  for (size_t x = 0; x < 2*Parameters.Size; x++)
  { // where is cooper, set Heater
    InitialTemperature[x] = (DomainMap[x] == 2) ? Parameters.HeaterTemperature : Parameters.CoolerTemperature;
  }
}// end of GenerateData
//------------------------------------------------------------------------------


/**
 * Store data in the file
 * @param [in] DomainMap
 * @param [in] DomainParameters
 * @param [in] InitialTemperature
 */
void StoreData(const int * DomainMap, const float * DomainParameters, const float * InitialTemperature)
{
  hid_t HDF5_File = H5Fcreate(Parameters.FileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  hsize_t ScalarSize[1] {1};
  hsize_t DomainSize[2] {Parameters.Size,Parameters.Size};

  long Size = long (Parameters.Size);

  H5LTmake_dataset_long(HDF5_File , "/EdgeSize"          , 1, ScalarSize, &Size);
  H5LTmake_dataset_float(HDF5_File, "/CoolerTemp"        , 1, ScalarSize, &Parameters.CoolerTemperature);
  H5LTmake_dataset_float(HDF5_File, "/HeaterTemp"        , 1, ScalarSize, &Parameters.HeaterTemperature);
  H5LTmake_dataset_int (HDF5_File , "/DomainMap"         , 2, DomainSize, DomainMap);
  H5LTmake_dataset_float(HDF5_File, "/DomainParameters"  , 2, DomainSize, DomainParameters);
  H5LTmake_dataset_float(HDF5_File, "/InitialTemperature", 2, DomainSize, InitialTemperature);

  H5Fclose(HDF5_File);
}// end of StoreData
//------------------------------------------------------------------------------

/**
 * main function
 * @param [in] argc
 * @param [in] argv
 * @return
 */
int main(int argc, char** argv)
{
  ParseCommandline(argc,argv);

  printf("---------------------------------------------\n");
  printf("--------- PPP 2020 data generator -----------\n");
  printf("File name  : %s\n",   Parameters.FileName.c_str());
  printf("Size       : [%ld,%ld]\n",   Parameters.Size, Parameters.Size);
  printf("Heater temp: %.2fC\n", Parameters.HeaterTemperature);
  printf("Cooler temp: %.2f\n", Parameters.CoolerTemperature);

  int   * DomainMap          = new int  [Parameters.Size * Parameters.Size];
  float * DomainParameters   = new float[Parameters.Size * Parameters.Size];
  float * InitialTemperature = new float[Parameters.Size * Parameters.Size];

  printf("Air      : %f\n", Air.GetF0(Parameters.dx,Parameters.dt));
  printf("Aluminum : %f\n", Aluminum.GetF0(Parameters.dx,Parameters.dt));
  printf("Copper   : %f\n", Copper.GetF0(Parameters.dx,Parameters.dt));

  if (!(Copper.CheckStability(Parameters.dx,Parameters.dt) &&
      Aluminum.CheckStability(Parameters.dx,Parameters.dt) &&
      Air.CheckStability(Parameters.dx,Parameters.dt))
     )
  {
    printf("dt and dx are too big, simulation may be unstable! \n");
  }

  printf("Generating data....");fflush(stdout);
  GenerateData(DomainMap, DomainParameters, InitialTemperature);
  printf("Done\n");fflush(stdout);
  printf("Storing data ....");fflush(stdout);
  StoreData(DomainMap, DomainParameters, InitialTemperature);
  printf("Done\n");fflush(stdout);

  delete [] DomainMap;
  delete [] DomainParameters;
  delete [] InitialTemperature;

  return 0;
}// end of main
//------------------------------------------------------------------------------
