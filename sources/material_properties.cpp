/**
 * @file    material_properties.cpp
 * @authors Filip Vaverka <ivaverka@fit.vutbr.cz>
 *          Jiri Jaros <jarosjir@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2019/2020 - Project 1
 *          This file contains class which represent materials in the simulation
 *          domain.
 *
 * @date    2018-03-28
 */

#include "material_properties.h"

#include <hdf5.h>

using namespace std;

void MaterialProperties::LoadMaterialData(const string &fileName, bool loadData)
{
    try
    {
        // 1. Open the input file for reading only.
        AutoHandle<hid_t> fileHandle(H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
        if(fileHandle < 0)
            throw ios::failure("Cannot open input file");

        {
            // 2. Read "/EdgeSize" dataset which is scalar and contains size of the
            //    domain.
            AutoHandle<hid_t> datasetHandle(H5Dopen(fileHandle, "/EdgeSize", H5P_DEFAULT), H5Dclose);
            if(datasetHandle == H5I_INVALID_HID)
                throw ios::failure("Cannot open dataset EdgeSize");
            H5Dread(datasetHandle, H5T_STD_I64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_edgeSize);
        }

        m_nGridPoints = m_edgeSize * m_edgeSize;

        {
            // 3. Read "/CoolerTemp" dataset which is scalar containing temperature
            //    of the cooler.
            AutoHandle<hid_t> datasetHandle(H5Dopen(fileHandle, "/CoolerTemp", H5P_DEFAULT), H5Dclose);
            if(datasetHandle == H5I_INVALID_HID)
                throw ios::failure("Cannot open dataset CoolerTemp");
            H5Dread(datasetHandle, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_coolerTemp);
        }

        {
            // 4. Read "/HeaterTemp" dataset which is scalar containing temperature
            //    of the heater element (CPU).
            AutoHandle<hid_t> datasetHandle(H5Dopen(fileHandle, "/HeaterTemp", H5P_DEFAULT), H5Dclose);
            if(datasetHandle == H5I_INVALID_HID)
                throw ios::failure("Cannot open dataset HeaterTemp");
            H5Dread(datasetHandle, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_heaterTemp);
        }

        // Do we want to actually read the data?
        if(loadData)
        {
            // Allocate memory for data in the domain.
            m_domainMap.resize(m_nGridPoints);
            m_domainParams.resize(m_nGridPoints);
            m_initTemp.resize(m_nGridPoints);

            {
                // 5. Read "/DomainMap" which contains material specifications.
                AutoHandle<hid_t> datasetHandle(H5Dopen(fileHandle, "/DomainMap", H5P_DEFAULT), H5Dclose);
                if(datasetHandle == H5I_INVALID_HID)
                    throw ios::failure("Cannot open dataset DomainMap");
                H5Dread(datasetHandle, H5T_STD_I32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, m_domainMap.data());
            }

            {
                // 6. Read "/DomainParameters" which contains thermal properties of the material.
                AutoHandle<hid_t> datasetHandle(H5Dopen(fileHandle, "/DomainParameters", H5P_DEFAULT), H5Dclose);
                if(datasetHandle == H5I_INVALID_HID)
                    throw ios::failure("Cannot open dataset DomainParameters");
                H5Dread(datasetHandle, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, m_domainParams.data());
            }

            {
                // 7. Read "/InitialTemperature" which contains initial temperature distribution.
                AutoHandle<hid_t> datasetHandle(H5Dopen(fileHandle, "/InitialTemperature", H5P_DEFAULT), H5Dclose);
                if(datasetHandle == H5I_INVALID_HID)
                    throw ios::failure("Cannot open dataset InitialTemperature");
                H5Dread(datasetHandle, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, m_initTemp.data());
            }
        }
    }
    catch(...)
    {
        // If any of the operations failed, print an error message and exit the
        // application.
        std::cerr << "Wrong input material file: " << fileName << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
}
