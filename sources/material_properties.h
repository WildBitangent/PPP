/**
 * @file    material_properties.h
 * @authors Filip Vaverka <ivaverka@fit.vutbr.cz>
 *          Jiri Jaros <jarosjir@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2019/2020 - Project 1
 *          This file contains class which represent materials in the simulation
 *          domain.
 *
 * @date    2018-03-28
 */

#ifndef MATERIAL_PROPERTIES_H
#define MATERIAL_PROPERTIES_H

#include <string>
#include <vector>

#include "base.h"

/**
 * @brief The MaterialProperties class represents the simulation domain and its
 *        contents.
 */
class MaterialProperties
{
public:
    MaterialProperties() {
    }

    // The object should not be copied.
    MaterialProperties(const MaterialProperties &) = delete;
    MaterialProperties &operator=(const MaterialProperties &) = delete;

    /**
     * @brief Load domain information from the input material file.
     * @param fileName Path to material file.
     * @param loadData Flag which specifies if we want actually load contents
     *                 of the domain (materials) or just meta-data.
     *                 NOTE: When "false" is specified, vectors: DomainMap,
     *                       DomainParams and InitTemp stay empty!
     */
    void LoadMaterialData(const std::string &fileName, bool loadData);

    inline std::vector<int, AlignedAllocator<int> > &GetDomainMap() { return m_domainMap; }
    inline std::vector<float, AlignedAllocator<float> > &GetDomainParams() { return m_domainParams; }
    inline std::vector<float, AlignedAllocator<float> > &GetInitTemp() { return m_initTemp; }

    inline float GetCoolerTemp() const { return m_coolerTemp; }
    inline float GetHeaterTemp() const { return m_heaterTemp; }
    inline size_t GetEdgeSize() const { return m_edgeSize; }
    inline size_t GetGridPoints() const { return m_nGridPoints; }

protected:
    /**
     * @brief Domain Map - defines type of the material at every gridpoint
     *        0 - air, 1 - aluminium, 2 - copper.
     */
    std::vector<int, AlignedAllocator<int> > m_domainMap;
    std::vector<float, AlignedAllocator<float> > m_domainParams; ///< Thermal properties of the medium.
    std::vector<float, AlignedAllocator<float> > m_initTemp;     ///< Initial temperature distribution.

    float m_coolerTemp;     ///< Temperature of the air.
    float m_heaterTemp;     ///< Temperature of the heater.
    size_t m_edgeSize;      ///< Size of the domain.
    size_t m_nGridPoints;   ///< Total number of gridpoint in the domain.
};

#endif // MATERIAL_PROPERTIES_H
