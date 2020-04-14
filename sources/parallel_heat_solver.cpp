/**
 * @file    parallel_heat_solver.cpp
 * @author  xkaras34 <xkaras34@stud.fit.vutbr.cz>
 *
 * @brief   Course: PPP 2019/2020 - Project 1
 *
 * @date    2020-04-15
 */

#include "parallel_heat_solver.h"

ParallelHeatSolver::ParallelHeatSolver(SimulationProperties &simulationProps,
									   MaterialProperties &materialProps)
	: BaseHeatSolver (simulationProps, materialProps)
	, mFileHandle(H5I_INVALID_HID, static_cast<void (*)(hid_t )>(nullptr))
{
	MPI_Comm_size(MPI_COMM_WORLD, &mSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &mRank);
	
	simulationProps.GetDecompGrid(mGridSize.x, mGridSize.y);
	mTileSize = { 
		materialProps.GetEdgeSize() / mGridSize.x,
		materialProps.GetEdgeSize() / mGridSize.y
	};

	// Create tile counts and displacements
	{
		mTileCounts = std::vector<int>(mGridSize.x * mGridSize.y, 1);

		size_t globDisp = 0;
		for (int y = 0; y < mGridSize.y; ++y)
		{
			size_t locDisp = 0;
			for (int x = 0; x < mGridSize.x; ++x)
			{
				mTileDisplacements.emplace_back(globDisp + locDisp);
				locDisp += mTileSize.x;
			}
			globDisp += mTileSize.x * mTileSize.y * mGridSize.x;
		}
	}
	
	if (!m_simulationProperties.GetOutputFileName().empty())
	{
		if (simulationProps.IsUseParallelIO())
		{
			const auto& path = simulationProps.GetOutputFileName("par").c_str();
			AutoHandle<hid_t> plist_id(H5Pcreate(H5P_FILE_ACCESS), H5Pclose);
			H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

			mFileHandle.Set(H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id), H5Fclose);
		}
		else
		{
			if (isRoot())
			{
				const auto& path = simulationProps.GetOutputFileName("seq").c_str();
				mFileHandle.Set(H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
			}
		}
	}	
	
	createTypes();
	scatterData();	
}

ParallelHeatSolver::~ParallelHeatSolver()
{
	// MPI_Group_free(&mGroupNeighbors);
	if (mTemperatureComm != MPI_COMM_NULL)
		MPI_Comm_free(&mTemperatureComm);

	for (auto& w : mWindows)
		MPI_Win_free(&w);	

	MPI_Type_free(&mTWorkerTile);
	MPI_Type_free(&mTWorkerTileInt);
	MPI_Type_free(&mTTileEastWest);
	MPI_Type_free(&mTTileNorthSouth);

	if (isRoot())
	{
		MPI_Type_free(&mTTileRoot);
		MPI_Type_free(&mTTileResized);
	}
}

void ParallelHeatSolver::RunSolver(std::vector<float, AlignedAllocator<float> > &outResult)
{
	size_t frontBuff = 0;
	size_t backBuff = 1;
	double startTime = MPI_Wtime();

	auto computeHalo = [&](Vec2s start, Vec2s end, Direction dir) 
	{
		for (size_t y = start.y; y < end.y; ++y)
			for (size_t x = start.x; x < end.x; ++x)
				ComputePoint(
					mBuffers[frontBuff].data(),
					&mBuffers[backBuff][0],
					mDomainParams.data(),
					mDomainMap.data(),
					y, x,
					mTileSize.x + 4,
					m_simulationProperties.GetAirFlowRate(),
					m_materialProperties.GetCoolerTemp()
				);

		sendHaloZone(&mBuffers[backBuff][0], mWindows[backBuff], dir);
	};

	for(size_t iter = 0; iter < m_simulationProperties.GetNumIterations(); ++iter)
	{
		// Just open the window for communication
		MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOPRECEDE, mWindows[backBuff]); // TODO check MPI_MODE_NOPRECEDE

		#pragma omp parallel sections // TODO this won't work in parallel most likely
		{
			#pragma omp section // Up
			{
				if (mRank >= mGridSize.x)
				{
					Vec2s start = { 2, 2 };
					Vec2s end = { 2 + mTileSize.x, 4 };

					if (mRank % mGridSize.x == 0) start.x = 4;
					if (mRank % mGridSize.x == mGridSize.x - 1) end.x = mTileSize.x;

					computeHalo(start, end, Direction::Up);
				}
			}
			
			#pragma omp section // Left
			{
				if (mRank % mGridSize.x != 0)
				{
					Vec2s start = { 2, 2 };
					Vec2s end = { 4, 2 + mTileSize.y };

					if (mRank < mGridSize.x) start.y = 4;
					if (mRank >= mGridSize.x * (mGridSize.y - 1)) end.y = mTileSize.y;

					computeHalo(start, end, Direction::Left);
				}
			}

			#pragma omp section // Right
			{
				if (mRank % mGridSize.x != mGridSize.x - 1)
				{
					Vec2s start = { mTileSize.x, 2 };
					Vec2s end = { 2 + mTileSize.x, 2 + mTileSize.y };

					if (mRank < mGridSize.x) start.y = 4;
					if (mRank >= mGridSize.x * (mGridSize.y - 1)) end.y = mTileSize.y;

					computeHalo(start, end, Direction::Right);
				}
			}

			#pragma omp section // Down
			{
				if (mRank < mGridSize.x * (mGridSize.y - 1))
				{
					Vec2s start = { 2, mTileSize.y };
					Vec2s end = { 2 + mTileSize.x, 2 + mTileSize.y };

					if (mRank % mGridSize.x == 0) start.x = 4;
					if (mRank % mGridSize.x == mGridSize.x - 1) end.x = mTileSize.x;

					computeHalo(start, end, Direction::Down);
				}
			}
		}

		UpdateTile(
			mBuffers[frontBuff].data(),
			&mBuffers[backBuff][0],
			mDomainParams.data(),
			mDomainMap.data(),
			4, 4, 
			mTileSize.x - 4, mTileSize.y - 4,
			mTileSize.x + 4,
			m_simulationProperties.GetAirFlowRate(),
			m_materialProperties.GetCoolerTemp()
		);

		// Wait for all the halo zone transfers
		MPI_Win_fence(0, mWindows[backBuff]); // TODO this is gonna be pain (maybe use groups)

		// Save data to file
		if (!m_simulationProperties.GetOutputFileName().empty() && (iter % m_simulationProperties.GetDiskWriteIntensity()) == 0)
			save(mBuffers[backBuff], iter);

		// Print progress
		if (ShouldPrintProgress(iter))
		{
			auto temp = computeMiddleTemp(mBuffers[backBuff]);

			if (isRoot())
				PrintProgressReport(iter, temp);
		}
		
		// Swap buffers 
		std::swap(frontBuff, backBuff);
	}

	gatherData(mBuffers[frontBuff], outResult);

	// Measure time and print results
	auto temperature = computeMiddleTemp(mBuffers[frontBuff]);
	if (isRoot())
	{
		double elapsedTime = MPI_Wtime() - startTime;
		PrintFinalReport(elapsedTime, temperature, "par");
	}
}

void ParallelHeatSolver::scatterData()
{
	const auto& temps = m_materialProperties.GetInitTemp();
	const auto& domainMap = m_materialProperties.GetDomainMap();
	const auto& domainParams = m_materialProperties.GetDomainParams();

	// Resize buffers
	auto tileSizeCount = (mTileSize.x + 4) * (mTileSize.y + 4);
	mBuffers[0].resize(tileSizeCount);
	mDomainMap.resize(tileSizeCount);
	mDomainParams.resize(tileSizeCount);

	auto scatter = [&](const void* what, void* where, MPI_Datatype& type)
	{
		MPI_Scatterv(
			what, 
			mTileCounts.data(),
			mTileDisplacements.data(),
			mTTileResized,
			where, 
			1,
			type,
			ROOT,
			MPI_COMM_WORLD
		);
	};

	auto fillHaloZones = [&](float* what, float* where)
	{
		MPI_Win win;
		MPI_Info info;
		auto winsize = sizeof(float) * (mTileSize.y + 4) * (mTileSize.x + 4);

		MPI_Info_create(&info);
		MPI_Info_set(info, "same_size", "true");
		MPI_Info_set(info, "same_disp_unit", "true");
		MPI_Info_set(info, "no_lock", "true");

		MPI_Win_create(where, winsize, sizeof(float), info, MPI_COMM_WORLD, &win);
		MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOPRECEDE, win);

		for (size_t i = 0; i < static_cast<size_t>(Direction::Count); ++i)
			sendHaloZone(what, win, static_cast<Direction>(i));
	
		MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOSUCCEED, win);

		MPI_Win_free(&win);
		MPI_Info_free(&info);
	};

	// Scatter temperature data
	scatter(temps.data(), &mBuffers[0][0], mTWorkerTile);
	fillHaloZones(&mBuffers[0][0], &mBuffers[0][0]);

	// Scatter params data
	scatter(domainParams.data(), &mDomainParams[0], mTWorkerTile);
	fillHaloZones(&mDomainParams[0], &mDomainParams[0]);
	
	// Scatter map data
	scatter(domainMap.data(), &mDomainMap[0], mTWorkerTileInt);

	mBuffers[1] = Bufferf(mBuffers[0].begin(), mBuffers[0].end());

	// Create windows
	{
		MPI_Info info;
		MPI_Info_create(&info);
		MPI_Info_set(info, "same_size", "true");
		MPI_Info_set(info, "same_disp_unit", "true");
		MPI_Info_set(info, "no_lock", "true"); // TODO check if it really works
		auto winsize = sizeof(float) * (mTileSize.y + 4) * (mTileSize.x + 4);

		for (size_t i = 0; i < mBuffers.size(); ++i)
			MPI_Win_create(&mBuffers[i][0], winsize, sizeof(float), info, MPI_COMM_WORLD, &mWindows[i]);

		MPI_Info_free(&info);
	}
}

void ParallelHeatSolver::gatherData(const Bufferf& what, Bufferf& where)
{
	if (isRoot())
		where.resize(m_materialProperties.GetInitTemp().size());

	MPI_Gatherv(
		what.data(),
		1,
		mTWorkerTile,
		&where[0],
		mTileCounts.data(),
		mTileDisplacements.data(),
		mTTileResized,
		ROOT,
		MPI_COMM_WORLD			
	);
}

void ParallelHeatSolver::createTypes()
{
	// Create worker tiles
	{
		int size[2] = { static_cast<int>(mTileSize.y) + 4, static_cast<int>(mTileSize.x) + 4 };
		int tile[2] = { static_cast<int>(mTileSize.y), static_cast<int>(mTileSize.x) };
		int start[2] = { 2, 2 };

		MPI_Type_create_subarray(2, size, tile, start, MPI_ORDER_C, MPI_FLOAT, &mTWorkerTile);
		MPI_Type_commit(&mTWorkerTile);

		// MPI_Type_contiguous(mTileSize.x * mTileSize.y, MPI_INT, &mTWorkerTileInt);
		MPI_Type_create_subarray(2, size, tile, start, MPI_ORDER_C, MPI_INT, &mTWorkerTileInt);
		MPI_Type_commit(&mTWorkerTileInt);
	}

	// Create root tile for gathering/scattering data
	if (isRoot())
	{
		int size[2] = { static_cast<int>(mTileSize.y * mGridSize.y), static_cast<int>(mTileSize.x * mGridSize.x) };
		int tile[2] = { static_cast<int>(mTileSize.y), static_cast<int>(mTileSize.x) };
		int start[2] = { 0, 0 };
		
		MPI_Type_create_subarray(2, size, tile, start, MPI_ORDER_C, MPI_FLOAT, &mTTileRoot);
		MPI_Type_commit(&mTTileRoot);
		MPI_Type_create_resized(mTTileRoot, 0, sizeof(int), &mTTileResized);
		MPI_Type_commit(&mTTileResized);
	}

	// Create 2xN / Nx2 tiles for Halo Zones
	MPI_Type_vector(mTileSize.y + 4, 2, mTileSize.x + 4, MPI_FLOAT, &mTTileEastWest);
	MPI_Type_commit(&mTTileEastWest);
	MPI_Type_contiguous((mTileSize.x + 4) * 2, MPI_FLOAT, &mTTileNorthSouth);
	MPI_Type_commit(&mTTileNorthSouth);

	// Create temperature communicator
	{
		auto middleRank = (m_materialProperties.GetEdgeSize() / 2 - 1) / mTileSize.x + 1;
		std::vector<int> ranks;

		for (int i = middleRank; i < mSize; i += mGridSize.x)
			ranks.emplace_back(i);

		MPI_Group temp;
		MPI_Comm_group(MPI_COMM_WORLD, &temp);
		MPI_Group_incl(temp, ranks.size(), ranks.data(), &temp); 
		MPI_Comm_create(MPI_COMM_WORLD, temp, &mTemperatureComm);
		MPI_Group_free(&temp);

		if (mTemperatureComm != MPI_COMM_NULL)
			MPI_Comm_rank(mTemperatureComm, &mTempRank);
	}
}

void ParallelHeatSolver::sendHaloZone(float* data, MPI_Win& win, Direction dir)
{
	auto idx = [this](int x, int y) { return y * (mTileSize.x + 4) + x; };
	auto tx = mTileSize.x;
	auto ty = mTileSize.y;

	switch (dir)
	{
	case Direction::Up: 
		if (mRank >= mGridSize.x) // Up
			MPI_Put(&data[idx(0, 2)], 1, mTTileNorthSouth, mRank - mGridSize.x, idx(0, ty + 2), 1, mTTileNorthSouth, win);
		break;

	case Direction::Left:
		if (mRank % mGridSize.x != 0) // Left
			MPI_Put(&data[idx(2, 0)], 1, mTTileEastWest, mRank - 1, idx(tx + 2, 0), 1, mTTileEastWest, win);
		break;

	case Direction::Right:
		if (mRank % mGridSize.x != mGridSize.x - 1) // Right
			MPI_Put(&data[idx(tx, 0)], 1, mTTileEastWest, mRank + 1, idx(0, 0), 1, mTTileEastWest, win);
		break;

	case Direction::Down:
		if (mRank < mGridSize.x * (mGridSize.y - 1)) // Down
			MPI_Put(&data[idx(0, ty)], 1, mTTileNorthSouth, mRank + mGridSize.x, idx(0, 0), 1, mTTileNorthSouth, win);
		break;
	default: break;
	}
}

void ParallelHeatSolver::save(const Bufferf& data, size_t iteration)
{
	if (m_simulationProperties.IsUseParallelIO())
	{
		hsize_t gridSize[] = { m_materialProperties.GetEdgeSize(), m_materialProperties.GetEdgeSize() };
		hsize_t chunkSize[] = { mTileSize.x, mTileSize.y };
		hsize_t tileSize[] = { mTileSize.x + 4, mTileSize.y + 4 };

		auto groupNum = static_cast<unsigned long long>(iteration / m_simulationProperties.GetDiskWriteIntensity());
		std::string groupName = "Timestep_" + std::to_string(groupNum);
		
		AutoHandle<hid_t> groupHandle(H5Gcreate(mFileHandle, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose);
		
		// Create dataset
		{
			std::string dataSetName("Temperature");

			// Enable chunks for collective writing
			AutoHandle<hid_t> propertyList(H5Pcreate(H5P_DATASET_CREATE), H5Pclose);
			H5Pset_chunk(propertyList, 2, chunkSize);
		
			AutoHandle<hid_t> fileSpaceHandle(H5Screate_simple(2, gridSize, NULL), H5Sclose);
			AutoHandle<hid_t> memSpaceHandle(H5Screate_simple(2, tileSize, NULL), H5Sclose);

			AutoHandle<hid_t> dataSetHandle(
				H5Dcreate(
					groupHandle, dataSetName.c_str(), H5T_NATIVE_FLOAT, fileSpaceHandle, H5P_DEFAULT, propertyList, H5P_DEFAULT
				), 
				H5Dclose
			);

			// create memory layout
			{
				const hsize_t start[] = { 2, 2 };
				const hsize_t count[] = { mTileSize.y, mTileSize.x };
				// const hsize_t stride[] = { mTileSize.y, mTileSize.x };
				// const hsize_t block[] = { mTileSize.y, mTileSize.x };
				H5Sselect_hyperslab(memSpaceHandle, H5S_SELECT_SET, start, nullptr, count, nullptr);
			}

			// create file layout
			{
				const hsize_t start[] = { mTileSize.y * (mRank / mGridSize.x), mTileSize.x * (mRank % mGridSize.x) };
				const hsize_t count[] = { mTileSize.y, mTileSize.x };
				// const hsize_t stride[] = { mTileSize.y, mTileSize.x };
				// const hsize_t block[] = { mTileSize.y, mTileSize.x };
				H5Sselect_hyperslab(fileSpaceHandle, H5S_SELECT_SET, start, nullptr, count, nullptr);
			}

			AutoHandle<hid_t> writeProp(H5Pcreate(H5P_DATASET_XFER), H5Pclose);
			H5Pset_dxpl_mpio(writeProp, H5FD_MPIO_COLLECTIVE);

			H5Dwrite(dataSetHandle, H5T_NATIVE_FLOAT, memSpaceHandle, fileSpaceHandle, writeProp, data.data());
		}

		// Write timestamp
		{
			std::string attributeName("Time");

			AutoHandle<hid_t> dataSpaceHandle(H5Screate(H5S_SCALAR), H5Sclose);
			AutoHandle<hid_t> attributeHandle(
				H5Acreate2(
					groupHandle, attributeName.c_str(), H5T_IEEE_F64LE, dataSpaceHandle, H5P_DEFAULT, H5P_DEFAULT
				), 
				H5Aclose
			);

			double snapshotTime = double(iteration);
			H5Awrite(attributeHandle, H5T_IEEE_F64LE, &snapshotTime);
		}
	}
	else
	{
		Bufferf tempData;
		gatherData(data, tempData);

		if (mFileHandle != H5I_INVALID_HID && isRoot())
			StoreDataIntoFile(mFileHandle, iteration, data.data());
	}
}

float ParallelHeatSolver::computeMiddleTemp(const Bufferf& data)
{
	auto middlePos = (m_materialProperties.GetEdgeSize() / 2) % mTileSize.x + 2;
	float localTemp = 0;
	float globalTemp;

	if (mTemperatureComm != MPI_COMM_NULL)
	{
		// compute local sum
		for (size_t y = 2; y < mTileSize.y + 2; ++y)
			localTemp += data[middlePos + y * (mTileSize.x + 4)];

		MPI_Reduce(&localTemp, &globalTemp, 1, MPI_FLOAT, MPI_SUM, ROOT, mTemperatureComm);

		// Send only if this process ain't global root
		if (mTempRank == ROOT && !isRoot()) 
			MPI_Send(&globalTemp, 1, MPI_FLOAT, ROOT, 42, MPI_COMM_WORLD);
	}

	// Recieve only if this process is global root
	if (isRoot())
	{
		if (mTempRank != ROOT)
			MPI_Recv(&globalTemp, 1, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		globalTemp /= m_materialProperties.GetEdgeSize();
	}

	return globalTemp;
}
