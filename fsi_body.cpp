#include <assert.h>
#include <stdlib.h>
#include <ctime>

#include "chrono/physics/ChSystemSMC.h"
#include "chrono/physics/ChBodyEasy.h"
#include "chrono/physics/ChLinkMotorLinearPosition.h"

#include "chrono/fea/ChElementShellANCF.h"
#include "chrono/fea/ChLinkPointFrame.h"
#include "chrono/fea/ChMesh.h"
#include "chrono/fea/ChMeshExporter.h"

#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGenerators.h"
#include "chrono/utils/ChUtilsGeometry.h"

#include "chrono_pardisomkl/ChSolverPardisoMKL.h"

#include "chrono_fsi/ChSystemFsi.h"
#include "chrono_fsi/utils/ChUtilsGeneratorFsi.h"
#include "chrono_fsi/utils/ChUtilsPrintSph.cuh"
#include "chrono_fsi/utils/ChUtilsJSON.h"

using namespace chrono;
using namespace fea;

typedef fsi::Real Real;

const std::string outs_dir = "FSI_BODY/"; // Directory for outputs
std::string out_dir; // Directory for this simulation

// Parameters
double M_TO_L = 1e2; // cm
double KG_TO_W = 1e3; // 1 gram
double S_TO_T = 1e0; // 1s

// Fluid domain dimension
Real fxDim = 0.11 * M_TO_L;
Real fyDim = 0.15 * M_TO_L;
Real fzDim = 0.08 * M_TO_L;

// Simulation domain dimension
Real bxDim = fxDim;
Real byDim = fyDim;
Real bzDim = fzDim;

// Solid materials
double E = 2e8 * KG_TO_W / M_TO_L / S_TO_T / S_TO_T; // kg*m/s^2/m^2
double nu = 0.3;
double rhoSolid = 1500 * KG_TO_W / M_TO_L /  M_TO_L / M_TO_L;

// Solid Dimension
double beamLength = 0.03 * M_TO_L;
double beamThickness = 0.0005 * M_TO_L; // space / 2
double beamArcLength = 0.03 * M_TO_L;
double beamAngle = 150.0 * CH_C_PI / 180.0;
double beamRadius = beamArcLength / beamAngle;
double flapLength = 0.05 * M_TO_L;
double flapHeight = 0.04 * M_TO_L; // beamRadius * sin(beamAngle / 2) * 2

double freq = 2.0; // Swing frequency
double ts = 0.0 * S_TO_T; // Wait for fluid to settle
double amplitude = 0.0 * M_TO_L;

double wallOffset = 0.05 * M_TO_L;
double zOffset = 0.0 * M_TO_L;

class ChFunction_Motor : public ChFunction {
public:
	virtual ChFunction_Motor* Clone() const override {
		return new ChFunction_Motor();
	}

	virtual double Get_y(double x) const override {
		if (x < ts) return 0;

		return amplitude * sin((x - ts) * freq * 2 * CH_C_PI - CH_C_PI / 2) + amplitude;
	}
};

// Connectivity of beam mesh, only need to calculate once
void calcBeamCon(
	std::ostringstream& meshConBuffer,
	std::shared_ptr<fea::ChMesh> fin
) {
	char dataBuffer[256];
	int numEs = fin->GetNelements(); // Number of elements

	// Connectivity
	snprintf(dataBuffer, sizeof(char) * 256, "\nCELLS %d %d\n", numEs, numEs * 5); // 4 indices + 1 count = 5
	meshConBuffer << std::string(dataBuffer);
	for (int i = 0; i < numEs; i++) {
		auto e = std::dynamic_pointer_cast<ChElementShellANCF>(fin->GetElement(i));
		int nodeAIndex = e->GetNodeN(0)->GetIndex() - 1;
		int nodeBIndex = e->GetNodeN(1)->GetIndex() - 1;
		int nodeCIndex = e->GetNodeN(2)->GetIndex() - 1;
		int nodeDIndex = e->GetNodeN(3)->GetIndex() - 1;
		snprintf(dataBuffer, sizeof(char) * 256, "4 %d %d %d %d\n", nodeAIndex, nodeBIndex, nodeCIndex, nodeDIndex);
		meshConBuffer << std::string(dataBuffer);
	}

	// Type
	snprintf(dataBuffer, sizeof(char) * 256, "\nCELL_TYPES %d\n", numEs); // 4 indices + 1 count = 5
	meshConBuffer << std::string(dataBuffer);
	for (int i = 0; i < numEs; i++) {
		snprintf(dataBuffer, sizeof(char) * 256, "%d\n", 9);
		meshConBuffer << std::string(dataBuffer);
	}
}

void writeBeamVtk(
	std::ostringstream& meshConBuffer,
	std::shared_ptr<fea::ChMesh> fin,
	int frameCurrent
) {
	char dataBuffer[256];
	std::ostringstream meshBuffer;

	int numNs = fin->GetNnodes(); // Number of nodes

	// Header
	snprintf(dataBuffer, sizeof(char) * 256, "# vtk DataFile Version 2.0\nFin\nASCII\nDATASET UNSTRUCTURED_GRID\n");
	meshBuffer << std::string(dataBuffer);

	// Position
	snprintf(dataBuffer, sizeof(char) * 256, "POINTS %d float\n", numNs);
	meshBuffer << std::string(dataBuffer);
	for (int i = 0; i < numNs; i++) {
		auto node = std::dynamic_pointer_cast<ChNodeFEAxyzD>(fin->GetNode(i));
		snprintf(dataBuffer, sizeof(char) * 256, "%f %f %f\n", node->GetPos().x(), node->GetPos().y(), node->GetPos().z());
		meshBuffer << std::string(dataBuffer);
	}

	std::ofstream meshFile;
	snprintf(dataBuffer, sizeof(char) * 256, (out_dir + "/Fin%d.vtk").c_str(), frameCurrent);
	meshFile.open(dataBuffer);
	meshFile << meshBuffer.str();
	meshFile << meshConBuffer.str();
	meshFile.close();
}

void writeFlapVtk(
	std::shared_ptr<ChBodyEasyBox> flap,
	double xDim, double yDim, double zDim,
	int frameCurrent
) {
	char dataBuffer[256];
	std::ostringstream bodyBuffer;

	// Header
	snprintf(dataBuffer, sizeof(char) * 256, "# vtk DataFile Version 2.0\nFlap\nASCII\nDATASET UNSTRUCTURED_GRID\n");
	bodyBuffer << std::string(dataBuffer);
	
	// Position
	int numNs = 8;  // 8 nodes for VTK_VOXEL (Type 11)
	snprintf(dataBuffer, sizeof(char) * 256, "POINTS %d float\n", numNs);
	bodyBuffer << std::string(dataBuffer);

	std::vector<ChVector<>> nodes;
	nodes.push_back(ChFrame<>(flap->GetCoord()) * ChVector<>(-xDim / 2, -yDim / 2, -zDim / 2));
	nodes.push_back(ChFrame<>(flap->GetCoord()) * ChVector<>(xDim / 2, -yDim / 2, -zDim / 2));
	nodes.push_back(ChFrame<>(flap->GetCoord()) * ChVector<>(-xDim / 2, yDim / 2, -zDim / 2));
	nodes.push_back(ChFrame<>(flap->GetCoord()) * ChVector<>(xDim / 2, yDim / 2, -zDim / 2));
	nodes.push_back(ChFrame<>(flap->GetCoord()) * ChVector<>(-xDim / 2, -yDim / 2, zDim / 2));
	nodes.push_back(ChFrame<>(flap->GetCoord()) * ChVector<>(xDim / 2, -yDim / 2, zDim / 2));
	nodes.push_back(ChFrame<>(flap->GetCoord()) * ChVector<>(-xDim / 2, yDim / 2, zDim / 2));
	nodes.push_back(ChFrame<>(flap->GetCoord()) * ChVector<>(xDim / 2, yDim / 2, zDim / 2));

	for (int i = 0; i < numNs; i++) {
		snprintf(dataBuffer, sizeof(char) * 256, "%f %f %f\n", nodes[i].x(), nodes[i].y(), nodes[i].z());
		bodyBuffer << std::string(dataBuffer);
	}

	snprintf(dataBuffer, sizeof(char) * 256, "\nCELLS %d %d\n", 1, 9);
	bodyBuffer << std::string(dataBuffer);
	snprintf(dataBuffer, sizeof(char) * 256, "%d %d %d %d %d %d %d %d %d\n", 8, 0, 1, 2, 3, 4, 5, 6, 7);
	bodyBuffer << std::string(dataBuffer);

	snprintf(dataBuffer, sizeof(char) * 256, "\nCELL_TYPES %d\n", 1); 
	bodyBuffer << std::string(dataBuffer);
	snprintf(dataBuffer, sizeof(char) * 256, "%d\n", 11);
	bodyBuffer << std::string(dataBuffer);

	std::ofstream bodyFile;
	snprintf(dataBuffer, sizeof(char) * 256, (out_dir + "/Flap%d.vtk").c_str(), frameCurrent);
	bodyFile.open(dataBuffer);
	bodyFile << bodyBuffer.str();
	bodyFile.close();
}

void writeForce(
	std::shared_ptr<ChLinkMotorLinearPosition> joint,
	int frameCurrent
) {
	char dataBuffer[256];

	ChVector<> totalForce = joint->Get_react_force();
	//printf("\n%f, %f, %f\n", totalForce.x(), totalForce.y(), totalForce.z());

	std::ofstream dataFile;
	snprintf(dataBuffer, sizeof(char) * 256, (out_dir + "/Data.csv").c_str());
	if (frameCurrent == 0) {
		// New file and print title
		dataFile.open(dataBuffer);
		snprintf(dataBuffer, sizeof(char) * 256, "fx, fy, fz\n");
		dataFile << std::string(dataBuffer);
	}
	else {
		// Append data
		dataFile.open(dataBuffer, std::ios::app);
	}
	ChVector<> eulerAngle;
	snprintf(dataBuffer, sizeof(char) * 256, "%0.7f,%0.7f,%0.7f\n", totalForce.x(), totalForce.y(), totalForce.z());
	dataFile << std::string(dataBuffer);
	dataFile.close();
}


int main(int argc, char* argv[]) {
	// Create a Chrono physical system
	ChSystemSMC mphysicalSystem;
	fsi::ChSystemFsi myFsiSystem(mphysicalSystem);

	std::shared_ptr<fsi::SimParams> paramsH = myFsiSystem.GetSimParams();
	std::string inputJson = "../../chrono-simulation/fsi_body.json"; // loado input from source folder

	if (!fsi::utils::ParseJSON(inputJson, paramsH, fsi::mR3(bxDim, byDim, bzDim))) {
		printf("Invalid json\n");
		return 1;
	}

	// Space between particles
	Real space = paramsH->MULT_INITSPACE * paramsH->HSML;

	// Ceneter of boundary domain is the center of fluid
	// Leave spaces for wall BCE if needed
	double eps = space * 4;
	paramsH->cMin = chrono::fsi::mR3(-bxDim / 2 - eps, -byDim / 2 - eps,  - bzDim / 2 - eps) * 10;
	paramsH->cMax = chrono::fsi::mR3(bxDim / 2 + eps, byDim / 2 + eps, bzDim / 2 + eps) * 10;

	// Finalize params and output dir
	myFsiSystem.SetFluidDynamics(paramsH->fluid_dynamic_type);
	myFsiSystem.SetFluidSystemLinearSolver(paramsH->LinearSolver);

	fsi::utils::FinalizeDomain(paramsH);
	fsi::utils::PrepareOutputDir(paramsH, out_dir, outs_dir, inputJson);

	double flapThickness = beamRadius - beamRadius * cos(beamAngle / 2);
	double beamDx = -(beamLength + flapLength) / 2;
	double beamDy = -amplitude - beamRadius + flapThickness / 2;
	double beamDAngle = 0.0;
	double beamDz = zOffset; // Offset  midpoint of arc to center of container

	// Vector of beam axis  
	ChVector<> beamA1(beamDx, beamDy, beamDz); // One end point of beam axis
	ChVector<> beamA2(beamDx + beamLength, beamDy, beamDz); // The other end point of beam axis
	ChVector<> beamA = beamA2 - beamA1;
	beamA.Normalize(); // Unit vector along the beam axis direction

	// Flap dimension and position
	//printf("\n%f, %f, %f\n", flapLength, flapThickness, flapHeight);
	ChVector<> sizeHalfFlap(flapLength / 2, flapThickness / 2, flapHeight / 2);
	ChVector<> posFlap(beamLength / 2, beamDy + beamRadius - flapThickness / 2, beamDz);

	// Create Fluid region and discretize with SPH particles
	ChVector<> boxCenter(0.0, 0.0, 0.0);
	ChVector<> boxHalfDim(fxDim / 2, fyDim / 2, fzDim / 2);
	utils::GridSampler<> sampler(space);
	utils::Generator::PointVector points = sampler.SampleBox(boxCenter, boxHalfDim); // Use a chrono sampler to create a bucket of points

	// Add fluid particles from the sampler points to the FSI system
	int numPartAdded = 0;
	for (int i = 0; i < points.size(); i++) {
		// Calculate the pressure of a steady state (p = rho*g*h)
		//Real pre_ini = paramsH->rho0 * abs(paramsH->gravity.z) * (-points[i].z() + fzDim / 2);
		//Real rho_ini = paramsH->rho0 + pre_ini / (paramsH->Cs * paramsH->Cs);
		Real pre_ini = paramsH->BASEPRES;
		Real rho_ini = paramsH->rho0;

		// Skip if too close to body
		double gap = space * 1;
		// Beam
		bool noContact = true;
		ChVector<> pt = points[i] - beamA1;
		Real dot = pt ^ beamA; // Project length along beam axis
		Real cross = (pt % beamA).Length(); // Distance to beam beam axis
		double angle = atan2(pt.y(), pt.z());
		if (
			dot <= beamLength + gap && // Length
			dot >= 0 - gap &&
			cross <= beamRadius + gap && // Ring
			cross >= beamRadius - gap &&
			angle >= (CH_C_PI - beamAngle) / 2 + beamDAngle - gap / beamRadius && // Arc
			angle <= (CH_C_PI + beamAngle) / 2 + beamDAngle + gap / beamRadius
			) {
			//printf("\n%f, %f, %f\n", pt.x(), pt.y(), pt.z());
			//printf("\n%f, %f\n", dot, cross);
			noContact = false;
		}
		// Flap
		if (
			points[i].x() <= posFlap.x() + sizeHalfFlap.x() + gap && // Length
			points[i].x() >= posFlap.x() - sizeHalfFlap.x() - gap &&
			points[i].y() <= posFlap.y() + sizeHalfFlap.y() + gap && // Thickness
			points[i].y() >= posFlap.y() - sizeHalfFlap.y() - gap &&
			points[i].z() <= posFlap.z() + sizeHalfFlap.z() + gap && // Height
			points[i].z() >= posFlap.z() - sizeHalfFlap.z() - gap
			) {
			noContact = false;
		}

		if (noContact) {
			myFsiSystem.GetDataManager()->AddSphMarker(
				fsi::mR4(points[i].x(), points[i].y(), points[i].z(), paramsH->HSML), // x, y, z, radius
				fsi::mR3(1e-10), // Velocity
				fsi::mR4(rho_ini, pre_ini, paramsH->mu0, -1)); // density, pressure, viscosity		
			numPartAdded++;
		}


	}

	// Initialize phases
	// Seems to relate to interaction between fluid, rigid and flexible solid. 
	size_t numPhases = myFsiSystem.GetDataManager()->fsiGeneralData->referenceArray.size();
	if (numPhases != 0) {
		printf("Wrong number of phases %d\n", (int)numPhases);
		return -1;
	}
	else {
		myFsiSystem.GetDataManager()->fsiGeneralData->referenceArray.push_back(mI4(0, (int)numPartAdded, -1, -1));
		myFsiSystem.GetDataManager()->fsiGeneralData->referenceArray.push_back(mI4((int)numPartAdded, (int)numPartAdded, 0, 0));
	}

	// Create Solid
	ChVector<> gravity = ChVector<>(paramsH->gravity.x, paramsH->gravity.y, paramsH->gravity.z);
	//ChVector<> gravity = ChVector<>(0.0, 0.0, 0.0);
	mphysicalSystem.Set_G_acc(gravity);

	// Container body
	// Wall should be 1 space away from the boundary particles
	ChVector<> sizeHalfContainer(fxDim / 2 + space * 3, fyDim / 2 + space * 3, fzDim / 2 + space * 3 + wallOffset / 2);
	ChVector<> posContainer(0, 0, wallOffset / 2);
	auto container = chrono_types::make_shared<ChBodyEasyBox>(sizeHalfContainer.x() * 2, sizeHalfContainer.y() * 2, sizeHalfContainer.z() * 2, 0.0 * rhoSolid);
	container->SetPos(posContainer);
	container->SetBodyFixed(true);
	mphysicalSystem.AddBody(container);

	// Only add BCE on specified box surface, 12 means top, -12 means bottom, default 3 layers inward
	fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, container, chrono::VNULL, chrono::QUNIT, sizeHalfContainer, -12);
	//fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, container, chrono::VNULL, chrono::QUNIT, sizeHalfContainer, 12, false, true);
	fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, container, chrono::VNULL, chrono::QUNIT, sizeHalfContainer, 23, false, true);
	fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, container, chrono::VNULL, chrono::QUNIT, sizeHalfContainer, -23, false, true);
	fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, container, chrono::VNULL, chrono::QUNIT, sizeHalfContainer, 13, false, true);
	fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, container, chrono::VNULL, chrono::QUNIT, sizeHalfContainer, -13, false, true);

	// Slider
	auto slider = chrono_types::make_shared<ChBodyEasyBox>(0.0 * M_TO_L, 0.0 * M_TO_L, 0.0 * M_TO_L, 0.0 * rhoSolid);
	slider->SetPos(ChVector<>(0, 0, 0));
	mphysicalSystem.AddBody(slider);

	auto motor = chrono_types::make_shared<ChLinkMotorLinearPosition>();
	motor->Initialize(slider, container, ChFrame<>(ChVector<>(0, 0, 0), Q_from_AngZ(CH_C_PI / 2))); // rotate to y directon
	//auto motor_pos = chrono_types::make_shared<ChFunction_Sine>(0, 0.5, 0);
	auto motor_pos = chrono_types::make_shared<ChFunction_Motor>();
	motor->SetMotionFunction(motor_pos);
	mphysicalSystem.AddLink(motor);

	// Flap
	auto flap = chrono_types::make_shared<ChBodyEasyBox>(sizeHalfFlap.x() * 2, sizeHalfFlap.y() * 2, sizeHalfFlap.z() * 2, rhoSolid);
	flap->SetPos(posFlap);
	mphysicalSystem.AddBody(flap);

	myFsiSystem.AddFsiBody(flap); // Add to fsi system for interaction other than boundary
	fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, flap, chrono::VNULL, chrono::QUNIT, sizeHalfFlap, 13, true); // Set solid
	fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, flap, chrono::VNULL, chrono::QUNIT, sizeHalfFlap, -13, true, true); // Set solid

	// Fin, curved to positive y
	auto fin = chrono_types::make_shared<fea::ChMesh>();

	int numDivX = 10;
	int numDivY = 0;
	int numDivZ = 10;
	int numNodeX = numDivX + 1;
	int numNodeY = numDivY + 1;
	int numNodeZ = numDivZ + 1;
	int numElements = numDivX * (numDivY + 1) * numDivZ;
	int numNodes = numNodeX * numNodeY * numNodeZ;

	double dx = beamLength / numDivX;
	double dy = beamThickness;
	double dz = beamAngle / numDivZ; // rad
	double dzLength = beamRadius * sin(dz / 2) * 2;

	// Add nodes to mesh
	for (int k = 0; k < numNodeZ; k++) { // Loop order MATTERS!!
		for (int j = 0; j < numNodeY; j++) {
			for (int i = 0; i < numNodeX; i++) {
				// Location
				double x = i * dx + beamDx;
				double r = j * dy + beamRadius;
				double theta = k * dz + (CH_C_PI - beamAngle) / 2 + beamDAngle;
				double y = r * sin(theta) + beamDy;
				double z = r * cos(theta) + beamDz;

				// Direction
				double dirX = 0;
				double dirY = sin(theta);
				double dirZ = cos(theta);
				//printf("\n%f, %f, %f, %f\n", x, y, z, theta);

				// Node
				auto node = chrono_types::make_shared<ChNodeFEAxyzD>(ChVector<>(x, y, z), ChVector<>(dirX, dirY, dirZ));
				node->SetMass(0.0);

				// Constraint one end
				if (i == 0) {
					auto finJoint = chrono_types::make_shared<ChLinkPointFrameGeneric>(true, true, true);
					finJoint->Initialize(node, slider);
					mphysicalSystem.Add(finJoint);
				}

				if (i == numNodeX - 1) {
					auto flapJoint = chrono_types::make_shared<ChLinkPointFrameGeneric>(true, true, true);
					flapJoint->Initialize(node, flap);
					mphysicalSystem.Add(flapJoint);
				}

				fin->AddNode(node);
			}
		}
	}

	// Shell material
	auto shellMat = chrono_types::make_shared<ChMaterialShellANCF>(rhoSolid, E, nu); // rho, E, nu

	std::vector<std::vector<int>> elementsNodes; // Relate element to node
	std::vector<std::vector<int>> nodeNeighborElement; // Relate node to element
	elementsNodes.resize(numElements);
	nodeNeighborElement.resize(numNodes);
	int elementCount = 0;
	// Add elements to mesh
	for (int i = 0; i < numDivX; i++) {
		for (int j = 0; j < numDivY + 1; j++) {
			for (int k = 0; k < numDivZ; k++) {
				// Element CCW node index
				int nodeAIndex = i + k * numNodeX + j * numNodeX * numNodeZ;
				int nodeBIndex = i + k * numNodeX + j * numNodeX * numNodeZ + 1;
				int nodeCIndex = i + (k + 1) * numNodeX + j * numNodeX * numNodeZ + 1;
				int nodeDIndex = i + (k + 1) * numNodeX + j * numNodeX * numNodeZ;
				//printf("\n%d, %d, %d, %d, %d\n", elementCount, nodeAIndex, nodeBIndex, nodeCIndex, nodeDIndex);

				// Element
				auto element = chrono_types::make_shared<ChElementShellANCF>();
				element->SetNodes(
					std::dynamic_pointer_cast<ChNodeFEAxyzD>(fin->GetNode(nodeAIndex)),
					std::dynamic_pointer_cast<ChNodeFEAxyzD>(fin->GetNode(nodeBIndex)),
					std::dynamic_pointer_cast<ChNodeFEAxyzD>(fin->GetNode(nodeCIndex)),
					std::dynamic_pointer_cast<ChNodeFEAxyzD>(fin->GetNode(nodeDIndex))
				);
				element->SetDimensions(dx, dzLength);
				element->AddLayer(dy, 0 * CH_C_DEG_TO_RAD, shellMat);
				element->SetAlphaDamp(0.01);
				element->SetGravityOn(false);

				fin->AddElement(element);

				elementsNodes[elementCount].push_back(nodeAIndex);
				elementsNodes[elementCount].push_back(nodeBIndex);
				elementsNodes[elementCount].push_back(nodeCIndex);
				elementsNodes[elementCount].push_back(nodeDIndex);
				nodeNeighborElement[nodeAIndex].push_back(elementCount);
				nodeNeighborElement[nodeBIndex].push_back(elementCount);
				nodeNeighborElement[nodeCIndex].push_back(elementCount);
				nodeNeighborElement[nodeDIndex].push_back(elementCount);

				elementCount++;
			}
		}
	}
	mphysicalSystem.Add(fin);

	std::vector<std::vector<int>> elementsNodes1D;
	fsi::utils::AddBCE_FromMesh(
		myFsiSystem.GetDataManager(), paramsH, fin,
		myFsiSystem.GetFsiNodes(), myFsiSystem.GetFsiCables(), myFsiSystem.GetFsiShells(),
		nodeNeighborElement, elementsNodes1D, elementsNodes,
		false, true, false, true, 0, 0
	);
	myFsiSystem.SetShellElementsNodes(elementsNodes);
	myFsiSystem.SetFsiMesh(fin);

	// Construction of the FSI system must be finalized before running
	myFsiSystem.Finalize();

	// Solver
	auto mkl_solver = chrono_types::make_shared<ChSolverPardisoMKL>();
	mkl_solver->LockSparsityPattern(true);
	mphysicalSystem.SetSolver(mkl_solver);

	// Time stepper
	mphysicalSystem.SetTimestepperType(ChTimestepper::Type::HHT);
	auto mystepper = std::static_pointer_cast<ChTimestepperHHT>(mphysicalSystem.GetTimestepper());
	mystepper->SetStepControl(false);

	std::ostringstream meshConBuffer;
	calcBeamCon(meshConBuffer, fin);
	//return 0;

	// Start the simulation
	double time = 0;
	int frameCurrent = -1;
	double tFrame = 1.0 / paramsH->out_fps;
	while (time < paramsH->tFinal) {
		int frameNew = (int)(time / tFrame);
		if (frameNew > frameCurrent) {
			frameCurrent = frameNew;
			// Save outputs
			fsi::utils::PrintToFile(
				myFsiSystem.GetDataManager()->sphMarkersD2->posRadD,
				myFsiSystem.GetDataManager()->sphMarkersD2->velMasD,
				myFsiSystem.GetDataManager()->sphMarkersD2->rhoPresMuD,
				myFsiSystem.GetDataManager()->fsiGeneralData->sr_tau_I_mu_i,
				myFsiSystem.GetDataManager()->fsiGeneralData->referenceArray,
				myFsiSystem.GetDataManager()->fsiGeneralData->referenceArray_FEA,
				out_dir, true
			);

			writeBeamVtk(meshConBuffer, fin, frameCurrent);
			writeFlapVtk(flap, flapLength, flapThickness, flapHeight, frameCurrent);
			writeForce(motor, frameCurrent);
		}

		printf("\nstep: %f, time: %f, current frame: %d\n", paramsH->dT, time, frameCurrent);

		// Call the FSI solver
		try {
			myFsiSystem.DoStepDynamics_FSI();
		}
		catch (const std::exception& e) {
			printf("ERROR: %s\n", e.what());
			return 0;
		}
		time += paramsH->dT;
	}

	return 0;
}
