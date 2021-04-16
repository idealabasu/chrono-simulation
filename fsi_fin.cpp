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

const std::string outs_dir = "FSI_FIN/"; // Directory for outputs
std::string out_dir; // Directory for this simulation

// Parameters
double M_TO_L = 1e1; // dm
double KG_TO_W = 1e3; // gram

// Simulation domain dimension
Real bxDim = 0.1 * M_TO_L;
Real byDim = 0.1 * M_TO_L;
Real bzDim = 0.1 * M_TO_L;

// Fluid domain dimension
Real fxDim = 0.13 * M_TO_L;
Real fyDim = 0.15 * M_TO_L;
Real fzDim = 0.1 * M_TO_L;

// Solid materials
double E = 1e8 * KG_TO_W / M_TO_L;
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
double ts = 0.2; // Wait for fluid to settle
double amplitude = 0.05 * M_TO_L;

double wallOffset = 0.05 * M_TO_L;
double zOffset = 0.01 * M_TO_L;

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
	printf("\n%f, %f, %f\n", totalForce.x(), totalForce.y(), totalForce.z());

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
	std::string inputJson = "fsi_fin.json";

	if (!fsi::utils::ParseJSON(inputJson, paramsH, fsi::mR3(bxDim, byDim, bzDim))) {
		printf("Invalid json\n");
		return 1;
	}

	// Space between particles
	Real space = paramsH->MULT_INITSPACE * paramsH->HSML;

	// Make boundary domain large
	paramsH->cMin = chrono::fsi::mR3(-bxDim / 2, -byDim / 2, -bzDim / 2) * 10;
	paramsH->cMax = chrono::fsi::mR3(bxDim / 2, byDim / 2, bzDim / 2) * 10;

	// Finalize params and output dir
	myFsiSystem.SetFluidDynamics(paramsH->fluid_dynamic_type);
	myFsiSystem.SetFluidSystemLinearSolver(paramsH->LinearSolver);

	fsi::utils::FinalizeDomain(paramsH);
	fsi::utils::PrepareOutputDir(paramsH, out_dir, outs_dir, inputJson);

	//double beamAngle = beamArcLength / beamRadius;
	double flapThickness = beamRadius - beamRadius * cos(beamAngle / 2);
	double beamDx = -(beamLength + flapLength) / 2;
	double beamDy = -amplitude - beamRadius + flapThickness / 2;
	double beamDAngle = 0.0;
	double beamDz = fzDim / 2 - zOffset; // Offset  midpoint of arc to center of container

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
	ChVector<> boxCenter(0.0, 0.0, fzDim / 2);
	ChVector<> boxHalfDim(fxDim / 2, fyDim / 2, fzDim / 2);
	utils::GridSampler<> sampler(space);
	utils::Generator::PointVector points = sampler.SampleBox(boxCenter, boxHalfDim); // Use a chrono sampler to create a bucket of points

	// Add fluid particles from the sampler points to the FSI system
	int numPartAdded = 0;
	for (int i = 0; i < points.size(); i++) {
		// Calculate the pressure of a steady state (p = rho*g*h)
		Real pre_ini = paramsH->rho0 * abs(paramsH->gravity.z) * (-points[i].z() + fzDim);
		// Real rho_ini = paramsH->rho0 + pre_ini / (paramsH->Cs * paramsH->Cs);
		// Real pre_ini = paramsH->BASEPRES;
		Real rho_ini = paramsH->rho0;

		// Skip if too close to body
		// Beam
		bool noContact = true;
		ChVector<> pt = points[i] - beamA1;
		Real dot = pt ^ beamA; // Project length along beam axis
		Real cross = (pt % beamA).Length(); // Distance to beam beam axis
		double angle = atan2(pt.y(), pt.z());
		if (
			dot <= beamLength + space && // Length
			dot >= 0 - space &&
			cross <= beamRadius + space && // Ring
			cross >= beamRadius - space &&
			angle >= (CH_C_PI - beamAngle) / 2 + beamDAngle - space / beamRadius && // Arc
			angle <= (CH_C_PI + beamAngle) / 2 + beamDAngle + space / beamRadius
			) {
			//printf("\n%f, %f, %f\n", pt.x(), pt.y(), pt.z());
			//printf("\n%f, %f\n", dot, cross);
			noContact = false;
		}
		// Flap
		if (
			points[i].x() <= posFlap.x() + sizeHalfFlap.x() + space && // Length
			points[i].x() >= posFlap.x() - sizeHalfFlap.x() - space &&
			points[i].y() <= posFlap.y() + sizeHalfFlap.y() + space && // Thickness
			points[i].y() >= posFlap.y() - sizeHalfFlap.y() - space &&
			points[i].z() <= posFlap.z() + sizeHalfFlap.z() + space && // Height
			points[i].z() >= posFlap.z() - sizeHalfFlap.z() - space
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
	//ChVector<> gravity = ChVector<>(paramsH->gravity.x, paramsH->gravity.y, paramsH->gravity.z);
	ChVector<> gravity = ChVector<>(0.0, 0.0, 0.0);
	mphysicalSystem.Set_G_acc(gravity);

	// Set common material Properties
	//auto surfMat = chrono_types::make_shared<ChMaterialSurfaceSMC>();
	//surfMat->SetYoungModulus(1e8);
	//surfMat->SetFriction(0.2f);
	//surfMat->SetRestitution(0.05f);
	//surfMat->SetAdhesion(0);

	// Container body
	auto container = chrono_types::make_shared<ChBody>();
	container->SetIdentifier(-1);
	container->SetBodyFixed(true);
	//container->SetCollide(true);
	//container->GetCollisionModel()->ClearModel();

	// Container wall geometry
	// Wall should be 1 space away from the boundary particles
	ChVector<> sizeHalfBtm(fxDim / 2 + space * 3, fyDim / 2 + space * 3, space * 2);
	ChVector<> posBtm(0, 0, -sizeHalfBtm.z() - space);

	ChVector<> sizeHalfLeft(space * 2, fyDim / 2, fzDim / 2 + wallOffset);
	ChVector<> posLeft(-fxDim / 2 - sizeHalfLeft.x() - space, 0, sizeHalfLeft.z());
	ChVector<> posRight(-posLeft.x(), posLeft.y(), posLeft.z());

	ChVector<> sizeHalfBack(fxDim / 2 + space * 3, space * 2, fzDim / 2 + wallOffset);
	ChVector<> posBack(0, -fyDim / 2 - sizeHalfBack.y() - space, sizeHalfBack.z());
	ChVector<> posFront(posBack.x(), -posBack.y(), posBack.z());

	// Add wall collision geometry 
	//chrono::utils::AddBoxGeometry(container.get(), surfMat, sizeHalfBtm, posBtm, chrono::QUNIT, true);
	//chrono::utils::AddBoxGeometry(container.get(), surfMat, sizeHalfLeft, posLeft, chrono::QUNIT, true);
	//chrono::utils::AddBoxGeometry(container.get(), surfMat, sizeHalfLeft, posRight, chrono::QUNIT, true);
	//chrono::utils::AddBoxGeometry(container.get(), surfMat, sizeHalfBack, posBack, chrono::QUNIT, true);
	//chrono::utils::AddBoxGeometry(container.get(), surfMat, sizeHalfBack, posFront, chrono::QUNIT, true);

	// Only add BCE on specified box surface, 12 means top, -12 means bottom, default 3 layers inward
	fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, container, posBtm, chrono::QUNIT, sizeHalfBtm, 12);
	fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, container, posLeft, chrono::QUNIT, sizeHalfLeft, 23);
	fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, container, posRight, chrono::QUNIT, sizeHalfLeft, -23);
	fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, container, posBack, chrono::QUNIT, sizeHalfBack, 13);
	fsi::utils::AddBoxBce(myFsiSystem.GetDataManager(), paramsH, container, posFront, chrono::QUNIT, sizeHalfBack, -13);

	mphysicalSystem.AddBody(container);

	// Slider
	auto slider = chrono_types::make_shared<ChBodyEasyBox>(0.1, 0.1, 0.1, rhoSolid);
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

	std::vector<std::shared_ptr<fea::ChLinkPointFrameGeneric>> sliderFinJoints;

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
				node->SetMass(0);

				// Constraint one end
				if (i == 0) {
					auto finJoint = chrono_types::make_shared<ChLinkPointFrameGeneric>(true, true, true);
					finJoint->Initialize(node, slider);
					mphysicalSystem.Add(finJoint);
					sliderFinJoints.push_back(finJoint);
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
	Real time = 0;
	int stepEnd = int(paramsH->tFinal / paramsH->dT);
	int stepSave = int(1.0 / paramsH->out_fps / paramsH->dT);
	for (int tStep = 0; tStep < stepEnd + 1; tStep++) {
		int frameCurrent = int(tStep / stepSave);
		printf("\nstep: %d, time: %f (s) current frame: %d\n", tStep, time, frameCurrent);
		//printf("\n%f, %f, %f\n", slider->GetPos().x(), slider->GetPos().y(), slider->GetPos().z());

		if (tStep % stepSave == 0) {
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

		// Call the FSI solver
		try {
			myFsiSystem.DoStepDynamics_FSI();
		}
		catch (const std::exception& e) {
			printf("%s\n", e.what());
			return 0;
		}
		time += paramsH->dT;
	}

	return 0;
}
